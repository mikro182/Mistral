"""Structured song data and the lyrics-writing step."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from mistralai.client import Mistral


@dataclass
class Section:
    name: str
    lines: list[str] = field(default_factory=list)


@dataclass
class Song:
    title: str
    genre: str
    bpm: int
    key: str
    sections: list[Section]

    @classmethod
    def from_dict(cls, data: dict) -> "Song":
        sections = [
            Section(name=str(s["name"]).strip().lower(), lines=[str(l) for l in s["lines"]])
            for s in data["sections"]
            if s.get("lines")
        ]
        if len(sections) < 2:
            raise ValueError("Song needs at least two sections.")
        bpm = int(data["bpm"])
        if not 60 <= bpm <= 180:
            bpm = min(max(bpm, 60), 180)
        return cls(
            title=str(data["title"]).strip() or "Study Song",
            genre=str(data.get("genre", "pop")).strip() or "pop",
            bpm=bpm,
            key=str(data.get("key", "C major")).strip() or "C major",
            sections=sections,
        )

    @property
    def all_lines(self) -> list[str]:
        return [line for section in self.sections for line in section.lines]

    def to_dict(self) -> dict:
        return asdict(self)


LYRICS_PROMPT = """You are a songwriter who turns study material into catchy educational songs.

Below is the content of one or more pages from a student's book.

<content>
{content}
</content>

{hint}Write a song that teaches this content. Rules:
- The song must include every important fact, term, definition, number or relationship from the content.
- Lines must be short and singable (5-10 words), with rhymes where natural.
- Repeat the most important facts in the chorus so they stick.
- Use a simple {genre} style in {key}.
- Structure the song as sections named "intro", "verse N", "chorus", "bridge", "outro".
- Length: {target_str} (~24-30 lines total) for a typical page.
- If the content is very dense or covers many pages, add more verses and a bridge so the song can run up to {max_minutes} minutes. Do not drop facts to keep it short.
- The intro and outro may be short (1-2 lines or a repeated hook).

Respond with ONLY a JSON object in this exact shape:
{{
  "title": "song title",
  "genre": "e.g. pop, rap, folk, rock",
  "bpm": integer between 70 and 140,
  "key": "e.g. C major, A minor",
  "sections": [
    {{"name": "intro", "lines": ["..."]}},
    {{"name": "verse 1", "lines": ["...", "..."]}}
  ]
}}"""


def _parse_song(raw: str) -> Song:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    return Song.from_dict(data)


def _target_str(seconds: float) -> str:
    minutes = int(seconds // 60)
    rest = int(round(seconds % 60))
    if minutes == 0:
        return f"about {rest} seconds"
    if rest == 0:
        return f"about {minutes} minutes"
    return f"about {minutes} minutes {rest} seconds"


def write_lyrics(
    client: Mistral,
    model: str,
    content: str,
    target_seconds: float,
    max_seconds: float,
    subject_hint: str = "",
) -> Song:
    hint = f'The student says the topic is: "{subject_hint}".\n' if subject_hint else ""
    prompt = LYRICS_PROMPT.format(
        content=content,
        hint=hint,
        genre="pop",
        key="a common key",
        target_str=_target_str(target_seconds),
        max_minutes=max(1, int(round(max_seconds / 60))),
    )

    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    return _parse_song(raw)
