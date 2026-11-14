"""End-to-end pipeline: page images -> study song (wav + lyrics)."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

from mistralai.client import Mistral

from learning_song.backing import SR, build_backing, fade
from learning_song.config import Config
from learning_song.lyrics import write_lyrics
from learning_song.render import INTRO_BARS, OUTRO_BARS, render_song, write_wav
from learning_song.vision import extract_page_content
from learning_song.vocals import synthesize_lines

ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def generate_song(
    images: list[tuple[bytes, str | None]],
    output_dir: str,
    subject_hint: str = "",
    instrumental: bool = False,
    config: Config | None = None,
    progress=None,
) -> dict:
    """Run the full pipeline and return metadata about the generated song."""
    config = config or Config.from_env()
    if not images:
        raise ValueError("At least one page image is required.")
    if len(images) > config.max_images:
        raise ValueError(f"At most {config.max_images} images can be processed at once.")

    client = Mistral(api_key=config.api_key)

    def report(stage: str, detail: str = "") -> None:
        if progress:
            progress(stage, detail)

    report("reading", "Reading your pages with the vision model…")
    content = extract_page_content(client, config.vision_model, images)

    report("writing", "Writing lyrics…")
    song = write_lyrics(
        client,
        config.lyrics_model,
        content,
        target_seconds=config.target_seconds,
        max_seconds=config.max_seconds,
        subject_hint=subject_hint,
    )

    song_id = uuid.uuid4().hex[:12]
    song_dir = os.path.join(output_dir, song_id)
    os.makedirs(song_dir, exist_ok=True)

    vocals_ok = False
    clips: list[list] = [[] for _ in song.sections]
    if not instrumental:
        report("singing", f"Recording vocals ({len(song.all_lines)} lines)…")
        flat = song.all_lines
        rendered = synthesize_lines(
            client,
            config.tts_model,
            flat,
            config.voice_id,
            progress=lambda: None,
        )
        vocals_ok = any(clip is not None for clip in rendered)
        index = 0
        for i, section in enumerate(song.sections):
            clips[i] = rendered[index: index + len(section.lines)]
            index += len(section.lines)

    report("producing", "Producing the backing track and mixing…")
    if vocals_ok:
        rendered_song = render_song(song, clips)
        audio = rendered_song.audio
        timed_lyrics = rendered_song.timed_lyrics
        duration = rendered_song.duration
    else:
        beat = 60.0 / song.bpm
        n_lines = max(len(song.all_lines), 1)
        total = (INTRO_BARS + OUTRO_BARS) * 4 * beat + n_lines * 3.5
        audio = fade(build_backing(total, song.bpm, song.key))
        timed_lyrics = []
        duration = total

    wav_path = os.path.join(song_dir, "song.wav")
    write_wav(wav_path, audio, config.sample_rate)

    meta = {
        "id": song_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": song.title,
        "genre": song.genre,
        "bpm": song.bpm,
        "key": song.key,
        "duration_seconds": round(duration, 1),
        "has_vocals": vocals_ok,
        "subject_hint": subject_hint,
        "sections": [
            {"name": s.name, "lines": s.lines} for s in song.sections
        ],
        "timed_lyrics": timed_lyrics,
    }
    meta_path = os.path.join(song_dir, "song.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    report("done", "Song ready!")
    return meta
