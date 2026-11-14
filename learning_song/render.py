"""Place vocal clips on a timeline, build the backing track and mix the final song."""

from __future__ import annotations

import math
import wave

import numpy as np

from learning_song.backing import SR, build_backing, fade
from learning_song.lyrics import Song

LINE_GAP = 0.25
SECTION_GAP = 0.9
INTRO_BARS = 4
OUTRO_BARS = 4


class RenderedSong:
    def __init__(self, audio: np.ndarray, timed_lyrics: list[dict], duration: float):
        self.audio = audio
        self.timed_lyrics = timed_lyrics
        self.duration = duration


def _build_timeline(
    song: Song, clips: list[list[np.ndarray | None]]
) -> tuple[list[tuple[int, np.ndarray]], list[dict], float]:
    beat = 60.0 / song.bpm
    bar = 4.0 * beat
    placements: list[tuple[int, np.ndarray]] = []
    timed: list[dict] = []
    cursor = INTRO_BARS * bar

    for section, section_clips in zip(song.sections, clips):
        section_start = math.ceil(cursor / beat) * beat
        line_cursor = section_start
        for line, clip in zip(section.lines, section_clips):
            if clip is None or len(clip) == 0:
                line_cursor += LINE_GAP
                continue
            placements.append((int(line_cursor * SR), clip))
            timed.append(
                {
                    "section": section.name,
                    "line": line,
                    "start": round(line_cursor, 2),
                    "end": round(line_cursor + len(clip) / SR, 2),
                }
            )
            line_cursor += len(clip) / SR + LINE_GAP
        cursor = line_cursor + SECTION_GAP

    total = cursor + OUTRO_BARS * bar
    return placements, timed, total


def render_song(song: Song, clips: list[list[np.ndarray | None]]) -> RenderedSong:
    placements, timed, total = _build_timeline(song, clips)

    chorus_spans = []
    for entry in timed:
        if "chorus" in entry["section"]:
            chorus_spans.append((entry["start"], entry["end"]))
    if not chorus_spans and timed:
        chorus_spans = [(timed[0]["start"], timed[-1]["end"])]

    backing = build_backing(total, song.bpm, song.key, chorus_spans)
    vocals = np.zeros_like(backing)
    for start_sample, clip in placements:
        end = min(len(vocals), start_sample + len(clip))
        vocals[start_sample:end] += clip[: end - start_sample]

    mixed = backing * 0.55 + vocals * 1.25
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0:
        mixed = mixed / peak * 0.92
    mixed = fade(mixed)

    return RenderedSong(mixed, timed, total)


def write_wav(path: str, audio: np.ndarray, sample_rate: int = SR) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
