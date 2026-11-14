"""Tiny synthesizer that builds an instrumental backing track with numpy."""

from __future__ import annotations

import math

import numpy as np

SR = 22050

NOTE_OFFSETS = {
    "c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5,
    "f#": 6, "gb": 6, "g": 7, "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10,
    "b": 11,
}

MAJOR_PROGRESSION = [(0, "maj"), (5, "maj"), (7, "maj"), (5, "maj")]
MINOR_PROGRESSION = [(0, "min"), (5, "min"), (10, "maj"), (7, "maj")]
TRIADS = {"maj": (0, 4, 7), "min": (0, 3, 7)}


def parse_key(key: str) -> tuple[int, bool]:
    text = key.strip().lower()
    root_name = text.split()[0] if text.split() else "c"
    minor = "min" in text
    root = NOTE_OFFSETS.get(root_name, 0)
    return root, minor


def _freq(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _sine(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    return np.sin(2.0 * np.pi * freq * t)


def _env(n: int, attack: float, release: float) -> np.ndarray:
    a = max(1, int(attack * SR))
    r = max(1, int(release * SR))
    envelope = np.ones(n)
    envelope[:a] = np.linspace(0.0, 1.0, a)
    envelope[-r:] *= np.linspace(1.0, 0.0, r)
    return envelope


def _pad_chord(midis: list[int], n: int) -> np.ndarray:
    chord = np.zeros(n)
    for midi in midis:
        f = _freq(midi)
        chord += _sine(f, n) + 0.35 * _sine(f * 2.0, n)
    chord /= len(midis) * 1.35
    return chord * _env(n, 0.05, 0.25)


def _bass_note(midi: int, n: int) -> np.ndarray:
    f = _freq(midi)
    tone = _sine(f, n) + 0.3 * _sine(f * 2.0, n) + 0.15 * _sine(f * 0.5, n)
    return tone * _env(n, 0.005, 0.12) * 0.9


def _kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    sweep = np.exp(-t * 22.0) * np.sin(2.0 * np.pi * (95.0 * np.exp(-t * 18.0) + 42.0) * t)
    return sweep * 1.1


def _hat(n: int) -> np.ndarray:
    noise = np.random.default_rng(n).uniform(-1.0, 1.0, n)
    return noise * np.exp(-np.arange(n) / (0.008 * SR)) * 0.16


def _snare(n: int) -> np.ndarray:
    rng = np.random.default_rng(n + 1)
    noise = rng.uniform(-1.0, 1.0, n) * np.exp(-np.arange(n) / (0.05 * SR))
    tone = _sine(190.0, n) * np.exp(-np.arange(n) / (0.03 * SR))
    return (noise * 0.5 + tone * 0.4) * 0.5


def _add(buffer: np.ndarray, clip: np.ndarray, start: int) -> None:
    end = min(len(buffer), start + len(clip))
    if start < len(buffer):
        buffer[start:end] += clip[: end - start]


def build_backing(
    duration: float,
    bpm: int,
    key: str,
    chorus_spans: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Render a mono backing track of `duration` seconds."""
    chorus_spans = chorus_spans or []
    beat = 60.0 / bpm
    bar = 4.0 * beat
    n_total = int(duration * SR)
    mix = np.zeros(n_total)

    root, minor = parse_key(key)
    progression = MINOR_PROGRESSION if minor else MAJOR_PROGRESSION
    n_bars = int(math.ceil(duration / bar))

    for i in range(n_bars):
        bar_start = int(i * bar * SR)
        degree, quality = progression[i % len(progression)]
        chord_root = 48 + root + degree
        intervals = TRIADS[quality]
        midis = [chord_root + iv for iv in intervals]

        in_chorus = any(
            i * bar < span_end and (i + 1) * bar > span_start
            for span_start, span_end in chorus_spans
        )

        _add(mix, _pad_chord(midis, int(bar * SR)) * (0.30 if in_chorus else 0.22), bar_start)

        for b in range(4):
            beat_start = bar_start + int(b * beat * SR)
            _add(mix, _bass_note(chord_root - 24 + intervals[0], int(beat * SR)) * 0.55, beat_start)
            eighth = int(beat * SR / 2)
            if b % 2 == 0:
                _add(mix, _kick(int(0.15 * SR)) * 0.8, beat_start)
            else:
                _add(mix, _snare(int(0.18 * SR)) * (0.9 if in_chorus else 0.6), beat_start)
            _add(mix, _hat(eighth), beat_start + eighth)

        if in_chorus:
            for s in range(8):
                note = midis[s % 3] + 12
                start = bar_start + int(s * beat * SR / 2)
                _add(mix, _sine(_freq(note), int(0.18 * SR)) * _env(int(0.18 * SR), 0.005, 0.1) * 0.18, start)

    mix = np.tanh(mix * 1.2) * 0.85
    return mix


def fade(buffer: np.ndarray, fade_in: float = 0.4, fade_out: float = 2.5) -> np.ndarray:
    result = buffer.copy()
    a = min(len(result) // 2, int(fade_in * SR))
    r = min(len(result) // 2, int(fade_out * SR))
    if a > 0:
        result[:a] *= np.linspace(0.0, 1.0, a)
    if r > 0:
        result[-r:] *= np.linspace(1.0, 0.0, r)
    return result
