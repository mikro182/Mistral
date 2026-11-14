import json
import os
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning_song.backing import SR, build_backing, fade
from learning_song.lyrics import Song
from learning_song.render import render_song, write_wav


def _clip(seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_song_parsing():
    song = Song.from_dict(
        {
            "title": "Photosynthesis",
            "genre": "pop",
            "bpm": 300,
            "key": "A minor",
            "sections": [
                {"name": "Intro", "lines": ["here we go"]},
                {"name": "Chorus", "lines": ["light into sugar", "oxygen for me"]},
            ],
        }
    )
    assert song.title == "Photosynthesis"
    assert song.bpm == 180
    assert song.sections[0].name == "intro"
    assert len(song.all_lines) == 3


def test_backing_duration():
    audio = build_backing(10.0, 100, "C major")
    assert len(audio) == int(10.0 * SR)
    assert np.max(np.abs(audio)) <= 1.0


def test_render_and_wav():
    song = Song.from_dict(
        {
            "title": "Test",
            "genre": "pop",
            "bpm": 96,
            "key": "C major",
            "sections": [
                {"name": "verse 1", "lines": ["line one", "line two"]},
                {"name": "chorus", "lines": ["chorus line"]},
            ],
        }
    )
    clips = [
        [_clip(1.0), _clip(1.2)],
        [_clip(0.8)],
    ]
    rendered = render_song(song, clips)
    assert len(rendered.audio) == int(rendered.duration * SR)
    assert rendered.duration > 4 * 4 * 60 / 96
    assert len(rendered.timed_lyrics) == 3
    assert rendered.timed_lyrics[0]["start"] > 0

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "song.wav")
        write_wav(path, rendered.audio)
        with wave.open(path, "rb") as wav:
            assert wav.getframerate() == SR
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            frames = wav.getnframes()
        assert abs(frames / SR - rendered.duration) < 0.05


def test_render_instrumental_fallback():
    song = Song.from_dict(
        {
            "title": "Test",
            "genre": "pop",
            "bpm": 90,
            "key": "C major",
            "sections": [
                {"name": "verse 1", "lines": ["only line"]},
                {"name": "chorus", "lines": ["hook"]},
            ],
        }
    )
    rendered = render_song(song, [[None], [None]])
    assert len(rendered.timed_lyrics) == 0
    assert rendered.duration > 0


def test_fade():
    audio = np.ones(SR * 3, dtype=np.float32)
    faded = fade(audio, 0.5, 1.0)
    assert faded[0] == 0.0
    assert abs(faded[-1]) < 1e-6
    assert faded[SR] == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
