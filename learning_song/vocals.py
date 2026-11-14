"""Vocal rendering through the Mistral speech API."""

from __future__ import annotations

import base64
import io
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from mistralai.client import Mistral

from learning_song.backing import SR


def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def resample(audio: np.ndarray, src_rate: int, dst_rate: int = SR) -> np.ndarray:
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32)
    n_dst = int(round(len(audio) * dst_rate / src_rate))
    if n_dst == 0:
        return np.zeros(0, dtype=np.float32)
    positions = np.linspace(0, len(audio) - 1, n_dst)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def _strip_silence(audio: np.ndarray, threshold: float = 0.004) -> np.ndarray:
    above = np.where(np.abs(audio) > threshold)[0]
    if len(above) == 0:
        return audio
    return audio[above[0]: above[-1] + 1]


def synthesize_line(
    client: Mistral,
    model: str,
    text: str,
    voice_id: str | None,
) -> np.ndarray:
    kwargs: dict = {
        "input": text,
        "response_format": "wav",
    }
    if model:
        kwargs["model"] = model
    if voice_id:
        kwargs["voice_id"] = voice_id
    response = client.audio.speech.complete(**kwargs)
    audio_b64 = response.audio_data
    if isinstance(audio_b64, str):
        raw = base64.b64decode(audio_b64)
    else:
        raw = bytes(audio_b64)
    audio, rate = _decode_wav(raw)
    return resample(_strip_silence(audio), rate)


def synthesize_lines(
    client: Mistral,
    model: str,
    lines: list[str],
    voice_id: str | None,
    max_workers: int = 4,
    progress=None,
) -> list[np.ndarray | None]:
    results: list[np.ndarray | None] = [None] * len(lines)

    def work(index: int, line: str) -> None:
        try:
            results[index] = synthesize_line(client, model, line, voice_id)
        except Exception:
            results[index] = None
        finally:
            if progress:
                progress()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, i, line) for i, line in enumerate(lines)]
        for future in futures:
            future.result()
    return results
