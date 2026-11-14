import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    api_key: str
    vision_model: str = "pixtral-12b-2409"
    lyrics_model: str = "mistral-small-latest"
    tts_model: str = "mistral-tts-latest"
    voice_id: str | None = None
    sample_rate: int = 22050
    target_seconds: float = 120.0
    max_seconds: float = 300.0
    max_images: int = 10

    @classmethod
    def from_env(cls) -> "Config":
        try:
            api_key = os.environ["MISTRAL_API_KEY"]
        except KeyError as exc:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Get a key at https://console.mistral.ai/ "
                "and export it: export MISTRAL_API_KEY='...'"
            ) from exc
        return cls(
            api_key=api_key,
            vision_model=os.environ.get("LS_VISION_MODEL", cls.vision_model),
            lyrics_model=os.environ.get("LS_LYRICS_MODEL", cls.lyrics_model),
            tts_model=os.environ.get("LS_TTS_MODEL", cls.tts_model),
            voice_id=os.environ.get("LS_VOICE_ID") or None,
        )
