"""Command-line interface: python -m learning_song.cli page1.jpg page2.jpg -o song.wav"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys

from learning_song.config import Config
from learning_song.pipeline import generate_song


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="learning_song",
        description="Turn pictures of book pages into a study song.",
    )
    parser.add_argument("images", nargs="+", help="Paths to page photos")
    parser.add_argument("-o", "--output", default="song.wav", help="Output wav path")
    parser.add_argument("--meta", help="Where to write the song.json metadata")
    parser.add_argument("--subject", default="", help="Optional topic hint")
    parser.add_argument("--instrumental", action="store_true", help="Skip vocals (TTS)")
    parser.add_argument("--voice", default=None, help="Mistral voice id")
    args = parser.parse_args(argv)

    images: list[tuple[bytes, str | None]] = []
    for path in args.images:
        if not os.path.isfile(path):
            print(f"error: {path} is not a file", file=sys.stderr)
            return 1
        with open(path, "rb") as fh:
            images.append((fh.read(), mimetypes.guess_type(path)[0]))

    config = Config.from_env()
    if args.voice:
        config = Config(
            api_key=config.api_key,
            vision_model=config.vision_model,
            lyrics_model=config.lyrics_model,
            tts_model=config.tts_model,
            voice_id=args.voice,
        )

    def progress(stage: str, detail: str) -> None:
        print(f"[{stage}] {detail}")

    meta = generate_song(
        images,
        output_dir=os.path.dirname(os.path.abspath(args.output)) or ".",
        subject_hint=args.subject,
        instrumental=args.instrumental,
        config=config,
        progress=progress,
    )

    generated = os.path.join(
        os.path.dirname(os.path.abspath(args.output)) or ".", meta["id"], "song.wav"
    )
    if os.path.abspath(generated) != os.path.abspath(args.output):
        os.replace(generated, args.output)
    if args.meta:
        import json

        with open(args.meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"\"{meta['title']}\" – {meta['duration_seconds']}s, "
          f"{meta['bpm']} bpm, {meta['key']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
