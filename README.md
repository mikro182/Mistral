# 🎸 PageBeats – learn with songs

Send pictures of a page in a book you want to learn, and the AI turns it into a
study song: it reads the pages with a vision model, writes catchy lyrics that
contain the important facts, records vocals with a text-to-speech model and
produces an instrumental backing track to sing them over.

A typical page becomes a song of about **2 minutes**; denser pages (or several
pages at once) automatically get longer songs with more verses.

## How it works

1. **Read** – the page photos go through Mistral's vision model, which
   transcribes all text and describes diagrams.
2. **Write** – a chat model turns the content into a structured song
   (title, genre, BPM, key, sections, lines) sized to the amount of content.
3. **Sing** – every line is rendered as vocals with the Mistral speech API.
4. **Produce** – a small synthesizer (numpy) builds a chord/bass/drum backing
   track in the song's key and tempo, and the vocals are mixed on top.

If the speech model is unavailable, the app still produces an instrumental
version and shows the lyrics.

## Setup

```bash
pip install -r requirements.txt
export MISTRAL_API_KEY="your-api-key"   # https://console.mistral.ai/
```

## Web app

```bash
python app.py
```

Then open <http://127.0.0.1:5000>, add up to 10 page photos, optionally say what
you are studying, and press **Make my song**. Generation takes 1–3 minutes;
afterwards you can play and download the song and read along with the lyrics.

## Command line

```bash
python -m learning_song.cli page1.jpg page2.jpg -o my_song.wav --subject "Photosynthesis"
```

Options: `--subject` topic hint, `--voice` Mistral voice id,
`--instrumental` skip vocals, `--meta song.json` also write the metadata.

## Configuration (environment variables)

| Variable           | Default                | Purpose                          |
| ------------------ | ---------------------- | -------------------------------- |
| `MISTRAL_API_KEY`  | – (required)           | Mistral API key                  |
| `LS_VISION_MODEL`  | `pixtral-12b-2409`     | Model that reads the pages       |
| `LS_LYRICS_MODEL`  | `mistral-small-latest` | Model that writes the lyrics     |
| `LS_TTS_MODEL`     | `mistral-tts-latest`   | Model that sings the lines       |
| `LS_VOICE_ID`      | server default         | Voice used for vocals            |
| `LS_OUTPUT_DIR`    | `outputs/`             | Where generated songs are stored |
| `LS_PORT`          | `5000`                 | Web app port                     |

## Notes

- Songs are stored under `outputs/<id>/` as `song.wav` plus a `song.json` with
  the lyrics and timings.
- The backing track is a lightweight synthesized demo arrangement; swap
  `learning_song/backing.py` for a real music-generation API if you want
  full production quality.

---

The original minimal Mistral chat example is still available in `main.py`.
