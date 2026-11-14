"""Web app: upload pictures of book pages, get a study song back.

Run with: python app.py  (then open http://127.0.0.1:5000)
"""

from __future__ import annotations

import mimetypes
import os
import uuid

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    send_from_directory,
    url_for,
    request,
)

from learning_song.config import Config
from learning_song.pipeline import ALLOWED_MIMES, generate_song

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["OUTPUT_DIR"] = os.environ.get("LS_OUTPUT_DIR", "outputs")

app.secret_key = os.environ.get("LS_SECRET_KEY", uuid.uuid4().hex)

ERRORS = {
    1: "Please add at least one picture of a page.",
    2: "Please upload image files (JPEG, PNG, WebP or GIF).",
    3: "Too many images. The app handles at most 10 pages at a time.",
}


@app.get("/")
def index():
    return render_template("index.html", error=request.args.get("error"))


@app.post("/generate")
def generate():
    uploads = [f for f in request.files.getlist("pages") if f and f.filename]
    if not uploads:
        return redirect(url_for("index", error=ERRORS[1]))
    if len(uploads) > 10:
        return redirect(url_for("index", error=ERRORS[3]))
    for upload in uploads:
        mimetype = upload.mimetype or mimetypes.guess_type(upload.filename)[0]
        if mimetype not in ALLOWED_MIMES:
            return redirect(url_for("index", error=ERRORS[2]))

    images = [(upload.read(), upload.mimetype or mimetypes.guess_type(upload.filename)[0]) for upload in uploads]
    subject = (request.form.get("subject") or "").strip()

    try:
        config = Config.from_env()
        meta = generate_song(
            images,
            output_dir=app.config["OUTPUT_DIR"],
            subject_hint=subject,
            config=config,
        )
    except ValueError as exc:
        return redirect(url_for("index", error=str(exc)))
    except Exception:
        app.logger.exception("Song generation failed")
        return redirect(
            url_for(
                "index",
                error="Song generation failed. Check the server logs and your MISTRAL_API_KEY.",
            )
        )
    return redirect(url_for("song", song_id=meta["id"]))


@app.get("/song/<song_id>")
def song(song_id: str):
    meta = _load_meta(song_id)
    if meta is None:
        abort(404)
    return render_template("song.html", song=meta)


@app.get("/outputs/<song_id>/<path:filename>")
def outputs(song_id: str, filename: str):
    directory = os.path.join(app.config["OUTPUT_DIR"], song_id)
    if not os.path.isdir(directory):
        abort(404)
    return send_from_directory(directory, filename)


def _load_meta(song_id: str) -> dict | None:
    path = os.path.join(app.config["OUTPUT_DIR"], song_id, "song.json")
    if not os.path.isfile(path):
        return None
    import json

    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("LS_PORT", "5000")), debug=False)
