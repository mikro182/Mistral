import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module


def test_index():
    client = app_module.app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"PageBeats" in response.data


def test_generate_requires_images():
    client = app_module.app.test_client()
    response = client.post("/generate", data={})
    assert response.status_code == 302
    assert "error=" in response.headers["Location"]


def test_generate_rejects_non_image():
    client = app_module.app.test_client()
    data = {"pages": (io.BytesIO(b"not an image"), "page.txt", "text/plain")}
    response = client.post(
        "/generate", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 302
    assert "error=" in response.headers["Location"]


def test_song_404():
    client = app_module.app.test_client()
    response = client.get("/song/doesnotexist")
    assert response.status_code == 404


def test_song_page_from_meta():
    with app_module.app.test_request_context():
        outdir = app_module.app.config["OUTPUT_DIR"]
        song_id = "testsong"
        os.makedirs(os.path.join(outdir, song_id), exist_ok=True)
        meta = {
            "id": song_id,
            "title": "Test Song",
            "genre": "pop",
            "bpm": 100,
            "key": "C major",
            "duration_seconds": 12.0,
            "has_vocals": True,
            "subject_hint": "",
            "sections": [{"name": "verse 1", "lines": ["hello"]}],
            "timed_lyrics": [],
        }
        import json

        with open(os.path.join(outdir, song_id, "song.json"), "w") as fh:
            json.dump(meta, fh)
        client = app_module.app.test_client()
        response = client.get(f"/song/{song_id}")
        assert response.status_code == 200
        assert b"Test Song" in response.data


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
