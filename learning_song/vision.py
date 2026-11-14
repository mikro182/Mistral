import base64
import mimetypes

from mistralai.client import Mistral

TRANSCRIBE_PROMPT = (
    "You are helping a student learn from a photo of a book page. "
    "Transcribe ALL readable text from the image verbatim, preserving headings, "
    "lists and structure. For diagrams, charts or figures, describe exactly what "
    "they show including labels, values and relationships. "
    "If several images are provided they are consecutive pages of the same material. "
    "Output only the extracted content, no commentary."
)


def extract_page_content(
    client: Mistral, model: str, images: list[tuple[bytes, str | None]]
) -> str:
    if not images:
        raise ValueError("At least one page image is required.")

    parts: list[dict] = [{"type": "text", "text": TRANSCRIBE_PROMPT}]
    for data, mime in images:
        mimetype = mime or mimetypes.guess_type("x.jpg")[0]
        encoded = base64.b64encode(data).decode("ascii")
        parts.append(
            {"type": "image_url", "image_url": f"data:{mimetype};base64,{encoded}"}
        )

    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": parts}],
        temperature=0.1,
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError("The vision model returned no content for the page images.")
    return content.strip()
