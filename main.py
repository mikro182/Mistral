import os

from mistralai.client import Mistral


def main() -> None:
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-small-latest"

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Schreibe ein kurzes Haiku über Programmieren.",
            },
        ],
    )

    print(chat_response.choices[0].message.content)


if __name__ == "__main__":
    main()
