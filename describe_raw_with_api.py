from __future__ import annotations

import argparse
from pathlib import Path

from openai import OpenAI

from convert_raw_to_jpg import OUTPUT_DIR, convert_raw_to_base64, validate_raw_file
from photo_description_config import (
    API_BASE_URL,
    API_KEY,
    DEFAULT_USER_PROMPT,
    MODEL_NAME,
    SYSTEM_PROMPT,
    TEMPERATURE,
)


def extract_message_text(message_content: object) -> str:
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    return str(message_content)


def describe_raw_photo(raw_path: Path, prompt: str, output_dir: Path = OUTPUT_DIR) -> str:
    image_base64 = convert_raw_to_base64(raw_path, output_dir=output_dir)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ],
        temperature=TEMPERATURE,
    )

    message = completion.choices[0].message
    return extract_message_text(message.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one RAW file, convert it to JPEG/base64, and request a description from an OpenAI-compatible API."
    )
    parser.add_argument("raw_file", type=Path, help="Path to a single RAW file.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_USER_PROMPT,
        help="User prompt sent together with the image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for cached JPEG files. Defaults to {OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = validate_raw_file(args.raw_file)
    output_dir = args.output_dir.expanduser().resolve()

    try:
        description = describe_raw_photo(raw_path, args.prompt, output_dir=output_dir)
    except Exception as exc:
        raise SystemExit(f"API request failed: {exc}") from exc

    print(description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())