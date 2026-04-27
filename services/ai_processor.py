import os
import json
import re
from groq import Groq
from concurrent.futures import ThreadPoolExecutor


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. Please set it before running the application."
        )
    return Groq(api_key=api_key)


def clean_json_response(response_text):
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    return match.group(0) if match else "[]"


def analyze_chunk(chunk):
    prompt = f"""
Extract structured data in JSON.

Return ONLY:
[{{"department":"","application":"","relationship":"","business_context":""}}]

If none found return []

Text:
{chunk}
"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content
    return clean_json_response(content)


def process_chunks_parallel(chunks):
    results = []

    def worker(chunk):
        try:
            res = analyze_chunk(chunk)
            parsed = json.loads(res)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = executor.map(worker, chunks)

    for out in outputs:
        results.extend(out)

    return results