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
You are an expert Document Analyst. Your task is to extract relationships between Departments and Software Applications from the text below.

### EXTRACTION RULES:
1. **Department**: Extract the organizational unit, team, or business group (e.g., "Finance", "HR", "North Regional Office").
2. **Application**: Extract the name of any software, tool, database, or system mentioned (e.g., "SAP", "Excel", "Legacy CRM", "SharePoint").
3. **Relationship**: Briefly describe how they interact (e.g., "uses for reporting", "manages data in", "primary owner").
4. **Business Context**: Provide 1 sentence of extra detail about why they use it.

### OUTPUT FORMAT:
Return ONLY a JSON array. If nothing is found, return [].
Example:
[
  {{"department": "Accounting", "application": "QuickBooks", "relationship": "uses for payroll", "business_context": "Processes monthly salary for 500 employees."}}
]

### TEXT TO ANALYZE:
{chunk}
"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a precise data extraction agent. Return only JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
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