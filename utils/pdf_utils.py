import pdfplumber


def extract_text_from_pdf(pdf_file):
    text_parts = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text_parts.append(content)
    return "\n".join(text_parts)


def chunk_text(text, chunk_size=2500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
