from io import BytesIO
from PyPDF2 import PdfReader

def extract_text_from_pdf(file) -> str:
    pdf = PdfReader(BytesIO(file.read()))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text
