
import fitz
from pathlib import Path



def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_document(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        doc = fitz.open(file_path)

        text = ""
        for page in doc:
            text += page.get_text() + "\n"

        doc.close()
        return text

    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")

    else:
        raise ValueError("Unsupported file format")


if __name__ == "__main__":
    text = load_document("documents/employee_handbook.txt")
   # print(text[:2000])
    #print total number of chunks
    print(len(chunk_text(text)))
    #first chunk
    print(chunk_text(text)[0])
    #second chunk
    print(chunk_text(text)[1])
