
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


if __name__ == "__main__":
    sample_chunks = [
        "Employees receive 12 casual leave days annually.",
        "The company provides 10 sick leave days.",
        "Employees receive 15 earned leave days."
    ]

    vectors = generate_embeddings(sample_chunks)

    print("Number of embeddings:", len(vectors))
    print("Vector dimensions:", vectors.shape)
    print("First embedding:", vectors[0])
