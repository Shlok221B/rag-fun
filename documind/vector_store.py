
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Get existing collection or create it
collection = client.get_or_create_collection(
    name="employee_handbook",
    metadata={"hnsw:space": "cosine"}
)


def store_embeddings(chunks, embeddings):
    collection.upsert(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": "employee_handbook.txt",
                "document_type": "employee_handbook"
            }
            for _ in chunks
        ]
    )

    print(f"Stored {len(chunks)} chunks!")


def search_documents(query, n_results=3):
    from embeddings import generate_embeddings

    query_embedding = generate_embeddings([query])

    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        where={"document_type": "employee_handbook"},
        include=["documents", "distances", "metadatas"]
    )
    for i, document in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]

        print(f"\nResult {i+1}")
        print("Cosine distance:", distance)
        print("Content:", document)

    return results
