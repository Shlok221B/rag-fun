
from hybrid_search import hybrid_search


def multi_query_search(queries, n_results=10, k=60):

    scores = {}
    documents = {}

    print("\nRunning searches for multiple queries...")

    for query_index, query in enumerate(queries, start=1):

        print(f"\nSearching Query {query_index}: {query}")

        results = hybrid_search(
            query=query,
            n_results=n_results
        )

        # Apply RRF to this query's ranking
        for rank, chunk in enumerate(results, start=1):

            chunk_id = str(chunk["id"])

            rrf_score = 1 / (k + rank)

            scores[chunk_id] = (
                scores.get(chunk_id, 0) + rrf_score
            )

            # Preserve the chunk and its metadata
            documents[chunk_id] = chunk

    # Sort by combined RRF score
    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Return unique chunks in combined ranking order
    final_results = []

    for chunk_id, score in ranked[:n_results]:

        chunk = documents[chunk_id].copy()

        chunk["multi_query_rrf_score"] = score

        final_results.append(chunk)

    return final_results


if __name__ == "__main__":

    queries = [
        "Can I work from home?",
        "What is the company's remote work policy?",
        "How many days per week can employees work remotely?",
        "Is manager approval required for working from home?"
    ]

    results = multi_query_search(
        queries,
        n_results=10
    )

    print("\nFinal Multi-Query RRF Results:")

    for result in results:

        print("\nChunk ID:", result["id"])
        print("RRF Score:", result["multi_query_rrf_score"])
        print("Content:", result["text"][:200])
