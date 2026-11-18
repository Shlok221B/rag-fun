
from multi_query import generate_multiple_queries
from multi_query_search import multi_query_search
from reranker import rerank_documents
from llm import generate_answer


# Experimental threshold. We will calibrate this using evaluations.
RELEVANCE_THRESHOLD = -4.0


def ask_question(question):

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Generate multiple search queries
    # --------------------------------------------------
    print("\n1. Generating multiple queries...")

    queries = generate_multiple_queries(question)

    # Always retain the original question
    all_queries = [question] + queries

    for index, query in enumerate(all_queries, start=1):
        print(f"Query {index}: {query}")

    # --------------------------------------------------
    # 2. Multi-Query Retrieval + RRF
    # --------------------------------------------------
    print("\n2. Running Multi-Query RRF...")

    unique_candidates = multi_query_search(
        queries=all_queries,
        n_results=10
    )

    print(f"\nTotal unique candidates: {len(unique_candidates)}")

    if not unique_candidates:
        answer = "I don't have enough relevant information to answer this question."

        print("\nANSWER:")
        print(answer)

        return {
            "answer": answer,
            "retrieved_chunks": [],
            "abstained": True
        }

    for chunk in unique_candidates:
        print(
            f"Chunk ID: {chunk['id']} | "
            f"RRF Score: {chunk['multi_query_rrf_score']:.6f}"
        )

    # --------------------------------------------------
    # 3. CrossEncoder Reranking
    # --------------------------------------------------
    print("\n3. Reranking documents...")

    ranked_chunks = rerank_documents(
        question,
        unique_candidates
    )

    for chunk in ranked_chunks:
        print(
            f"Chunk ID: {chunk['id']} | "
            f"Rerank Score: {chunk['rerank_score']:.4f}"
        )

    # --------------------------------------------------
    # 4. Relevance Check
    # --------------------------------------------------
    print("\n4. Checking retrieval relevance...")

    if (
        not ranked_chunks
        or ranked_chunks[0]["rerank_score"] < RELEVANCE_THRESHOLD
    ):
        answer = (
            "I don't have enough relevant information "
            "to answer this question."
        )

        print("\nANSWER:")
        print(answer)

        return {
            "answer": answer,
            "retrieved_chunks": ranked_chunks[:3],
            "abstained": True
        }

    print("Relevant documents found.")

    # Select top 3 chunks for the LLM
    top_chunks = ranked_chunks[:3]

    # --------------------------------------------------
    # 5. Generate Answer
    # --------------------------------------------------
    print("\n5. Generating answer...")

    answer = generate_answer(
        question,
        top_chunks
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)

    # --------------------------------------------------
    # 6. Display Sources
    # --------------------------------------------------
    print("\nSOURCES:")

    for chunk in top_chunks:
        metadata = chunk.get("metadata", {})

        print(
            f"- Chunk ID: {chunk['id']} | "
            f"Source: {metadata.get('source', 'Unknown')} | "
            f"Rerank Score: {chunk['rerank_score']:.4f}"
        )

    return {
        "answer": answer,
        "retrieved_chunks": top_chunks,
        "abstained": False
    }


if __name__ == "__main__":

    question = "Can I work from home?"

    result = ask_question(question)
