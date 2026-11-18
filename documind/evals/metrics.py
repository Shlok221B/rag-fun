
def calculate_context_precision(retrieved_ids, relevant_ids):
    """
    Calculate Context Precision (Average Precision).

    retrieved_ids: chunk IDs in ranked order
    relevant_ids: chunk IDs known to be relevant
    """

    if not relevant_ids:
        return None

    relevant_set = set(map(str, relevant_ids))

    relevant_count = 0
    precision_sum = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):

        if str(chunk_id) in relevant_set:
            relevant_count += 1

            precision_at_rank = relevant_count / rank
            precision_sum += precision_at_rank

    return precision_sum / len(relevant_set)




if __name__ == "__main__":

    retrieved = ["2", "5", "4", "1", "3"]
    relevant = ["4", "1"]

    score = calculate_context_precision(
        retrieved,
        relevant
    )

    print("Context Precision:", score)
