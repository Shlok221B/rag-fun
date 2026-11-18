
import json
import sys
import traceback
from pathlib import Path

print("Evaluation script started!", flush=True)

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

print("Project root:", PROJECT_ROOT, flush=True)

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from main import ask_question
    print("RAG pipeline imported successfully!", flush=True)

except Exception:
    print("Failed to import main.py", flush=True)
    traceback.print_exc()
    sys.exit(1)


def run_evaluations():

    dataset_path = PROJECT_ROOT / "evals" / "dataset.json"

    print("\nLoading dataset...", flush=True)

    if not dataset_path.exists():
        print("ERROR: dataset.json not found:", dataset_path)
        return

    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    print(f"Loaded {len(dataset)} test cases!", flush=True)

    results = []

    print("\n========== DOCUMIND EVALUATION ==========")

    for test_case in dataset:

        print("\n----------------------------------")
        print(f"Running Test Case {test_case['id']}")
        print("Question:", test_case["question"])

        try:
            result = ask_question(test_case["question"])

            retrieved_ids = {
                str(chunk["id"])
                for chunk in result["retrieved_chunks"]
            }

            expected_ids = {
                str(chunk_id)
                for chunk_id in test_case["expected_chunk_ids"]
            }

            if test_case["answerable"]:

                retrieval_passed = bool(
                    retrieved_ids.intersection(expected_ids)
                )

            else:

                # For unanswerable questions, expect
                # the relevance gate to reject retrieval.
                retrieval_passed = (
                    len(result["retrieved_chunks"]) == 0
                )

            evaluation_result = {
                "id": test_case["id"],
                "question": test_case["question"],
                "retrieval_passed": retrieval_passed,
                "retrieved_chunks": list(retrieved_ids)
            }

            results.append(evaluation_result)

            print("\nRetrieved IDs:", retrieved_ids)
            print("Expected IDs:", expected_ids)

            print(
                "Retrieval Evaluation:",
                "PASS" if retrieval_passed else "FAIL"
            )

        except Exception:
            print("Test case failed due to an error:")
            traceback.print_exc()

    # Final summary
    passed = sum(
        1 for result in results
        if result["retrieval_passed"]
    )

    failed = len(results) - passed

    print("\n========== FINAL RESULTS ==========")
    print("Total tests:", len(results))
    print("Passed:", passed)
    print("Failed:", failed)

    if results:
        print("\nDetailed Results:")

        for result in results:
            status = (
                "PASS"
                if result["retrieval_passed"]
                else "FAIL"
            )

            print(
                f"Test {result['id']}: {status}"
                f" | {result['question']}"
            )


if __name__ == "__main__":
    try:
        run_evaluations()

    except Exception:
        print("\nEvaluation runner crashed:")
        traceback.print_exc()
