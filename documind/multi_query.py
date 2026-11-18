
import json

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def generate_multiple_queries(question):

    response = client.responses.create(
        model=MODEL,

        instructions="""
        You are a search query generation assistant.

        Generate exactly 3 alternative search queries
        for the user's question.

        Rules:
        1. Preserve the original intent.
        2. Explore different relevant wording.
        3. Do not invent facts or assumptions.
        4. Each query should focus on a different
           aspect of the original question.
        5. Return valid JSON only in this format:

        {
            "queries": [
                "query 1",
                "query 2",
                "query 3"
            ]
        }
        """,

        input=f"Original question: {question}"
    )

    result = json.loads(response.output_text)

    return result["queries"]


if __name__ == "__main__":

    question = "Can I work from home?"

    queries = generate_multiple_queries(question)

    print("Original Question:", question)

    print("\nGenerated Queries:")

    for i, query in enumerate(queries, start=1):
        print(f"{i}. {query}")
