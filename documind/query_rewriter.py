
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def rewrite_query(question):

    response = client.responses.create(
        model=MODEL,

        instructions="""
        You are a search query optimization assistant.

        Your job is to rewrite user questions into
        clear and specific search queries for a
        company employee handbook.

        Rules:
        1. Preserve the original intent.
        2. Expand ambiguous wording.
        3. Include important policy-related keywords.
        4. Do not answer the question.
        5. Return only the rewritten query.
        """,

        input=f"Original question: {question}"
    )

    rewritten_query = response.output_text.strip()

    return rewritten_query


if __name__ == "__main__":

    question = "Can I work from home?"

    rewritten = rewrite_query(question)

    print("Original Query:", question)
    print("Rewritten Query:", rewritten)
