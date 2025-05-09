import os
import csv
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Path to CSV file
csv_file_path = os.path.join(os.path.dirname(__file__), 'evalsnippet.csv')

# Read CSV data
rows = []
with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        rows.append(row)

def evaluate_reasoning_with_gpt(search_query, truthfulness_rating, llm_reasoning):
    prompt = f"""
You are an expert judge evaluating the quality of an LLM's reasoning and its truthfulness rating for a given claim.

Claim: {search_query}
LLM Reasoning: {llm_reasoning}
Truthfulness Rating: {truthfulness_rating}

Evaluate if the reasoning and rating are appropriate for the claim. Respond with a short critique and a final verdict: 'Good' or 'Bad'.
"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content

# Evaluate each row and collect stats
results = []
good_count = 0
bad_count = 0
for row in rows:
    result = evaluate_reasoning_with_gpt(
        row['search_query_used'],
        row['truthfulness_rating'],
        row['LLM_reasoning']
    )
    print(f"Claim: {row['search_query_used']}")
    print(f"Evaluation: {result}\n{'-'*40}")
    results.append(result)
    # Count verdicts
    verdict = None
    if 'Good' in result:
        good_count += 1
        verdict = 'Good'
    elif 'Bad' in result:
        bad_count += 1
        verdict = 'Bad'
    else:
        verdict = 'Unknown'

# Output statistics
print("\nFinal Statistics:")
print(f"Total Evaluations: {len(results)}")
print(f"Good: {good_count}")
print(f"Bad: {bad_count}")
if len(results) > 0:
    print(f"Good %: {good_count / len(results) * 100:.2f}%")
    print(f"Bad %: {bad_count / len(results) * 100:.2f}%")