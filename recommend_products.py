import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env.local if it exists
if os.path.exists('.env.local'):
    load_dotenv('.env.local')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment or .env.local")

CSV_PATH = 'products.csv'
RECOMMENDATION_COUNT = 50
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-exp:free"

# Load and clean the CSV (skip bad lines, handle large files)
def load_products(csv_path):
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)
        # Only drop rows where 'name' is missing
        if 'name' in df.columns:
            df = df.dropna(subset=['name'])
        else:
            print("Warning: 'name' column not found in CSV. Keeping all rows.")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

def get_recommendations(prompt, products_df, n=RECOMMENDATION_COUNT):
    # Prepare a summary of products for the LLM (truncate for token limit)
    product_samples = products_df.sample(min(100, len(products_df)))  # Sample 100 products for context
    product_descriptions = []
    for _, row in product_samples.iterrows():
        desc = ', '.join([f"{col}: {row[col]}" for col in products_df.columns if pd.notnull(row[col])])
        product_descriptions.append(desc)
    products_text = '\n'.join(product_descriptions)

    system_prompt = f"""
You are a product recommendation AI. Given a user prompt and a list of products, select the {n} most suitable products for the user. Only recommend products from the provided list. For each recommendation, include the product name and a short reason why it matches the prompt.
"""
    user_prompt = f"User prompt: {prompt}\n\nProduct list:\n{products_text}\n\nReturn a numbered list of the top {n} product recommendations with reasons."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",  # Replace with your domain or leave as is for local
        "X-Title": "Gift Recommendation AI"
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
    if response.status_code != 200:
        print(f"OpenRouter API error: {response.status_code} - {response.text}")
        exit(1)
    result = response.json()
    return result['choices'][0]['message']['content']

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Product Recommendation by Prompt using OpenRouter Gemini Flash 2.0")
    parser.add_argument('--prompt', type=str, required=True, help='User prompt for recommendations')
    args = parser.parse_args()

    products_df = load_products(CSV_PATH)
    print(f"Loaded {len(products_df)} products.")
    recommendations = get_recommendations(args.prompt, products_df, n=RECOMMENDATION_COUNT)
    print("\nRecommendations:\n")
    print(recommendations)

if __name__ == "__main__":
    main() 