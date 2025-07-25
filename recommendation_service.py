import os
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

# Embedding imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI()
products_df = None
product_embeddings = None
embedding_model = None

class PromptRequest(BaseModel):
    prompt: str

def get_product_text(row) -> str:
    # Combine relevant fields for embedding
    fields = []
    for col in ['name', 'main_category', 'sub_category', 'description']:
        if col in row and pd.notnull(row[col]):
            fields.append(str(row[col]))
    return ' | '.join(fields)

@app.on_event("startup")
def load_products():
    global products_df, product_embeddings, embedding_model
    try:
        products_df = pd.read_csv(CSV_PATH, on_bad_lines='skip', low_memory=False, skiprows=3)
        if 'name' in products_df.columns:
            products_df = products_df.dropna(subset=['name'])
        else:
            print("Warning: 'name' column not found in CSV. Keeping all rows.")
        print(f"Loaded {len(products_df)} products.")
        # Load embedding model
        print("Loading embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # Compute embeddings for all products
        print("Computing product embeddings...")
        product_texts = [get_product_text(row) for _, row in products_df.iterrows()]
        product_embeddings = embedding_model.encode(product_texts, show_progress_bar=True, batch_size=256)
        print(f"Computed embeddings for {len(product_embeddings)} products.")
    except Exception as e:
        print(f"Error loading CSV or embeddings: {e}")
        products_df = pd.DataFrame()
        product_embeddings = None
        embedding_model = None

def find_top_products(prompt: str, top_n: int = 100) -> List[int]:
    global product_embeddings, embedding_model, products_df
    if embedding_model is None or product_embeddings is None or products_df is None or products_df.empty:
        return []
    try:
        prompt_emb = embedding_model.encode([prompt])[0]
        sims = cosine_similarity([prompt_emb], product_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_n]
        return top_idx
    except Exception as e:
        print(f"Embedding similarity error: {e}")
        return []

@app.post("/recommend")
def recommend_products(req: PromptRequest):
    if products_df is None or products_df.empty:
        raise HTTPException(status_code=500, detail="No products loaded.")
    prompt = req.prompt
    n = RECOMMENDATION_COUNT
    # Try embedding-based filtering
    top_idx = find_top_products(prompt, top_n=100)
    if len(top_idx) > 0:
        product_samples = products_df.iloc[top_idx]
    else:
        # Fallback: keyword filtering
        mask = (
            products_df['name'].str.contains(prompt, case=False, na=False) |
            products_df.get('main_category', pd.Series(['']*len(products_df))).str.contains(prompt, case=False, na=False)
        )
        filtered = products_df[mask]
        if len(filtered) > 0:
            product_samples = filtered.sample(min(100, len(filtered)))
        else:
            product_samples = products_df.sample(min(100, len(products_df)))
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
        "HTTP-Referer": "http://localhost:8000",  # For local dev
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
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.status_code} - {response.text}")
    result = response.json()
    return {"recommendations": result['choices'][0]['message']['content']} 