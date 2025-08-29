import os
import pandas as pd
import requests
import json
import pickle
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import uuid

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
EMBEDDINGS_CACHE_FILE = 'product_embeddings.pkl'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

products_df = None
product_embeddings = None
embedding_model = None

# In-memory storage for user data (replace with database in production)
wishlists = {}
carts = {}
user_profiles = {}

class RecipientProfile(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    interests: List[str] = []
    hobbies: List[str] = []
    relationship: Optional[str] = None
    personality: List[str] = []
    lifestyle: List[str] = []
    preferences: List[str] = []

class OccasionInfo(BaseModel):
    occasion: str
    mood: Optional[str] = None
    formality: Optional[str] = None
    budget_range: Optional[Dict[str, float]] = None

class FilterOptions(BaseModel):
    category: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    eco_friendly: Optional[bool] = None
    handmade: Optional[bool] = None
    local: Optional[bool] = None
    rating_min: Optional[float] = None
    sort_by: Optional[str] = None  # price, rating, popularity

class PromptRequest(BaseModel):
    prompt: str
    recipient_profile: Optional[RecipientProfile] = None
    occasion_info: Optional[OccasionInfo] = None
    filter_options: Optional[FilterOptions] = None

class GreetingCardRequest(BaseModel):
    recipient_name: str
    occasion: str
    message_style: str  # funny, formal, emotional, romantic
    personal_message: Optional[str] = None

class ThankYouRequest(BaseModel):
    gift_name: str
    sender_name: str
    occasion: str
    message_style: str

def get_product_text(row) -> str:
    # Combine relevant fields for embedding
    fields = []
    for col in ['name', 'main_category', 'sub_category', 'description']:
        if col in row and pd.notnull(row[col]):
            fields.append(str(row[col]))
    return ' | '.join(fields)

def save_embeddings(embeddings, file_path):
    """Save embeddings to disk"""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {file_path}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def load_embeddings(file_path):
    """Load embeddings from disk"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"Embeddings loaded from {file_path}")
            return embeddings
        return None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def analyze_recipient_from_prompt(prompt: str) -> RecipientProfile:
    """Extract recipient information from prompt using AI"""
    try:
        system_prompt = """
        You are an expert at analyzing gift requests. Extract detailed information about the recipient from the user's prompt.
        Return a JSON object with these fields:
        - age: number or null
        - gender: string or null
        - interests: array of strings
        - hobbies: array of strings
        - relationship: string
        - personality: array of strings
        - lifestyle: array of strings
        - preferences: array of strings
        """
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Gift Recommendation AI"
        }
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this gift request: {prompt}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.3,
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            analysis_text = result['choices'][0]['message']['content']
            # Try to parse JSON from the response
            try:
                analysis = json.loads(analysis_text)
                return RecipientProfile(**analysis)
            except:
                # Fallback to basic extraction
                return RecipientProfile(relationship="friend")
        return RecipientProfile(relationship="friend")
    except Exception as e:
        print(f"Error analyzing recipient: {e}")
        return RecipientProfile(relationship="friend")

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
        
        # Try to load embeddings from disk first
        print("Checking for cached embeddings...")
        cached_embeddings = load_embeddings(EMBEDDINGS_CACHE_FILE)
        
        if cached_embeddings is not None and len(cached_embeddings) == len(products_df):
            print("Using cached embeddings!")
            product_embeddings = cached_embeddings
        else:
            # Compute embeddings for all products
            print("Computing product embeddings...")
            product_texts = [get_product_text(row) for _, row in products_df.iterrows()]
            product_embeddings = embedding_model.encode(product_texts, show_progress_bar=True, batch_size=256)
            print(f"Computed embeddings for {len(product_embeddings)} products.")
            
            # Save embeddings to disk for future use
            print("Saving embeddings to disk...")
            save_embeddings(product_embeddings, EMBEDDINGS_CACHE_FILE)
            
    except Exception as e:
        print(f"Error loading CSV or embeddings: {e}")
        products_df = pd.DataFrame()
        product_embeddings = None
        embedding_model = None

def find_top_products(prompt: str, recipient_profile: RecipientProfile, occasion_info: OccasionInfo, filter_options: FilterOptions, top_n: int = 100) -> List[int]:
    global product_embeddings, embedding_model, products_df
    if embedding_model is None or product_embeddings is None or products_df is None or products_df.empty:
        return []
    
    try:
        # Create enhanced prompt with recipient and occasion info
        enhanced_prompt = f"{prompt}"
        if recipient_profile:
            enhanced_prompt += f" Recipient: {recipient_profile.interests} {recipient_profile.hobbies} {recipient_profile.personality}"
        if occasion_info:
            enhanced_prompt += f" Occasion: {occasion_info.occasion} {occasion_info.mood}"
        
        prompt_emb = embedding_model.encode([enhanced_prompt])[0]
        sims = cosine_similarity([prompt_emb], product_embeddings)[0]
        
        # Apply filters
        filtered_indices = []
        for i, sim in enumerate(sims):
            if filter_options:
                row = products_df.iloc[i]
                # Price filter
                if filter_options.price_min and pd.notnull(row.get('actual_price', 0)):
                    try:
                        price = float(str(row.get('actual_price', 0)).replace('₹', '').replace(',', ''))
                        if price < filter_options.price_min:
                            continue
                    except:
                        pass
                if filter_options.price_max and pd.notnull(row.get('actual_price', 0)):
                    try:
                        price = float(str(row.get('actual_price', 0)).replace('₹', '').replace(',', ''))
                        if price > filter_options.price_max:
                            continue
                    except:
                        pass
                # Category filter
                if filter_options.category and filter_options.category.lower() not in str(row.get('main_category', '')).lower():
                    continue
                # Rating filter
                if filter_options.rating_min and pd.notnull(row.get('ratings', 0)):
                    try:
                        rating = float(row.get('ratings', 0))
                        if rating < filter_options.rating_min:
                            continue
                    except:
                        pass
            filtered_indices.append(i)
        
        # Sort by similarity and take top N
        filtered_sims = [sims[i] for i in filtered_indices]
        top_filtered_idx = np.argsort(filtered_sims)[::-1][:top_n]
        return [filtered_indices[i] for i in top_filtered_idx]
    except Exception as e:
        print(f"Embedding similarity error: {e}")
        return []

def generate_greeting_card(recipient_name: str, occasion: str, message_style: str, personal_message: str = None) -> Dict[str, str]:
    """Generate AI greeting card content"""
    try:
        system_prompt = f"""
        You are an expert greeting card writer. Create a personalized greeting card for {occasion}.
        Style: {message_style}
        Recipient: {recipient_name}
        Personal message: {personal_message or 'None provided'}
        
        Return a JSON object with:
        - title: card title
        - message: main greeting message
        - signature: suggested signature
        """
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Gift Recommendation AI"
        }
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a {message_style} greeting card for {recipient_name} for {occasion}"}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            card_text = result['choices'][0]['message']['content']
            try:
                return json.loads(card_text)
            except:
                return {
                    "title": f"Happy {occasion}!",
                    "message": card_text,
                    "signature": "With love"
                }
        return {"title": "Greeting Card", "message": "Happy occasion!", "signature": "Best wishes"}
    except Exception as e:
        print(f"Error generating greeting card: {e}")
        return {"title": "Greeting Card", "message": "Happy occasion!", "signature": "Best wishes"}

def generate_thank_you_note(gift_name: str, sender_name: str, occasion: str, message_style: str) -> str:
    """Generate thank you note"""
    try:
        system_prompt = f"""
        You are an expert at writing thank you notes. Create a {message_style} thank you note.
        Gift: {gift_name}
        Sender: {sender_name}
        Occasion: {occasion}
        """
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Gift Recommendation AI"
        }
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a {message_style} thank you note for {gift_name} from {sender_name}"}
            ],
            "max_tokens": 300,
            "temperature": 0.7,
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        return f"Thank you so much for the {gift_name}! It's perfect for {occasion}."
    except Exception as e:
        print(f"Error generating thank you note: {e}")
        return f"Thank you for the {gift_name}!"

@app.post("/recommend")
def recommend_products(req: PromptRequest):
    if products_df is None or products_df.empty:
        raise HTTPException(status_code=500, detail="No products loaded.")
    
    # Analyze recipient if not provided
    if not req.recipient_profile:
        req.recipient_profile = analyze_recipient_from_prompt(req.prompt)
    
    # Set default occasion if not provided
    if not req.occasion_info:
        req.occasion_info = OccasionInfo(occasion="general")
    
    # Set default filter options if not provided
    if not req.filter_options:
        req.filter_options = FilterOptions()
    
    prompt = req.prompt
    n = RECOMMENDATION_COUNT
    
    # Find top products using embeddings and filters
    top_idx = find_top_products(prompt, req.recipient_profile, req.occasion_info, req.filter_options, top_n=100)
    
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

    # Enhanced system prompt with explainability
    system_prompt = f"""
    You are an expert gift recommendation AI. Given a user prompt and a list of products, select the {n} most suitable products for the user.
    
    Recipient Profile:
    - Age: {req.recipient_profile.age or 'Not specified'}
    - Interests: {', '.join(req.recipient_profile.interests) or 'Not specified'}
    - Hobbies: {', '.join(req.recipient_profile.hobbies) or 'Not specified'}
    - Relationship: {req.recipient_profile.relationship or 'Not specified'}
    - Personality: {', '.join(req.recipient_profile.personality) or 'Not specified'}
    
    Occasion: {req.occasion_info.occasion}
    Mood: {req.occasion_info.mood or 'Not specified'}
    
    For each recommendation, include:
    1. Product name
    2. Why this gift is perfect (explainability)
    3. How it matches the recipient's profile
    4. Why it fits the occasion
    
    Only recommend products from the provided list.
    """
    
    user_prompt = f"User prompt: {prompt}\n\nProduct list:\n{products_text}\n\nReturn a numbered list of the top {n} product recommendations with detailed explanations."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
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
    
    return {
        "recommendations": result['choices'][0]['message']['content'],
        "recipient_profile": req.recipient_profile.dict(),
        "occasion_info": req.occasion_info.dict(),
        "filter_options": req.filter_options.dict()
    }

@app.post("/greeting-card")
def create_greeting_card(req: GreetingCardRequest):
    card_content = generate_greeting_card(
        req.recipient_name,
        req.occasion,
        req.message_style,
        req.personal_message
    )
    return {
        "card_id": str(uuid.uuid4()),
        "content": card_content,
        "created_at": datetime.now().isoformat()
    }

@app.post("/thank-you")
def create_thank_you_note(req: ThankYouRequest):
    note = generate_thank_you_note(
        req.gift_name,
        req.sender_name,
        req.occasion,
        req.message_style
    )
    return {
        "note_id": str(uuid.uuid4()),
        "content": note,
        "created_at": datetime.now().isoformat()
    }

@app.post("/wishlist/{user_id}")
def add_to_wishlist(user_id: str, product_id: str):
    if user_id not in wishlists:
        wishlists[user_id] = []
    if product_id not in wishlists[user_id]:
        wishlists[user_id].append(product_id)
    return {"message": "Added to wishlist", "wishlist": wishlists[user_id]}

@app.get("/wishlist/{user_id}")
def get_wishlist(user_id: str):
    return {"wishlist": wishlists.get(user_id, [])}

@app.post("/cart/{user_id}")
def add_to_cart(user_id: str, product_id: str, quantity: int = 1):
    if user_id not in carts:
        carts[user_id] = {}
    if product_id in carts[user_id]:
        carts[user_id][product_id] += quantity
    else:
        carts[user_id][product_id] = quantity
    return {"message": "Added to cart", "cart": carts[user_id]}

@app.get("/cart/{user_id}")
def get_cart(user_id: str):
    return {"cart": carts.get(user_id, {})}

@app.get("/health")
def health_check():
    return {"status": "healthy", "products_loaded": len(products_df) if products_df is not None else 0} 