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
import re

# Enhanced embedding imports
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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

# Enhanced embedding model for better semantic understanding
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # Better than all-MiniLM-L6-v2
EMBEDDINGS_CACHE_FILE = 'product_embeddings.pkl'
PRODUCT_FEATURES_CACHE_FILE = 'product_features.pkl'

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
tfidf_vectorizer = None
product_features = None

# In-memory storage for user data (replace with database in production)
wishlists = {}
carts = {}
user_profiles = {}
recommendation_feedback = {}  # Store user feedback for model improvement

class RecipientProfile(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    interests: List[str] = []
    hobbies: List[str] = []
    relationship: Optional[str] = None
    personality: List[str] = []
    lifestyle: List[str] = []
    preferences: List[str] = []
    budget_preference: Optional[str] = None  # low, medium, high, luxury
    tech_savviness: Optional[str] = None  # low, medium, high
    style_preference: Optional[str] = None  # classic, modern, trendy, minimalist

class OccasionInfo(BaseModel):
    occasion: str
    mood: Optional[str] = None
    formality: Optional[str] = None
    budget_range: Optional[Dict[str, float]] = None
    urgency: Optional[str] = None  # immediate, planned, last_minute
    group_size: Optional[int] = None  # individual, couple, family, group

class FilterOptions(BaseModel):
    category: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    eco_friendly: Optional[bool] = None
    handmade: Optional[bool] = None
    local: Optional[bool] = None
    rating_min: Optional[float] = None
    sort_by: Optional[str] = None  # price, rating, popularity, relevance
    exclude_categories: List[str] = []
    include_brands: List[str] = []

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

class FeedbackRequest(BaseModel):
    user_id: str
    product_id: str
    rating: int  # 1-5
    feedback_type: str  # like, dislike, purchase, view
    context: Optional[str] = None

def preprocess_text(text: str) -> str:
    """Enhanced text preprocessing for better embeddings"""
    if pd.isna(text) or not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_enhanced_product_text(row) -> str:
    """Enhanced product text generation with better feature extraction"""
    fields = []
    
    # Primary fields with higher weight
    primary_fields = ['name', 'main_category', 'sub_category', 'description']
    for col in primary_fields:
        if col in row and pd.notnull(row[col]):
            fields.append(str(row[col]))
    
    # Secondary fields for context
    secondary_fields = ['brand', 'color', 'material', 'size', 'style']
    for col in secondary_fields:
        if col in row and pd.notnull(row[col]):
            fields.append(str(row[col]))
    
    # Price context
    if 'actual_price' in row and pd.notnull(row['actual_price']):
        try:
            price = float(str(row['actual_price']).replace('₹', '').replace(',', ''))
            if price < 500:
                fields.append("budget friendly affordable")
            elif price < 2000:
                fields.append("mid range moderate price")
            elif price < 5000:
                fields.append("premium quality expensive")
            else:
                fields.append("luxury high end exclusive")
        except:
            pass
    
    # Rating context
    if 'ratings' in row and pd.notnull(row['ratings']):
        try:
            rating = float(row['ratings'])
            if rating >= 4.5:
                fields.append("highly rated excellent quality")
            elif rating >= 4.0:
                fields.append("well rated good quality")
            elif rating >= 3.5:
                fields.append("decent rating average quality")
        except:
            pass
    
    return ' | '.join(fields)

def extract_product_features(products_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract comprehensive product features for better matching"""
    features = {}
    
    for idx, row in products_df.iterrows():
        product_id = str(idx)
        features[product_id] = {
            'categories': [],
            'keywords': [],
            'price_tier': 'unknown',
            'rating_tier': 'unknown',
            'age_appropriateness': 'all',
            'gender_target': 'unisex',
            'occasion_suitability': [],
            'complexity': 'simple'
        }
        
        # Extract categories
        for col in ['main_category', 'sub_category']:
            if col in row and pd.notnull(row[col]):
                features[product_id]['categories'].append(str(row[col]).lower())
        
        # Extract keywords from name and description
        text = get_enhanced_product_text(row)
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        features[product_id]['keywords'] = list(set(keywords))
        
        # Price tier classification
        if 'actual_price' in row and pd.notnull(row['actual_price']):
            try:
                price = float(str(row['actual_price']).replace('₹', '').replace(',', ''))
                if price < 500:
                    features[product_id]['price_tier'] = 'budget'
                elif price < 2000:
                    features[product_id]['price_tier'] = 'mid_range'
                elif price < 5000:
                    features[product_id]['price_tier'] = 'premium'
                else:
                    features[product_id]['price_tier'] = 'luxury'
            except:
                pass
        
        # Rating tier classification
        if 'ratings' in row and pd.notnull(row['ratings']):
            try:
                rating = float(row['ratings'])
                if rating >= 4.5:
                    features[product_id]['rating_tier'] = 'excellent'
                elif rating >= 4.0:
                    features[product_id]['rating_tier'] = 'good'
                elif rating >= 3.5:
                    features[product_id]['rating_tier'] = 'average'
                else:
                    features[product_id]['rating_tier'] = 'poor'
            except:
                pass
        
        # Age appropriateness
        text_lower = text.lower()
        if any(word in text_lower for word in ['toy', 'kids', 'child', 'baby', 'toddler']):
            features[product_id]['age_appropriateness'] = 'children'
        elif any(word in text_lower for word in ['teen', 'adolescent', 'youth']):
            features[product_id]['age_appropriateness'] = 'teen'
        elif any(word in text_lower for word in ['adult', 'mature', 'professional']):
            features[product_id]['age_appropriateness'] = 'adult'
        
        # Gender targeting
        if any(word in text_lower for word in ['men', 'male', 'guy', 'boy']):
            features[product_id]['gender_target'] = 'male'
        elif any(word in text_lower for word in ['women', 'female', 'girl', 'lady']):
            features[product_id]['gender_target'] = 'female'
        
        # Occasion suitability
        occasions = []
        if any(word in text_lower for word in ['birthday', 'party', 'celebration']):
            occasions.append('birthday')
        if any(word in text_lower for word in ['wedding', 'marriage', 'anniversary']):
            occasions.append('wedding')
        if any(word in text_lower for word in ['christmas', 'holiday', 'festival']):
            occasions.append('holiday')
        if any(word in text_lower for word in ['graduation', 'achievement', 'success']):
            occasions.append('achievement')
        features[product_id]['occasion_suitability'] = occasions
    
    return features

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

def save_features(features, file_path):
    """Save product features to disk"""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Product features saved to {file_path}")
    except Exception as e:
        print(f"Error saving features: {e}")

def load_features(file_path):
    """Load product features from disk"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                features = pickle.load(f)
            print(f"Product features loaded from {file_path}")
            return features
        return None
    except Exception as e:
        print(f"Error loading features: {e}")
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
    global products_df, product_embeddings, embedding_model, tfidf_vectorizer, product_features
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
            product_texts = [get_enhanced_product_text(row) for _, row in products_df.iterrows()]
            product_embeddings = embedding_model.encode(product_texts, show_progress_bar=True, batch_size=256)
            print(f"Computed embeddings for {len(product_embeddings)} products.")
            
            # Save embeddings to disk for future use
            print("Saving embeddings to disk...")
            save_embeddings(product_embeddings, EMBEDDINGS_CACHE_FILE)
            
        # Load TF-IDF vectorizer
        print("Loading TF-IDF vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        tfidf_vectorizer.fit(product_texts)
        print("TF-IDF vectorizer loaded.")

        # Load product features
        print("Loading product features...")
        product_features = extract_product_features(products_df)
        save_features(product_features, PRODUCT_FEATURES_CACHE_FILE)
        print("Product features loaded.")
        
    except Exception as e:
        print(f"Error loading CSV or embeddings: {e}")
        products_df = pd.DataFrame()
        product_embeddings = None
        embedding_model = None
        tfidf_vectorizer = None
        product_features = None

def find_top_products(prompt: str, recipient_profile: RecipientProfile, occasion_info: OccasionInfo, filter_options: FilterOptions, top_n: int = 100) -> List[int]:
    """Enhanced product search with multiple ranking factors"""
    global product_embeddings, embedding_model, products_df, tfidf_vectorizer, product_features
    if embedding_model is None or product_embeddings is None or products_df is None or products_df.empty:
        return []
    
    try:
        # Create enhanced prompt with recipient and occasion info
        enhanced_prompt = f"{prompt}"
        if recipient_profile:
            enhanced_prompt += f" Recipient: {recipient_profile.interests} {recipient_profile.hobbies} {recipient_profile.personality}"
        if occasion_info:
            enhanced_prompt += f" Occasion: {occasion_info.occasion} {occasion_info.mood}"
        
        # Get semantic similarity scores
        prompt_emb = embedding_model.encode([enhanced_prompt])[0]
        semantic_sims = cosine_similarity([prompt_emb], product_embeddings)[0]
        
        # Get TF-IDF similarity scores
        tfidf_sims = np.zeros(len(products_df))
        if tfidf_vectorizer:
            try:
                prompt_tfidf = tfidf_vectorizer.transform([enhanced_prompt])
                product_tfidf = tfidf_vectorizer.transform([get_enhanced_product_text(row) for _, row in products_df.iterrows()])
                tfidf_sims = cosine_similarity(prompt_tfidf, product_tfidf)[0]
            except:
                pass
        
        # Calculate feature-based scores
        feature_scores = np.zeros(len(products_df))
        for i, (idx, row) in enumerate(products_df.iterrows()):
            score = 0
            product_id = str(idx)
            
            if product_features and product_id in product_features:
                features = product_features[product_id]
                
                # Age appropriateness scoring
                if recipient_profile.age:
                    if recipient_profile.age < 13 and features['age_appropriateness'] == 'children':
                        score += 0.3
                    elif 13 <= recipient_profile.age < 18 and features['age_appropriateness'] == 'teen':
                        score += 0.3
                    elif recipient_profile.age >= 18 and features['age_appropriateness'] == 'adult':
                        score += 0.3
                
                # Gender targeting scoring
                if recipient_profile.gender and features['gender_target'] != 'unisex':
                    if recipient_profile.gender.lower() in features['gender_target']:
                        score += 0.2
                
                # Occasion suitability scoring
                if occasion_info.occasion and occasion_info.occasion.lower() in features['occasion_suitability']:
                    score += 0.4
                
                # Budget preference scoring
                if recipient_profile.budget_preference and features['price_tier'] != 'unknown':
                    budget_map = {'low': 'budget', 'medium': 'mid_range', 'high': 'premium', 'luxury': 'luxury'}
                    if budget_map.get(recipient_profile.budget_preference) == features['price_tier']:
                        score += 0.3
                
                # Rating preference scoring
                if features['rating_tier'] in ['excellent', 'good']:
                    score += 0.2
            
            feature_scores[i] = score
        
        # Combine all scores with weights
        combined_scores = (
            0.5 * semantic_sims +      # Semantic similarity (50% weight)
            0.3 * tfidf_sims +         # TF-IDF similarity (30% weight)
            0.2 * feature_scores       # Feature-based scoring (20% weight)
        )
        
        # Apply filters
        filtered_indices = []
        for i, score in enumerate(combined_scores):
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
                # Exclude categories
                if filter_options.exclude_categories:
                    if any(cat.lower() in str(row.get('main_category', '')).lower() for cat in filter_options.exclude_categories):
                        continue
                # Include brands
                if filter_options.include_brands:
                    if not any(brand.lower() in str(row.get('brand', '')).lower() for brand in filter_options.include_brands):
                        continue
            filtered_indices.append(i)
        
        # Sort by combined score and take top N
        filtered_scores = [combined_scores[i] for i in filtered_indices]
        top_filtered_idx = np.argsort(filtered_scores)[::-1][:top_n]
        return [filtered_indices[i] for i in top_filtered_idx]
    except Exception as e:
        print(f"Enhanced product search error: {e}")
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

    # Advanced system prompt with comprehensive analysis
    system_prompt = f"""
    You are an expert gift recommendation AI with deep understanding of human psychology, relationships, and gift-giving etiquette. 
    Your task is to analyze the user's request and provide highly personalized, thoughtful gift recommendations.
    
    RECIPIENT ANALYSIS:
    - Age: {req.recipient_profile.age or 'Not specified'} (Consider age-appropriate gifts)
    - Gender: {req.recipient_profile.gender or 'Not specified'} (Respect preferences)
    - Interests: {', '.join(req.recipient_profile.interests) or 'Not specified'} (Primary matching factor)
    - Hobbies: {', '.join(req.recipient_profile.hobbies) or 'Not specified'} (Activity-based gifts)
    - Relationship: {req.recipient_profile.relationship or 'Not specified'} (Formality level)
    - Personality: {', '.join(req.recipient_profile.personality) or 'Not specified'} (Gift style)
    - Budget Preference: {req.recipient_profile.budget_preference or 'Not specified'} (Price sensitivity)
    - Tech Savviness: {req.recipient_profile.tech_savviness or 'Not specified'} (Technology comfort)
    - Style Preference: {req.recipient_profile.style_preference or 'Not specified'} (Aesthetic taste)
    
    OCCASION CONTEXT:
    - Occasion: {req.occasion_info.occasion} (Cultural significance)
    - Mood: {req.occasion_info.mood or 'Not specified'} (Emotional tone)
    - Formality: {req.occasion_info.formality or 'Not specified'} (Event type)
    - Urgency: {req.occasion_info.urgency or 'Not specified'} (Time sensitivity)
    - Group Size: {req.occasion_info.group_size or 'Not specified'} (Social context)
    
    RECOMMENDATION CRITERIA:
    1. PERSONALIZATION: How well does it match the recipient's unique characteristics?
    2. OCCASION FIT: Is it appropriate for the specific occasion and mood?
    3. RELATIONSHIP APPROPRIATENESS: Does it match the relationship dynamic?
    4. PRACTICALITY: Will the recipient actually use/enjoy this gift?
    5. MEMORABILITY: Will it create a lasting positive impression?
    6. UNIQUENESS: Is it thoughtful and not generic?
    
    For each recommendation, provide:
    - Product name and key details
    - PERSONALIZATION REASONING: Why this specific gift matches this specific person
    - OCCASION FIT: How it enhances the celebration/event
    - RELATIONSHIP VALUE: How it strengthens the relationship
    - PRACTICAL BENEFITS: What the recipient will gain from it
    - PERSONALIZATION TIPS: How to make it even more special
    
    Focus on creating emotional connections and meaningful experiences, not just material items.
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

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    if feedback.user_id not in recommendation_feedback:
        recommendation_feedback[feedback.user_id] = []
    recommendation_feedback[feedback.user_id].append(feedback.dict())
    return {"message": "Feedback submitted", "feedback_count": len(recommendation_feedback[feedback.user_id])}

@app.get("/feedback/{user_id}")
def get_user_feedback(user_id: str):
    return {"feedback": recommendation_feedback.get(user_id, [])}

@app.get("/health")
def health_check():
    return {"status": "healthy", "products_loaded": len(products_df) if products_df is not None else 0} 