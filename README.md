# 🎁 Gifto.ai - AI-Powered Gift Recommendation Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-blue)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://typescriptlang.org/)

## 📖 Overview

Gifto.ai is an intelligent gift recommendation platform that leverages advanced AI and machine learning to provide personalized gift suggestions. The platform combines natural language processing, semantic search, and recipient profiling to deliver highly relevant gift recommendations for any occasion.

## 🎯 Key Features

- **🤖 AI-Powered Recommendations**: Advanced LLM integration with semantic understanding
- **👤 Recipient Profiling**: Intelligent analysis of recipient characteristics and preferences
- **🎉 Occasion Awareness**: Context-aware recommendations for birthdays, anniversaries, holidays
- **💬 Interactive Chat Interface**: Conversational UI for natural gift discovery
- **📝 AI-Generated Content**: Custom greeting cards and thank-you notes
- **❤️ Wishlist Management**: Save and organize favorite gifts
- **🛒 Shopping Cart**: Seamless e-commerce integration
- **📊 Advanced Filtering**: Price, category, rating, and sustainability filters
- **🎨 Modern UI/UX**: Beautiful, responsive design with dark/light themes

## 🏗️ Architecture & Tech Stack

### Frontend (Next.js 14)
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript 5.0
- **Styling**: Tailwind CSS
- **UI Components**: Custom React components
- **State Management**: React hooks and localStorage
- **Deployment**: Vercel-ready

### Backend (FastAPI)
- **Framework**: FastAPI (Python)
- **AI/ML**: 
  - Sentence Transformers (`all-MiniLM-L6-v2`)
  - Scikit-learn (Cosine Similarity)
  - OpenRouter API (Gemini Flash 2.0)
- **Data Processing**: Pandas
- **API**: RESTful endpoints with CORS support
- **Server**: Uvicorn ASGI server

### AI/ML Technologies
- **Embeddings**: Local sentence transformers for semantic search
- **LLM Integration**: OpenRouter API with Gemini Flash 2.0
- **Vector Search**: Cosine similarity for product matching
- **Natural Language Processing**: Prompt analysis and recipient profiling

### Data & Storage
- **Product Database**: Large-scale CSV with 550K+ products
- **Local Storage**: Browser-based chat history and wishlist
- **Caching**: In-memory product embeddings for fast search

## 🚀 Features Breakdown

### 1. Intelligent Recommendation Engine
- Semantic search with embeddings
- Cosine similarity for product matching
- Context-aware filtering

### 2. Recipient Profiling System
- Age analysis and interest extraction
- Relationship context understanding
- Personality trait recognition

### 3. Advanced Filtering & Sorting
- Price range filtering
- Category-based recommendations
- Rating and sustainability filters

### 4. AI Content Generation
- Personalized greeting cards
- Custom thank you notes
- Gift explanation reasoning

### 5. Interactive User Experience
- Persistent chat history
- Wishlist management
- Responsive product pages

## 📦 Installation & Setup

### Prerequisites
- Node.js 18+ 
- Python 3.10+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/darshan-stack/Gifto.ai.git
cd Gifto.ai
```

### 2. Frontend Setup (Next.js)
```bash
npm install
cp .env.example .env.local
npm run dev
```

### 3. Backend Setup (FastAPI)
```bash
pip install -r requirements.txt
uvicorn recommendation_service:app --reload
```

### 4. Environment Configuration
Create `.env.local` file:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## 🔧 API Endpoints

### Core Recommendation
```http
POST /api/ai-recommendations
{
  "prompt": "Birthday gift for a 10-year-old who loves science",
  "recipient_profile": {
    "age": 10,
    "interests": ["science", "experiments"],
    "hobbies": ["reading", "building"],
    "relationship": "nephew",
    "personality": ["curious", "creative"]
  },
  "occasion_info": {
    "occasion": "birthday",
    "mood": "exciting"
  },
  "filter_options": {
    "max_price": 50,
    "categories": ["toys", "books"],
    "min_rating": 4.0
  }
}
```

### AI Content Generation
```http
POST /api/greeting-card
POST /api/thank-you
```

### Wishlist Management
```http
POST /wishlist/{user_id}
GET /wishlist/{user_id}
```

## 🎨 UI Components

### Chat Sidebar
- Persistent chat history
- Search conversations
- Load previous recommendations

### Wishlist Icon
- Real-time wishlist counter
- Quick preview dropdown
- Direct navigation

### Product Pages
- Detailed product information
- AI reasoning for recommendations
- Add to wishlist/cart functionality

## 📊 Performance & Scalability

### Optimizations
- Embedding caching for fast search
- Batch processing for large datasets
- Lazy loading for better UX
- CDN-ready static assets

### Scalability Features
- Microservice architecture
- API rate limiting
- Efficient CSV processing
- Optimized memory management

## 🔒 Security & Privacy

- Secure API key management
- CORS configuration
- Input validation with Pydantic
- Graceful error handling

## 🚀 Deployment

### Frontend (Vercel)
```bash
npm run build
npm run start
```

### Backend (Docker)
```dockerfile
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "recommendation_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Business Value

### For E-commerce Companies
- Increased conversion through personalization
- Customer retention via engaging UX
- Data insights for product optimization
- Revenue growth through better recommendations

### For Gift Retailers
- Seasonal optimization capabilities
- Inventory management insights
- Enhanced customer satisfaction
- Improved brand loyalty

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**DarshanX.ai** - [GitHub Profile](https://github.com/darshan-stack)

## 🙏 Acknowledgments

- OpenRouter for LLM API access
- Sentence Transformers for semantic embeddings
- Next.js Team for the amazing framework
- FastAPI for the high-performance backend

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/darshan-stack/Gifto.ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/darshan-stack/Gifto.ai/discussions)

---

⭐ **Star this repository if you find it helpful!**
