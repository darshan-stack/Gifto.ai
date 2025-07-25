export interface Product {
  id: string
  name: string
  description: string
  price: number
  originalPrice?: number
  image: string
  rating: number
  reviewCount: number
  category: string
  brand: string
  features: string[]
  inStock: boolean
  fastShipping: boolean
  // AI-specific fields
  aiReasoning?: string
  suitabilityScore?: number
  occasionMatch?: number
  ageAppropriate?: boolean
}

export interface ChatMessage {
  id: string
  content: string
  isUser: boolean
  timestamp: Date
}

export interface FilterOptions {
  category: string
  priceRange: string
  sortBy: string
  searchTerm: string
}

export interface UserPreferences {
  favoriteCategories: string[]
  priceRange: [number, number]
  preferredBrands: string[]
}

export interface RecipientAnalysis {
  age?: number
  gender?: "male" | "female" | "non-binary" | "unknown"
  interests: string[]
  relationship: string
  occasion: string
  budget?: {
    min: number
    max: number
  }
  personality: string[]
  lifestyle: string[]
  preferences: string[]
}

export interface AIRecommendationResponse {
  success: boolean
  data: {
    recipientAnalysis: RecipientAnalysis
    products: Product[]
    totalRecommendations: number
  }
}
