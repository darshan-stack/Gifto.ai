import { generateObject } from "ai"
import { openai } from "@ai-sdk/openai"
import { z } from "zod"
import type { Product } from "./types"

const RecipientAnalysisSchema = z.object({
  age: z.number().optional(),
  gender: z.enum(["male", "female", "non-binary", "unknown"]).optional(),
  interests: z.array(z.string()),
  relationship: z.string(),
  occasion: z.string(),
  budget: z
    .object({
      min: z.number(),
      max: z.number(),
    })
    .optional(),
  personality: z.array(z.string()),
  lifestyle: z.array(z.string()),
  preferences: z.array(z.string()),
})

const ProductRecommendationSchema = z.object({
  recommendations: z.array(
    z.object({
      name: z.string(),
      description: z.string(),
      category: z.string(),
      priceRange: z.object({
        min: z.number(),
        max: z.number(),
      }),
      reasoning: z.string(),
      suitabilityScore: z.number().min(1).max(10),
      tags: z.array(z.string()),
      ageAppropriate: z.boolean(),
      occasionMatch: z.number().min(1).max(10),
    }),
  ),
})

export async function analyzeRecipient(prompt: string) {
  try {
    // Validate API key exists
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OpenAI API key is not configured")
    }

    // Check if API key format is valid
    if (!process.env.OPENAI_API_KEY.startsWith("sk-")) {
      throw new Error("Invalid OpenAI API key format")
    }

    const { object } = await generateObject({
      model: openai("gpt-4o"),
      system: `You are an expert gift consultant. Analyze the user's prompt to understand who they're shopping for and extract key details about the recipient, occasion, and preferences. Be thorough in your analysis.`,
      prompt: `Analyze this gift request and extract detailed information about the recipient: "${prompt}"`,
      schema: RecipientAnalysisSchema,
    })

    return object
  } catch (error) {
    console.error("Error analyzing recipient:", error)

    // Handle specific API key errors
    if (error instanceof Error) {
      if (error.message.includes("Incorrect API key") || error.message.includes("Invalid API key")) {
        throw new Error("INVALID_API_KEY")
      }
      if (error.message.includes("API key is not configured")) {
        throw new Error("MISSING_API_KEY")
      }
      if (error.message.includes("quota") || error.message.includes("billing")) {
        throw new Error("QUOTA_EXCEEDED")
      }
      if (error.message.includes("rate limit")) {
        throw new Error("RATE_LIMITED")
      }
    }

    throw new Error("AI_SERVICE_ERROR")
  }
}

export async function generateAIRecommendations(prompt: string, recipientAnalysis: any) {
  try {
    const { object } = await generateObject({
      model: openai("gpt-4o"),
      system: `You are an expert gift recommendation AI. Based on the recipient analysis, suggest 20-30 highly relevant gift ideas. Consider:
      - The recipient's age, interests, and personality
      - The occasion and relationship context
      - Budget constraints if mentioned
      - Current trends and popular items
      - Practical vs. experiential gifts
      - Personalization opportunities
      
      Provide diverse recommendations across different categories and price points. Each recommendation should include a clear reasoning for why it's suitable.`,
      prompt: `
      Original request: "${prompt}"
      
      Recipient Analysis:
      - Age: ${recipientAnalysis.age || "Not specified"}
      - Gender: ${recipientAnalysis.gender || "Not specified"}
      - Interests: ${recipientAnalysis.interests?.join(", ") || "Not specified"}
      - Relationship: ${recipientAnalysis.relationship}
      - Occasion: ${recipientAnalysis.occasion}
      - Budget: ${recipientAnalysis.budget ? `$${recipientAnalysis.budget.min}-$${recipientAnalysis.budget.max}` : "Not specified"}
      - Personality: ${recipientAnalysis.personality?.join(", ") || "Not specified"}
      - Lifestyle: ${recipientAnalysis.lifestyle?.join(", ") || "Not specified"}
      - Preferences: ${recipientAnalysis.preferences?.join(", ") || "Not specified"}
      
      Generate personalized gift recommendations that match these specific details.
      `,
      schema: ProductRecommendationSchema,
    })

    return object.recommendations
  } catch (error) {
    console.error("Error generating recommendations:", error)

    // Handle specific API key errors
    if (error instanceof Error) {
      if (error.message.includes("Incorrect API key") || error.message.includes("Invalid API key")) {
        throw new Error("INVALID_API_KEY")
      }
      if (error.message.includes("quota") || error.message.includes("billing")) {
        throw new Error("QUOTA_EXCEEDED")
      }
      if (error.message.includes("rate limit")) {
        throw new Error("RATE_LIMITED")
      }
    }

    throw new Error("AI_SERVICE_ERROR")
  }
}

export function convertAIRecommendationsToProducts(aiRecommendations: any[], prompt: string): Product[] {
  const productImages = {
    Electronics: [
      "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1586953208448-b95a79798f07?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1543512214-318c7553f230?w=400&h=400&fit=crop",
    ],
    Fashion: [
      "https://images.unsplash.com/photo-1601924994987-69e26d50dc26?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400&h=400&fit=crop",
    ],
    "Home & Garden": [
      "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1514228742587-6b1558fcf93a?w=400&h=400&fit=crop",
    ],
    Books: [
      "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1589829085413-56de8ae18c73?w=400&h=400&fit=crop",
    ],
    Sports: [
      "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1523362628745-0c100150b504?w=400&h=400&fit=crop",
    ],
    Beauty: [
      "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=400&h=400&fit=crop",
    ],
    Toys: [
      "https://images.unsplash.com/photo-1558877385-09c4d8b7b7a9?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=400&h=400&fit=crop",
      "https://images.unsplash.com/photo-1473968512647-3e447244af8f?w=400&h=400&fit=crop",
    ],
  }

  const brands = [
    "Apple",
    "Samsung",
    "Nike",
    "Adidas",
    "Amazon",
    "Sony",
    "Microsoft",
    "Canon",
    "Fitbit",
    "Bose",
    "Lego",
    "Nintendo",
  ]

  return aiRecommendations.map((rec, index) => {
    const categoryImages = productImages[rec.category as keyof typeof productImages] || productImages["Electronics"]
    const randomImage = categoryImages[index % categoryImages.length]
    const basePrice = Math.floor((rec.priceRange.min + rec.priceRange.max) / 2)
    const hasDiscount = Math.random() > 0.7

    return {
      id: `ai-rec-${index + 1}`,
      name: rec.name,
      description: rec.description,
      price: basePrice,
      originalPrice: hasDiscount ? Math.floor(basePrice * 1.2) : undefined,
      image: randomImage,
      rating: Math.max(4, Math.min(5, Math.floor(rec.suitabilityScore / 2) + 3)),
      reviewCount: Math.floor(Math.random() * 1000) + 100,
      category: rec.category,
      brand: brands[Math.floor(Math.random() * brands.length)],
      features: rec.tags.slice(0, 4),
      inStock: Math.random() > 0.05,
      fastShipping: Math.random() > 0.3,
      aiReasoning: rec.reasoning,
      suitabilityScore: rec.suitabilityScore,
      occasionMatch: rec.occasionMatch,
      ageAppropriate: rec.ageAppropriate,
    } as Product & { aiReasoning: string; suitabilityScore: number; occasionMatch: number; ageAppropriate: boolean }
  })
}
