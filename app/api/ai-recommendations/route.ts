import { type NextRequest, NextResponse } from "next/server"

interface RecipientProfile {
  age?: number
  gender?: string
  interests: string[]
  hobbies: string[]
  relationship?: string
  personality: string[]
  lifestyle: string[]
  preferences: string[]
}

interface OccasionInfo {
  occasion: string
  mood?: string
  formality?: string
  budget_range?: { min: number; max: number }
}

interface FilterOptions {
  category?: string
  price_min?: number
  price_max?: number
  eco_friendly?: boolean
  handmade?: boolean
  local?: boolean
  rating_min?: number
  sort_by?: string
}

interface PromptRequest {
  prompt: string
  recipient_profile?: RecipientProfile
  occasion_info?: OccasionInfo
  filter_options?: FilterOptions
}

export async function POST(req: NextRequest) {
  try {
    const body: PromptRequest = await req.json()
    const { prompt, recipient_profile, occasion_info, filter_options } = body

    if (!prompt || typeof prompt !== "string") {
      return NextResponse.json({ error: "Valid prompt is required" }, { status: 400 })
    }

    // Call the Python FastAPI service with enhanced features
    const response = await fetch('http://localhost:8000/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        recipient_profile,
        occasion_info,
        filter_options
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Python service error:', errorText)
      return NextResponse.json({ 
        error: 'Python service error',
        details: errorText
      }, { status: 500 })
    }

    const data = await response.json()

    return NextResponse.json({
      success: true,
      data: {
        recommendations: data.recommendations,
        recipient_profile: data.recipient_profile,
        occasion_info: data.occasion_info,
        filter_options: data.filter_options,
        totalRecommendations: 50, // Fixed count from backend
      },
    })
  } catch (error) {
    console.error("Error in AI recommendations API:", error)
    return NextResponse.json(
      {
        error: "UNKNOWN_ERROR",
        fallback: true,
        message: "An unexpected error occurred. Please try again.",
      },
      { status: 500 },
    )
  }
}
