import { type NextRequest, NextResponse } from "next/server"

interface GreetingCardRequest {
  recipient_name: string
  occasion: string
  message_style: string // funny, formal, emotional, romantic
  personal_message?: string
}

export async function POST(req: NextRequest) {
  try {
    const body: GreetingCardRequest = await req.json()
    const { recipient_name, occasion, message_style, personal_message } = body

    if (!recipient_name || !occasion || !message_style) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    // Call the Python FastAPI service for greeting card generation
    const response = await fetch('http://localhost:8000/greeting-card', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recipient_name,
        occasion,
        message_style,
        personal_message
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Greeting card service error:', errorText)
      return NextResponse.json({ 
        error: 'Greeting card service error',
        details: errorText
      }, { status: 500 })
    }

    const data = await response.json()
    
    return NextResponse.json({
      success: true,
      data: {
        card_id: data.card_id,
        content: data.content,
        created_at: data.created_at
      },
    })
  } catch (error) {
    console.error("Error in greeting card API:", error)
    return NextResponse.json(
      {
        error: "UNKNOWN_ERROR",
        message: "An unexpected error occurred. Please try again.",
      },
      { status: 500 },
    )
  }
} 