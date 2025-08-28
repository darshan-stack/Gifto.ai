import { type NextRequest, NextResponse } from "next/server"

interface ThankYouRequest {
  gift_name: string
  sender_name: string
  occasion: string
  message_style: string
}

export async function POST(req: NextRequest) {
  try {
    const body: ThankYouRequest = await req.json()
    const { gift_name, sender_name, occasion, message_style } = body

    if (!gift_name || !sender_name || !occasion || !message_style) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    // Call the Python FastAPI service for thank you note generation
    const response = await fetch('http://localhost:8000/thank-you', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        gift_name,
        sender_name,
        occasion,
        message_style
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Thank you note service error:', errorText)
      return NextResponse.json({ 
        error: 'Thank you note service error',
        details: errorText
      }, { status: 500 })
    }

    const data = await response.json()
    
    return NextResponse.json({
      success: true,
      data: {
        note_id: data.note_id,
        content: data.content,
        created_at: data.created_at
      },
    })
  } catch (error) {
    console.error("Error in thank you note API:", error)
    return NextResponse.json(
      {
        error: "UNKNOWN_ERROR",
        message: "An unexpected error occurred. Please try again.",
      },
      { status: 500 },
    )
  }
} 