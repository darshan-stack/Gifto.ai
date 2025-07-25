import { type NextRequest, NextResponse } from "next/server"
import {
  analyzeRecipient,
  generateAIRecommendations,
  convertAIRecommendationsToProducts,
} from "@/lib/ai-recommendations"

export async function POST(req: Request) {
  const { prompt } = await req.json();

  // Call the Python FastAPI service
  const response = await fetch('http://localhost:8000/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });

  if (!response.ok) {
    return new Response(JSON.stringify({ error: 'Python service error' }), { status: 500 });
  }

  const data = await response.json();
  return new Response(JSON.stringify({ recommendations: data.recommendations }), { status: 200 });
}
