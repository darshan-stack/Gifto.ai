"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { MessageSquare, Heart, Sparkles, Gift, Star, ShoppingCart, History, Filter } from "lucide-react"
import { ChatSidebar } from "@/components/chat-sidebar"
import { WishlistIcon } from "@/components/wishlist-icon"

interface Product {
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
  aiReasoning?: string
  suitabilityScore?: number
  occasionMatch?: number
  ageAppropriate?: boolean
}

interface ChatHistory {
  id: string
  prompt: string
  timestamp: Date
  recipient_profile?: any
  occasion_info?: any
}

export default function HomePage() {
  const [prompt, setPrompt] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [recommendations, setRecommendations] = useState<string>("")
  const [errorType, setErrorType] = useState<string | null>(null)
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(false)
  const [currentConversation, setCurrentConversation] = useState<ChatHistory | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim()) return

    setIsLoading(true)
    setErrorType(null)

    try {
      const response = await fetch("/api/ai-recommendations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      })

      const data = await response.json()

      if (data.success) {
        setRecommendations(data.data.recommendations)
        // Save to chat history
        saveToChatHistory(prompt, data.data.recipient_profile, data.data.occasion_info)
      } else {
        setErrorType(data.error || "UNKNOWN_ERROR")
        setRecommendations("")
      }
    } catch (error) {
      console.error("Error:", error)
      setErrorType("UNKNOWN_ERROR")
      setRecommendations("")
    } finally {
      setIsLoading(false)
    }
  }

  const saveToChatHistory = (prompt: string, recipient_profile?: any, occasion_info?: any) => {
    const saved = localStorage.getItem("gift-chat-history")
    let history = saved ? JSON.parse(saved) : []
    
    const newHistory: ChatHistory = {
      id: Date.now().toString(),
      prompt,
      timestamp: new Date(),
      recipient_profile,
      occasion_info
    }
    
    history = [newHistory, ...history.slice(0, 49)] // Keep last 50 conversations
    localStorage.setItem("gift-chat-history", JSON.stringify(history))
  }

  const loadConversation = (history: ChatHistory) => {
    setPrompt(history.prompt)
    setCurrentConversation(history)
    setIsChatSidebarOpen(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Gift className="h-8 w-8 text-blue-600" />
                <h1 className="text-xl font-bold text-gray-900">Gifto.ai</h1>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsChatSidebarOpen(!isChatSidebarOpen)}
                className="flex items-center space-x-2"
              >
                <History className="h-5 w-5" />
                <span className="hidden sm:inline">History</span>
              </Button>
              
              <WishlistIcon />
            </div>
          </div>
        </div>
      </header>

      {/* Chat Sidebar */}
      <ChatSidebar
        isOpen={isChatSidebarOpen}
        onClose={() => setIsChatSidebarOpen(false)}
        onLoadConversation={loadConversation}
        currentConversation={currentConversation}
      />

      <main className="max-w-4xl mx-auto px-4 py-8">
          {/* Hero Section */}
          <div className="text-center mb-12">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <Sparkles className="h-8 w-8 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900">
              AI-Powered Gift Recommendations
            </h1>
            <Sparkles className="h-8 w-8 text-blue-600" />
          </div>
          <p className="text-xl text-gray-600 mb-8">
            Get personalized gift suggestions powered by advanced AI. 
            Just describe who you're shopping for and let our AI find the perfect gifts!
          </p>
              </div>

        {/* Input Form */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <MessageSquare className="h-5 w-5 text-blue-600" />
              <span>Tell us about your gift recipient</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="prompt" className="text-sm font-medium text-gray-700">
                  Describe who you're shopping for:
                </label>
                <Input
                  id="prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="e.g., Birthday present for my 10-year-old nephew who loves science and robots"
                  className="h-12 text-lg"
                  disabled={isLoading}
                />
              </div>
              
              <div className="flex flex-wrap gap-2">
                <Badge 
                  variant="secondary" 
                  className="cursor-pointer hover:bg-blue-100"
                  onClick={() => setPrompt("Birthday gift for my mom who loves gardening")}
                >
                  Mom - Gardening
                </Badge>
                <Badge 
                  variant="secondary" 
                  className="cursor-pointer hover:bg-blue-100"
                  onClick={() => setPrompt("Anniversary gift for my husband who loves cooking")}
                >
                  Husband - Cooking
                </Badge>
                <Badge 
                  variant="secondary" 
                  className="cursor-pointer hover:bg-blue-100"
                  onClick={() => setPrompt("Graduation gift for my friend who loves technology")}
                >
                  Friend - Technology
                </Badge>
                <Badge 
                  variant="secondary" 
                  className="cursor-pointer hover:bg-blue-100"
                  onClick={() => setPrompt("Christmas gift for my 5-year-old daughter who loves princesses")}
                >
                  Daughter - Princesses
                </Badge>
              </div>

                <Button
                  type="submit"
                disabled={isLoading || !prompt.trim()} 
                className="w-full h-12 text-lg"
                >
                  {isLoading ? (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    <span>Finding perfect gifts...</span>
                    </div>
                  ) : (
                  <div className="flex items-center space-x-2">
                    <Sparkles className="h-5 w-5" />
                    <span>Get AI Recommendations</span>
                    </div>
                  )}
                </Button>
            </form>
          </CardContent>
        </Card>

        {/* Error Messages */}
        {errorType === "MISSING_API_KEY" && (
          <Card className="p-6 mb-12 bg-red-50 border-red-200">
            <div className="flex items-start gap-4">
              <MessageSquare className="h-6 w-6 text-red-600 mt-1" />
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-red-800 mb-2">API Key Setup Required</h3>
                <p className="text-red-700 mb-4">
                  Your OpenAI API key is not configured. Please add OPENAI_API_KEY to your environment variables.
                </p>
              </div>
            </div>
          </Card>
        )}

          {errorType === "INVALID_API_KEY" && (
            <Card className="p-6 mb-12 bg-red-50 border-red-200">
              <div className="flex items-start gap-4">
              <MessageSquare className="h-6 w-6 text-red-600 mt-1" />
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-red-800 mb-2">API Key Setup Required</h3>
                  <p className="text-red-700 mb-4">
                    Your OpenAI API key is invalid or has been revoked. Here's how to fix it:
                  </p>
                  <ol className="list-decimal list-inside space-y-2 text-sm text-red-700 mb-4">
                    <li>
                      Go to{" "}
                      <a
                        href="https://platform.openai.com/api-keys"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline"
                      >
                        OpenAI Platform
                      </a>
                    </li>
                    <li>Create a new API key</li>
                    <li>Copy the key (starts with sk-)</li>
                    <li>Add it to your environment variables as OPENAI_API_KEY</li>
                  </ol>
                  <Button
                    onClick={() => window.open("https://platform.openai.com/api-keys", "_blank")}
                    className="bg-red-600 hover:bg-red-700 text-white"
                  >
                  <MessageSquare className="h-4 w-4 mr-2" />
                    Get New API Key
                  </Button>
                </div>
              </div>
            </Card>
          )}

        {/* Recommendations */}
        {recommendations && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Gift className="h-5 w-5 text-green-600" />
                <span>AI Recommendations</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose max-w-none">
                <div className="whitespace-pre-wrap text-gray-700 leading-relaxed">
                  {recommendations}
          </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Features Section */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="text-center p-6">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Sparkles className="h-6 w-6 text-blue-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">AI-Powered</h3>
            <p className="text-gray-600 text-sm">
              Advanced AI analyzes your prompt to understand recipient preferences and suggest perfect gifts.
            </p>
          </Card>
          
          <Card className="text-center p-6">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Heart className="h-6 w-6 text-green-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Personalized</h3>
            <p className="text-gray-600 text-sm">
              Get recommendations tailored to the recipient's age, interests, and the occasion.
            </p>
          </Card>
          
          <Card className="text-center p-6">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <ShoppingCart className="h-6 w-6 text-purple-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Actionable</h3>
            <p className="text-gray-600 text-sm">
              Each recommendation includes reasoning and can be easily saved to your wishlist.
            </p>
                </Card>
        </div>
      </main>
    </div>
  )
}
