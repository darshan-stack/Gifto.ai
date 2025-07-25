"use client"

import type React from "react"
import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Sparkles, Gift, Heart, Zap, Shield, Users, Brain, Target, Clock, AlertTriangle, Key } from "lucide-react"
import Image from "next/image"

export default function HomePage() {
  const [prompt, setPrompt] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStage, setLoadingStage] = useState("")
  const [errorMessage, setErrorMessage] = useState("")
  const [errorType, setErrorType] = useState("")
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim()) return

    setIsLoading(true)
    setErrorMessage("")
    setErrorType("")

    try {
      // Stage 1: Analyzing prompt
      setLoadingStage("Analyzing your request...")
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Stage 2: Understanding recipient
      setLoadingStage("Understanding the recipient...")
      await new Promise((resolve) => setTimeout(resolve, 1500))

      // Stage 3: Generating recommendations
      setLoadingStage("Finding perfect gifts with AI...")

      // Make API call to get AI recommendations
      const response = await fetch("/api/ai-recommendations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      })

      const data = await response.json()

      if (!response.ok) {
        if (data.fallback) {
          // Show specific error message but continue with fallback
          setErrorType(data.error)
          setErrorMessage(data.message)
          setLoadingStage("Using fallback recommendations...")
          await new Promise((resolve) => setTimeout(resolve, 1000))
          router.push(`/recommendations?q=${encodeURIComponent(prompt)}`)
          return
        }
        throw new Error(data.error || "Failed to get recommendations")
      }

      // Store the AI analysis and recommendations in sessionStorage
      sessionStorage.setItem("aiRecommendations", JSON.stringify(data))

      // Navigate to recommendations page
      router.push(`/recommendations?q=${encodeURIComponent(prompt)}&ai=true`)
    } catch (error) {
      console.error("Error getting AI recommendations:", error)
      setLoadingStage("Error occurred. Using fallback recommendations...")
      setErrorType("NETWORK_ERROR")
      setErrorMessage("Network error occurred. Using fallback recommendations.")
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Fallback to regular recommendations
      router.push(`/recommendations?q=${encodeURIComponent(prompt)}`)
    } finally {
      setIsLoading(false)
      setLoadingStage("")
    }
  }

  const getErrorIcon = () => {
    switch (errorType) {
      case "INVALID_API_KEY":
        return <Key className="h-5 w-5" />
      case "QUOTA_EXCEEDED":
        return <AlertTriangle className="h-5 w-5" />
      case "RATE_LIMITED":
        return <Clock className="h-5 w-5" />
      default:
        return <AlertTriangle className="h-5 w-5" />
    }
  }

  const getErrorColor = () => {
    switch (errorType) {
      case "INVALID_API_KEY":
        return "bg-red-50 border-red-200 text-red-800"
      case "QUOTA_EXCEEDED":
        return "bg-orange-50 border-orange-200 text-orange-800"
      case "RATE_LIMITED":
        return "bg-yellow-50 border-yellow-200 text-yellow-800"
      default:
        return "bg-yellow-50 border-yellow-200 text-yellow-800"
    }
  }

  const examplePrompts = [
    "Birthday gift for my 28-year-old brother who's a software engineer and loves gaming, coffee, and minimalist design",
    "Anniversary present for my wife who enjoys yoga, organic skincare, and reading mystery novels",
    "Christmas gift for my 12-year-old niece who loves art, K-pop, and wants to learn guitar",
    "Graduation gift for my best friend studying medicine who's stressed and needs relaxation",
    "Mother's Day gift for my mom who loves gardening, cooking Italian food, and vintage jewelry",
    "Father's Day present for my dad who's into woodworking, craft beer, and classic rock music",
  ]

  const featuredCategories = [
    { name: "Electronics", icon: "📱", count: "2,500+ items", aiMatch: "Tech enthusiasts" },
    { name: "Fashion", icon: "👗", count: "1,800+ items", aiMatch: "Style conscious" },
    { name: "Home & Garden", icon: "🏡", count: "3,200+ items", aiMatch: "Homebodies" },
    { name: "Beauty", icon: "💄", count: "1,200+ items", aiMatch: "Self-care lovers" },
    { name: "Sports", icon: "⚽", count: "900+ items", aiMatch: "Active lifestyle" },
    { name: "Books", icon: "📚", count: "5,000+ items", aiMatch: "Knowledge seekers" },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-pink-50">
      {/* Error Banner */}
      {errorMessage && (
        <div className={`border-b px-4 py-3 ${getErrorColor()}`}>
          <div className="container mx-auto">
            <div className="flex items-center gap-3">
              {getErrorIcon()}
              <div className="flex-1">
                <p className="text-sm font-medium">
                  {errorType === "INVALID_API_KEY" && "API Key Issue"}
                  {errorType === "QUOTA_EXCEEDED" && "Quota Exceeded"}
                  {errorType === "RATE_LIMITED" && "Rate Limited"}
                  {errorType === "MISSING_API_KEY" && "API Key Missing"}
                  {!["INVALID_API_KEY", "QUOTA_EXCEEDED", "RATE_LIMITED", "MISSING_API_KEY"].includes(errorType) &&
                    "Service Issue"}
                </p>
                <p className="text-xs opacity-90">{errorMessage}</p>
                {errorType === "INVALID_API_KEY" && (
                  <p className="text-xs opacity-75 mt-1">
                    Please update your OpenAI API key in the environment variables.
                  </p>
                )}
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setErrorMessage("")
                  setErrorType("")
                }}
                className="opacity-70 hover:opacity-100"
              >
                ×
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl">
                <Gift className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  GiftGenius AI
                </h1>
                <p className="text-sm text-gray-600">AI-Powered Gift Discovery</p>
              </div>
            </div>

            <div className="hidden md:flex items-center gap-6 text-sm text-gray-600">
              <span className="flex items-center gap-1">
                <Brain className="h-4 w-4 text-purple-600" />
                AI-Powered
              </span>
              <span className="flex items-center gap-1">
                <Users className="h-4 w-4" />
                50K+ Happy Customers
              </span>
              <span className="flex items-center gap-1">
                <Shield className="h-4 w-4" />
                Secure Shopping
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 bg-gradient-to-r from-purple-100 to-pink-100 text-purple-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
              <Brain className="h-4 w-4" />
              Advanced AI Gift Intelligence
            </div>

            <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
              <span className="bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 bg-clip-text text-transparent">
                AI Finds Your Perfect Gift
              </span>
              <br />
              <span className="text-gray-800">Every Time</span>
            </h1>

            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto leading-relaxed">
              Our advanced AI analyzes your description to understand the recipient's personality, interests, and your
              relationship - then recommends gifts that truly matter.
            </p>

            {/* AI Features */}
            <div className="flex flex-wrap justify-center gap-4 mb-12 text-sm">
              <div className="flex items-center gap-2 bg-white/70 px-4 py-2 rounded-full">
                <Target className="h-4 w-4 text-purple-600" />
                <span>Context-Aware</span>
              </div>
              <div className="flex items-center gap-2 bg-white/70 px-4 py-2 rounded-full">
                <Brain className="h-4 w-4 text-pink-600" />
                <span>Personality Analysis</span>
              </div>
              <div className="flex items-center gap-2 bg-white/70 px-4 py-2 rounded-full">
                <Clock className="h-4 w-4 text-orange-600" />
                <span>Real-time Processing</span>
              </div>
            </div>

            {/* Stats */}
            <div className="flex flex-wrap justify-center gap-8 mb-12 text-sm">
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">100K+</div>
                <div className="text-gray-600">Products Analyzed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-pink-600">95%</div>
                <div className="text-gray-600">AI Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">3 Sec</div>
                <div className="text-gray-600">Average Response</div>
              </div>
            </div>
          </div>

          {/* AI Chat Interface */}
          <Card className="p-8 shadow-xl border-0 bg-white/70 backdrop-blur-sm mb-12 relative overflow-hidden">
            {/* AI Processing Overlay */}
            {isLoading && (
              <div className="absolute inset-0 bg-white/90 backdrop-blur-sm flex items-center justify-center z-10">
                <div className="text-center">
                  <div className="relative mb-6">
                    <div className="w-16 h-16 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mx-auto"></div>
                    <Brain className="h-6 w-6 text-purple-600 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">AI is thinking...</h3>
                  <p className="text-purple-600 font-medium">{loadingStage}</p>
                  <div className="mt-4 flex justify-center gap-1">
                    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                    <div
                      className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.1s" }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    ></div>
                  </div>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="relative">
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the person in detail... (e.g., 'Birthday gift for my 25-year-old tech-savvy brother who loves gaming, coffee, and minimalist design. He's introverted but social with close friends, works as a software engineer, and has been stressed lately.')"
                  className="min-h-[140px] text-lg resize-none border-2 border-gray-200 focus:border-purple-400 rounded-xl p-4 pr-16"
                  disabled={isLoading}
                />
                <Button
                  type="submit"
                  disabled={!prompt.trim() || isLoading}
                  className="absolute bottom-3 right-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 rounded-lg px-4 py-2"
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      AI Processing...
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <Brain className="h-4 w-4" />
                      Get AI Recommendations
                    </div>
                  )}
                </Button>
              </div>

              <div className="text-xs text-gray-500 bg-blue-50 p-3 rounded-lg">
                <strong>💡 Pro Tip:</strong> The more details you provide about the person's interests, personality,
                lifestyle, and your relationship, the better our AI can recommend the perfect gift!
              </div>
            </form>

            {/* Enhanced Example Prompts */}
            <div className="mt-8">
              <p className="text-sm font-medium text-gray-700 mb-4 flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-purple-600" />
                Try these detailed examples:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {examplePrompts.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setPrompt(example)}
                    className="text-left p-4 rounded-lg border border-gray-200 hover:border-purple-300 hover:bg-purple-50 transition-all duration-200 group"
                    disabled={isLoading}
                  >
                    <div className="flex items-start gap-3">
                      <div className="p-1 bg-purple-100 rounded group-hover:bg-purple-200 transition-colors">
                        <Heart className="h-3 w-3 text-purple-600" />
                      </div>
                      <span className="text-sm text-gray-700 group-hover:text-purple-700 line-clamp-3">{example}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </Card>

          {/* API Key Setup Help */}
          {errorType === "INVALID_API_KEY" && (
            <Card className="p-6 mb-12 bg-red-50 border-red-200">
              <div className="flex items-start gap-4">
                <Key className="h-6 w-6 text-red-600 mt-1" />
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
                    <Key className="h-4 w-4 mr-2" />
                    Get New API Key
                  </Button>
                </div>
              </div>
            </Card>
          )}

          {/* AI-Enhanced Categories */}
          <div className="mb-16">
            <h2 className="text-3xl font-bold text-center mb-4 text-gray-800">AI-Curated Categories</h2>
            <p className="text-center text-gray-600 mb-8">
              Our AI understands different personality types and matches them to perfect categories
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {featuredCategories.map((category, index) => (
                <Card
                  key={index}
                  className="p-6 text-center hover:shadow-lg transition-all duration-300 cursor-pointer group relative overflow-hidden"
                >
                  <div className="absolute top-2 right-2">
                    <Brain className="h-3 w-3 text-purple-400" />
                  </div>
                  <div className="text-4xl mb-3">{category.icon}</div>
                  <h3 className="font-semibold text-gray-800 mb-1 group-hover:text-purple-600 transition-colors">
                    {category.name}
                  </h3>
                  <p className="text-xs text-gray-500 mb-1">{category.count}</p>
                  <p className="text-xs text-purple-600 font-medium">{category.aiMatch}</p>
                </Card>
              ))}
            </div>
          </div>

          {/* Enhanced Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            <div className="text-center p-6">
              <div className="w-16 h-16 bg-purple-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Brain className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-3">Advanced AI Analysis</h3>
              <p className="text-gray-600">
                Our AI analyzes personality traits, interests, relationships, and occasions to find gifts that create
                genuine connections
              </p>
            </div>

            <div className="text-center p-6">
              <div className="w-16 h-16 bg-pink-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Target className="h-8 w-8 text-pink-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-3">Context-Aware Matching</h3>
              <p className="text-gray-600">
                Every recommendation comes with detailed reasoning explaining why it's perfect for your specific
                situation
              </p>
            </div>

            <div className="text-center p-6">
              <div className="w-16 h-16 bg-orange-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Zap className="h-8 w-8 text-orange-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-3">Real-Time Processing</h3>
              <p className="text-gray-600">
                Get personalized recommendations in seconds, powered by the latest AI technology and real-time analysis
              </p>
            </div>
          </div>

          {/* Sample Products Preview */}
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">AI Success Stories</h2>
            <p className="text-gray-600 mb-8">See how our AI found the perfect gifts</p>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {[
                {
                  name: "Noise-Canceling Headphones",
                  price: "$299",
                  image: "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=300&h=300&fit=crop",
                  aiReason: "Perfect for introverted tech worker who values focus",
                },
                {
                  name: "Smart Yoga Mat",
                  price: "$149",
                  image: "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=300&h=300&fit=crop",
                  aiReason: "Ideal for wellness-focused anniversary gift",
                },
                {
                  name: "Digital Art Tablet",
                  price: "$199",
                  image: "https://images.unsplash.com/photo-1558877385-09c4d8b7b7a9?w=300&h=300&fit=crop",
                  aiReason: "Matches creative teen's artistic interests",
                },
                {
                  name: "Aromatherapy Diffuser",
                  price: "$79",
                  image: "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=300&h=300&fit=crop",
                  aiReason: "Perfect for stressed medical student's relaxation",
                },
              ].map((product, index) => (
                <Card key={index} className="overflow-hidden hover:shadow-lg transition-all duration-300 group">
                  <div className="aspect-square overflow-hidden relative">
                    <Image
                      src={product.image || "/placeholder.svg"}
                      alt={product.name}
                      width={300}
                      height={300}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute top-2 left-2">
                      <Badge className="bg-purple-600 text-white text-xs">
                        <Brain className="h-3 w-3 mr-1" />
                        AI Pick
                      </Badge>
                    </div>
                  </div>
                  <div className="p-4">
                    <h3 className="font-medium text-gray-800 mb-1">{product.name}</h3>
                    <p className="text-purple-600 font-semibold mb-2">{product.price}</p>
                    <p className="text-xs text-gray-600 italic">"{product.aiReason}"</p>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Gift className="h-6 w-6 text-purple-400" />
                <span className="text-xl font-bold">GiftGenius AI</span>
              </div>
              <p className="text-gray-400">The world's most advanced AI gift recommendation engine.</p>
            </div>

            <div>
              <h3 className="font-semibold mb-4">AI Features</h3>
              <ul className="space-y-2 text-gray-400">
                <li>Personality Analysis</li>
                <li>Context Understanding</li>
                <li>Relationship Mapping</li>
                <li>Occasion Matching</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Support</h3>
              <ul className="space-y-2 text-gray-400">
                <li>How AI Works</li>
                <li>Contact Us</li>
                <li>Returns</li>
                <li>Shipping Info</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Company</h3>
              <ul className="space-y-2 text-gray-400">
                <li>About Our AI</li>
                <li>Privacy Policy</li>
                <li>Terms of Service</li>
                <li>Careers</li>
              </ul>
            </div>
          </div>

          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 GiftGenius AI. Powered by advanced artificial intelligence.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
