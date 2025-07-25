"use client"

import { useState, useEffect, Suspense } from "react"
import { useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  ArrowLeft,
  Star,
  Heart,
  ShoppingCart,
  Truck,
  Gift,
  Search,
  Filter,
  Grid3X3,
  List,
  Share2,
  Bookmark,
  Brain,
  Lightbulb,
} from "lucide-react"
import Link from "next/link"
import Image from "next/image"
import type { Product, RecipientAnalysis, AIRecommendationResponse } from "@/lib/types"
import { generateMockProducts, formatPrice } from "@/lib/utils"

function RecommendationsContent() {
  const searchParams = useSearchParams()
  const query = searchParams.get("q") || "Perfect Gift"
  const isAIPowered = searchParams.get("ai") === "true"

  const [products, setProducts] = useState<Product[]>([])
  const [filteredProducts, setFilteredProducts] = useState<Product[]>([])
  const [recipientAnalysis, setRecipientAnalysis] = useState<RecipientAnalysis | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")
  const [sortBy, setSortBy] = useState("ai-score")
  const [filterCategory, setFilterCategory] = useState("all")
  const [priceRange, setPriceRange] = useState("all")
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid")
  const [favorites, setFavorites] = useState<Set<string>>(new Set())
  const [showAnalysis, setShowAnalysis] = useState(true)

  useEffect(() => {
    const loadProducts = async () => {
      setLoading(true)

      try {
        if (isAIPowered) {
          // Try to get AI recommendations from sessionStorage
          const storedData = sessionStorage.getItem("aiRecommendations")
          if (storedData) {
            const aiData: AIRecommendationResponse = JSON.parse(storedData)
            setProducts(aiData.data.products)
            setFilteredProducts(aiData.data.products)
            setRecipientAnalysis(aiData.data.recipientAnalysis)
          } else {
            // Fallback to mock data if no AI data available
            const mockProducts = generateMockProducts(query)
            setProducts(mockProducts)
            setFilteredProducts(mockProducts)
          }
        } else {
          // Use mock data for non-AI requests
          const mockProducts = generateMockProducts(query)
          setProducts(mockProducts)
          setFilteredProducts(mockProducts)
        }
      } catch (error) {
        console.error("Error loading products:", error)
        // Fallback to mock data
        const mockProducts = generateMockProducts(query)
        setProducts(mockProducts)
        setFilteredProducts(mockProducts)
      } finally {
        setLoading(false)
      }
    }

    loadProducts()
  }, [query, isAIPowered])

  useEffect(() => {
    let filtered = products.filter(
      (product) =>
        product.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        product.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        product.brand.toLowerCase().includes(searchTerm.toLowerCase()),
    )

    if (filterCategory !== "all") {
      filtered = filtered.filter((product) => product.category === filterCategory)
    }

    if (priceRange !== "all") {
      const [min, max] = priceRange.split("-").map(Number)
      filtered = filtered.filter((product) => {
        if (max) return product.price >= min && product.price <= max
        return product.price >= min
      })
    }

    // Sort products
    switch (sortBy) {
      case "ai-score":
        filtered.sort((a, b) => (b.suitabilityScore || 0) - (a.suitabilityScore || 0))
        break
      case "occasion-match":
        filtered.sort((a, b) => (b.occasionMatch || 0) - (a.occasionMatch || 0))
        break
      case "price-low":
        filtered.sort((a, b) => a.price - b.price)
        break
      case "price-high":
        filtered.sort((a, b) => b.price - a.price)
        break
      case "rating":
        filtered.sort((a, b) => b.rating - a.rating)
        break
      case "reviews":
        filtered.sort((a, b) => b.reviewCount - a.reviewCount)
        break
      case "name":
        filtered.sort((a, b) => a.name.localeCompare(b.name))
        break
      default:
        break
    }

    setFilteredProducts(filtered)
  }, [products, searchTerm, sortBy, filterCategory, priceRange])

  const toggleFavorite = (productId: string) => {
    const newFavorites = new Set(favorites)
    if (newFavorites.has(productId)) {
      newFavorites.delete(productId)
    } else {
      newFavorites.add(productId)
    }
    setFavorites(newFavorites)
  }

  const categories = [...new Set(products.map((p) => p.category))]

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center gap-4 mb-8">
            <Link href="/">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 border-2 border-purple-200 border-t-purple-600 rounded-full animate-spin"></div>
              <span className="text-lg font-semibold">AI is analyzing and finding perfect gifts...</span>
            </div>
          </div>

          <div className="bg-white rounded-lg border p-4 mb-6">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="h-10 bg-gray-200 rounded animate-pulse" />
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {Array.from({ length: 12 }).map((_, i) => (
              <Card key={i} className="p-4">
                <div className="aspect-square bg-gray-200 rounded-lg mb-4 animate-pulse" />
                <div className="space-y-2">
                  <div className="h-4 bg-gray-200 rounded animate-pulse" />
                  <div className="h-4 bg-gray-200 rounded w-3/4 animate-pulse" />
                  <div className="h-6 bg-gray-200 rounded w-1/2 animate-pulse" />
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="sm">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back
                </Button>
              </Link>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl font-semibold text-gray-800">
                    {isAIPowered ? "AI-Powered" : ""} Gift Recommendations
                  </h1>
                  {isAIPowered && <Brain className="h-5 w-5 text-purple-600" />}
                </div>
                <p className="text-sm text-gray-600">
                  {filteredProducts.length} {isAIPowered ? "AI-curated" : ""} results for "{query}"
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Button
                  variant={viewMode === "grid" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setViewMode("grid")}
                >
                  <Grid3X3 className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setViewMode("list")}
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>

              <div className="flex items-center gap-2">
                <Gift className="h-5 w-5 text-purple-600" />
                <span className="font-medium text-purple-600">GiftGenius AI</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        {/* AI Analysis Card */}
        {isAIPowered && recipientAnalysis && showAnalysis && (
          <Card className="mb-6 p-6 bg-gradient-to-r from-purple-50 to-pink-50 border-purple-200">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-600" />
                <h3 className="text-lg font-semibold text-gray-800">AI Analysis Results</h3>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowAnalysis(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ×
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <h4 className="font-medium text-gray-700 mb-2">Recipient Profile</h4>
                <div className="space-y-1 text-sm text-gray-600">
                  {recipientAnalysis.age && <p>Age: {recipientAnalysis.age}</p>}
                  {recipientAnalysis.gender && <p>Gender: {recipientAnalysis.gender}</p>}
                  <p>Relationship: {recipientAnalysis.relationship}</p>
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-2">Interests</h4>
                <div className="flex flex-wrap gap-1">
                  {recipientAnalysis.interests.slice(0, 3).map((interest, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {interest}
                    </Badge>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-2">Occasion</h4>
                <Badge className="bg-purple-100 text-purple-700">{recipientAnalysis.occasion}</Badge>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-2">Budget</h4>
                <p className="text-sm text-gray-600">
                  {recipientAnalysis.budget
                    ? `$${recipientAnalysis.budget.min} - $${recipientAnalysis.budget.max}`
                    : "Not specified"}
                </p>
              </div>
            </div>
          </Card>
        )}

        {/* Filters and Search */}
        <div className="bg-white rounded-lg border p-4 mb-6 shadow-sm">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
            <div className="relative lg:col-span-2">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search products..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>

            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger>
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                {isAIPowered && (
                  <>
                    <SelectItem value="ai-score">AI Suitability Score</SelectItem>
                    <SelectItem value="occasion-match">Occasion Match</SelectItem>
                  </>
                )}
                <SelectItem value="relevance">Most Relevant</SelectItem>
                <SelectItem value="price-low">Price: Low to High</SelectItem>
                <SelectItem value="price-high">Price: High to Low</SelectItem>
                <SelectItem value="rating">Highest Rated</SelectItem>
                <SelectItem value="reviews">Most Reviews</SelectItem>
                <SelectItem value="name">Name A-Z</SelectItem>
              </SelectContent>
            </Select>

            <Select value={filterCategory} onValueChange={setFilterCategory}>
              <SelectTrigger>
                <SelectValue placeholder="Category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                {categories.map((category) => (
                  <SelectItem key={category} value={category}>
                    {category}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={priceRange} onValueChange={setPriceRange}>
              <SelectTrigger>
                <SelectValue placeholder="Price Range" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Prices</SelectItem>
                <SelectItem value="0-50">Under $50</SelectItem>
                <SelectItem value="50-100">$50 - $100</SelectItem>
                <SelectItem value="100-200">$100 - $200</SelectItem>
                <SelectItem value="200-500">$200 - $500</SelectItem>
                <SelectItem value="500">$500+</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" className="flex items-center gap-2 bg-transparent">
              <Filter className="h-4 w-4" />
              More Filters
            </Button>
          </div>
        </div>

        {/* Results Summary */}
        <div className="flex items-center justify-between mb-6">
          <div className="text-sm text-gray-600">
            Showing {filteredProducts.length} of {products.length} {isAIPowered ? "AI-curated" : ""} products
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Share2 className="h-4 w-4 mr-1" />
              Share Results
            </Button>
            <Button variant="outline" size="sm">
              <Bookmark className="h-4 w-4 mr-1" />
              Save Search
            </Button>
          </div>
        </div>

        {/* Products Grid/List */}
        <div
          className={
            viewMode === "grid" ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6" : "space-y-4"
          }
        >
          {filteredProducts.map((product) => (
            <Card
              key={product.id}
              className={`group hover:shadow-lg transition-all duration-300 overflow-hidden ${
                viewMode === "list" ? "flex" : ""
              }`}
            >
              <div className={`relative ${viewMode === "list" ? "w-48 flex-shrink-0" : ""}`}>
                <div className={`overflow-hidden ${viewMode === "list" ? "h-48" : "aspect-square"}`}>
                  <Image
                    src={product.image || "/placeholder.svg"}
                    alt={product.name}
                    width={400}
                    height={400}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    loading="lazy"
                  />
                </div>

                <Button
                  size="sm"
                  variant="ghost"
                  className={`absolute top-2 right-2 bg-white/80 hover:bg-white ${
                    favorites.has(product.id) ? "text-red-500" : ""
                  }`}
                  onClick={() => toggleFavorite(product.id)}
                >
                  <Heart className={`h-4 w-4 ${favorites.has(product.id) ? "fill-current" : ""}`} />
                </Button>

                {/* AI Score Badge */}
                {isAIPowered && product.suitabilityScore && (
                  <Badge className="absolute top-2 left-2 bg-purple-600 text-white">
                    <Brain className="h-3 w-3 mr-1" />
                    {product.suitabilityScore}/10
                  </Badge>
                )}

                {product.originalPrice && (
                  <Badge className="absolute top-10 left-2 bg-red-500">
                    {Math.round((1 - product.price / product.originalPrice) * 100)}% OFF
                  </Badge>
                )}

                {product.fastShipping && (
                  <Badge variant="secondary" className="absolute bottom-2 left-2">
                    <Truck className="h-3 w-3 mr-1" />
                    Fast Shipping
                  </Badge>
                )}
              </div>

              <div className={`p-4 ${viewMode === "list" ? "flex-1" : ""}`}>
                <div className="flex items-center gap-1 mb-2">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Star
                      key={i}
                      className={`h-4 w-4 ${i < product.rating ? "text-yellow-400 fill-current" : "text-gray-300"}`}
                    />
                  ))}
                  <span className="text-sm text-gray-600 ml-1">({product.reviewCount.toLocaleString()})</span>
                </div>

                <h3
                  className={`font-semibold text-gray-800 mb-2 group-hover:text-purple-600 transition-colors ${
                    viewMode === "list" ? "text-lg" : "line-clamp-2"
                  }`}
                >
                  {product.name}
                </h3>

                <p className={`text-sm text-gray-600 mb-3 ${viewMode === "list" ? "" : "line-clamp-2"}`}>
                  {product.description}
                </p>

                {/* AI Reasoning */}
                {isAIPowered && product.aiReasoning && (
                  <div className="mb-3 p-2 bg-purple-50 rounded-lg">
                    <div className="flex items-start gap-2">
                      <Lightbulb className="h-4 w-4 text-purple-600 mt-0.5 flex-shrink-0" />
                      <p className="text-xs text-purple-700 italic">{product.aiReasoning}</p>
                    </div>
                  </div>
                )}

                <div className="flex items-center gap-2 mb-3 flex-wrap">
                  <Badge variant="outline" className="text-xs">
                    {product.category}
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    {product.brand}
                  </Badge>
                  {product.features.slice(0, 2).map((feature, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {feature}
                    </Badge>
                  ))}
                </div>

                <div className={`flex items-center ${viewMode === "list" ? "justify-between" : "justify-between"}`}>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold text-gray-800">{formatPrice(product.price)}</span>
                    {product.originalPrice && (
                      <span className="text-sm text-gray-500 line-through">{formatPrice(product.originalPrice)}</span>
                    )}
                  </div>

                  <Button size="sm" className="bg-purple-600 hover:bg-purple-700" disabled={!product.inStock}>
                    <ShoppingCart className="h-4 w-4 mr-1" />
                    {product.inStock ? "Add to Cart" : "Out of Stock"}
                  </Button>
                </div>

                {!product.inStock && (
                  <Badge variant="destructive" className="w-full mt-2 justify-center">
                    Out of Stock - Notify When Available
                  </Badge>
                )}
              </div>
            </Card>
          ))}
        </div>

        {filteredProducts.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <Search className="h-10 w-10 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-800 mb-3">No products found</h3>
            <p className="text-gray-600 mb-6">Try adjusting your search terms or filters</p>
            <div className="flex justify-center gap-4">
              <Button onClick={() => setSearchTerm("")}>Clear Search</Button>
              <Button
                variant="outline"
                onClick={() => {
                  setFilterCategory("all")
                  setPriceRange("all")
                  setSortBy(isAIPowered ? "ai-score" : "relevance")
                }}
              >
                Reset Filters
              </Button>
            </div>
          </div>
        )}

        {/* Load More Button */}
        {filteredProducts.length > 0 && filteredProducts.length >= 20 && (
          <div className="text-center mt-12">
            <Button variant="outline" size="lg">
              Load More AI Recommendations
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}

export default function RecommendationsPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-600">AI is analyzing your request...</p>
          </div>
        </div>
      }
    >
      <RecommendationsContent />
    </Suspense>
  )
}
