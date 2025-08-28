"use client"

import { useState, useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import { Heart, ShoppingCart, Star, ArrowLeft, Share2, Truck, Shield, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

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

export default function ProductDetailPage() {
  const params = useParams()
  const router = useRouter()
  const [product, setProduct] = useState<Product | null>(null)
  const [loading, setLoading] = useState(true)
  const [isInWishlist, setIsInWishlist] = useState(false)
  const [quantity, setQuantity] = useState(1)

  useEffect(() => {
    if (params.id) {
      loadProduct(params.id as string)
      checkWishlistStatus(params.id as string)
    }
  }, [params.id])

  const loadProduct = async (productId: string) => {
    try {
      // In a real app, you'd fetch from your API
      // For now, we'll create a mock product
      const mockProduct: Product = {
        id: productId,
        name: "Premium Gift Item",
        description: "This is a high-quality gift item perfect for various occasions. Made with premium materials and designed for lasting enjoyment.",
        price: 2999,
        originalPrice: 3999,
        image: "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&h=400&fit=crop",
        rating: 4.5,
        reviewCount: 1247,
        category: "Electronics",
        brand: "Premium Brand",
        features: ["High Quality", "Durable", "Eco-friendly", "1 Year Warranty"],
        inStock: true,
        fastShipping: true,
        aiReasoning: "Perfect for tech enthusiasts who appreciate quality gadgets",
        suitabilityScore: 9,
        occasionMatch: 8,
        ageAppropriate: true
      }
      setProduct(mockProduct)
    } catch (error) {
      console.error("Error loading product:", error)
    } finally {
      setLoading(false)
    }
  }

  const checkWishlistStatus = (productId: string) => {
    const saved = localStorage.getItem("wishlist-default")
    if (saved) {
      try {
        const wishlist = JSON.parse(saved)
        setIsInWishlist(wishlist.some((item: any) => item.id === productId))
      } catch (error) {
        console.error("Error checking wishlist status:", error)
      }
    }
  }

  const toggleWishlist = () => {
    if (!product) return

    const saved = localStorage.getItem("wishlist-default")
    let wishlist = saved ? JSON.parse(saved) : []

    if (isInWishlist) {
      wishlist = wishlist.filter((item: any) => item.id !== product.id)
    } else {
      wishlist.push({
        id: product.id,
        name: product.name,
        price: product.price,
        image: product.image,
        category: product.category,
        addedAt: new Date()
      })
    }

    localStorage.setItem("wishlist-default", JSON.stringify(wishlist))
    setIsInWishlist(!isInWishlist)
  }

  const addToCart = () => {
    if (!product) return

    const saved = localStorage.getItem("cart-default")
    let cart = saved ? JSON.parse(saved) : {}

    if (cart[product.id]) {
      cart[product.id].quantity += quantity
    } else {
      cart[product.id] = {
        product,
        quantity
      }
    }

    localStorage.setItem("cart-default", JSON.stringify(cart))
    alert("Added to cart!")
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR'
    }).format(price)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading product...</p>
        </div>
      </div>
    )
  }

  if (!product) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Product not found</h2>
          <p className="text-gray-600 mb-4">The product you're looking for doesn't exist.</p>
          <Button onClick={() => router.push("/")}>
            Go Home
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Breadcrumb */}
        <div className="flex items-center space-x-2 mb-8">
          <Button
            variant="ghost"
            onClick={() => router.back()}
            className="flex items-center space-x-2"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>Back</span>
          </Button>
          <span className="text-gray-400">/</span>
          <span className="text-gray-600">{product.category}</span>
          <span className="text-gray-400">/</span>
          <span className="text-gray-900 font-medium">{product.name}</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Product Image */}
          <div className="space-y-4">
            <div className="aspect-square bg-white rounded-lg overflow-hidden shadow-sm">
              <img
                src={product.image}
                alt={product.name}
                className="w-full h-full object-cover"
              />
            </div>
            
            {/* Action Buttons */}
            <div className="flex space-x-4">
              <Button
                variant={isInWishlist ? "default" : "outline"}
                onClick={toggleWishlist}
                className="flex-1"
              >
                <Heart className={`h-4 w-4 mr-2 ${isInWishlist ? "fill-current" : ""}`} />
                {isInWishlist ? "In Wishlist" : "Add to Wishlist"}
              </Button>
              <Button
                variant="outline"
                onClick={() => navigator.share?.({ title: product.name, url: window.location.href })}
              >
                <Share2 className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Product Info */}
          <div className="space-y-6">
            <div>
              <Badge variant="secondary" className="mb-2">
                {product.category}
              </Badge>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                {product.name}
              </h1>
              <p className="text-gray-600 mb-4">
                {product.description}
              </p>
              
              {/* Rating */}
              <div className="flex items-center space-x-2 mb-4">
                <div className="flex items-center">
                  {[...Array(5)].map((_, i) => (
                    <Star
                      key={i}
                      className={`h-4 w-4 ${
                        i < Math.floor(product.rating) ? "text-yellow-400 fill-current" : "text-gray-300"
                      }`}
                    />
                  ))}
                </div>
                <span className="text-sm text-gray-600">
                  {product.rating} ({product.reviewCount} reviews)
                </span>
              </div>
            </div>

            {/* Price */}
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <span className="text-3xl font-bold text-green-600">
                  {formatPrice(product.price)}
                </span>
                {product.originalPrice && (
                  <span className="text-lg text-gray-500 line-through">
                    {formatPrice(product.originalPrice)}
                  </span>
                )}
                {product.originalPrice && (
                  <Badge variant="destructive">
                    {Math.round(((product.originalPrice - product.price) / product.originalPrice) * 100)}% OFF
                  </Badge>
                )}
              </div>
              
              {product.aiReasoning && (
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-900 mb-2">Why This Gift?</h4>
                  <p className="text-blue-800 text-sm">{product.aiReasoning}</p>
                </div>
              )}
            </div>

            {/* Features */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Key Features</h3>
              <div className="grid grid-cols-2 gap-2">
                {product.features.map((feature, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-gray-600">{feature}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Add to Cart */}
            <div className="space-y-4">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <label className="text-sm font-medium">Quantity:</label>
                  <select
                    value={quantity}
                    onChange={(e) => setQuantity(Number(e.target.value))}
                    className="border border-gray-300 rounded px-2 py-1"
                  >
                    {[1, 2, 3, 4, 5].map(num => (
                      <option key={num} value={num}>{num}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  <Truck className="h-4 w-4 text-green-600" />
                  <span className="text-sm text-gray-600">Free shipping</span>
                </div>
              </div>
              
              <Button
                onClick={addToCart}
                disabled={!product.inStock}
                className="w-full"
                size="lg"
              >
                <ShoppingCart className="h-5 w-5 mr-2" />
                {product.inStock ? "Add to Cart" : "Out of Stock"}
              </Button>
            </div>

            {/* Additional Info */}
            <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-200">
              <div className="flex items-center space-x-2">
                <Shield className="h-4 w-4 text-gray-400" />
                <span className="text-sm text-gray-600">Secure Payment</span>
              </div>
              <div className="flex items-center space-x-2">
                <RotateCcw className="h-4 w-4 text-gray-400" />
                <span className="text-sm text-gray-600">Easy Returns</span>
              </div>
            </div>
          </div>
        </div>

        {/* Product Details Tabs */}
        <div className="mt-12">
          <Tabs defaultValue="description" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="description">Description</TabsTrigger>
              <TabsTrigger value="specifications">Specifications</TabsTrigger>
              <TabsTrigger value="reviews">Reviews</TabsTrigger>
            </TabsList>
            
            <TabsContent value="description" className="mt-6">
              <Card>
                <CardContent className="p-6">
                  <p className="text-gray-700 leading-relaxed">
                    {product.description}
                  </p>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="specifications" className="mt-6">
              <Card>
                <CardContent className="p-6">
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <span className="font-medium text-gray-900">Brand:</span>
                        <span className="ml-2 text-gray-600">{product.brand}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-900">Category:</span>
                        <span className="ml-2 text-gray-600">{product.category}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-900">Availability:</span>
                        <span className="ml-2 text-gray-600">
                          {product.inStock ? "In Stock" : "Out of Stock"}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-900">Shipping:</span>
                        <span className="ml-2 text-gray-600">
                          {product.fastShipping ? "Fast Delivery" : "Standard Delivery"}
                        </span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="reviews" className="mt-6">
              <Card>
                <CardContent className="p-6">
                  <div className="text-center text-gray-500">
                    <Star className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                    <p>Reviews coming soon!</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
} 