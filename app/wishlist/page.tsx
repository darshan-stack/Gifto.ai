"use client"

import { useState, useEffect } from "react"
import { Heart, Trash2, Eye, ShoppingCart, ArrowLeft, Filter, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useRouter } from "next/navigation"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface WishlistItem {
  id: string
  name: string
  price: number
  image: string
  category: string
  addedAt: Date
}

export default function WishlistPage() {
  const [wishlistItems, setWishlistItems] = useState<WishlistItem[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [sortBy, setSortBy] = useState("addedAt")
  const [filterCategory, setFilterCategory] = useState("all")
  const router = useRouter()

  // Load wishlist from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("wishlist-default")
      if (saved) {
      try {
        const items = JSON.parse(saved).map((item: any) => ({
          ...item,
          addedAt: new Date(item.addedAt)
        }))
        setWishlistItems(items)
    } catch (error) {
      console.error("Error loading wishlist:", error)
    }
  }
  }, [])

  const removeFromWishlist = (itemId: string) => {
    const updatedItems = wishlistItems.filter(item => item.id !== itemId)
    setWishlistItems(updatedItems)
    localStorage.setItem("wishlist-default", JSON.stringify(updatedItems))
  }

  const clearWishlist = () => {
    setWishlistItems([])
    localStorage.removeItem("wishlist-default")
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR'
    }).format(price)
  }

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  // Filter and sort items
  const filteredAndSortedItems = wishlistItems
    .filter(item => {
      const matchesSearch = item.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           item.category.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesCategory = filterCategory === "all" || item.category === filterCategory
      return matchesSearch && matchesCategory
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "name":
          return a.name.localeCompare(b.name)
        case "price":
          return a.price - b.price
        case "category":
          return a.category.localeCompare(b.category)
        case "addedAt":
        default:
          return b.addedAt.getTime() - a.addedAt.getTime()
      }
    })

  const categories = Array.from(new Set(wishlistItems.map(item => item.category)))

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            onClick={() => router.back()}
              className="flex items-center space-x-2"
          >
            <ArrowLeft className="h-4 w-4" />
              <span>Back</span>
          </Button>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">My Wishlist</h1>
              <p className="text-gray-600">
                {wishlistItems.length} item{wishlistItems.length !== 1 ? 's' : ''} saved
              </p>
          </div>
        </div>
        
        {wishlistItems.length > 0 && (
          <Button
            variant="outline"
            onClick={clearWishlist}
            className="text-red-600 hover:text-red-700"
          >
              <Trash2 className="h-4 w-4 mr-2" />
            Clear All
          </Button>
        )}
      </div>

        {/* Filters and Search */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search items..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            
            <Select value={filterCategory} onValueChange={setFilterCategory}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                {categories.map(category => (
                  <SelectItem key={category} value={category}>
                    {category}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger>
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="addedAt">Recently Added</SelectItem>
                <SelectItem value="name">Name</SelectItem>
                <SelectItem value="price">Price</SelectItem>
                <SelectItem value="category">Category</SelectItem>
              </SelectContent>
            </Select>
            
            <div className="text-sm text-gray-500 flex items-center justify-end">
              {filteredAndSortedItems.length} of {wishlistItems.length} items
            </div>
          </div>
      </div>

      {/* Wishlist Items */}
      {wishlistItems.length === 0 ? (
          <Card className="text-center py-16">
            <CardContent>
          <Heart className="h-16 w-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Your wishlist is empty</h3>
              <p className="text-gray-600 mb-6">
                Start adding items to your wishlist to see them here
              </p>
          <Button onClick={() => router.push("/")}>
            Start Shopping
          </Button>
            </CardContent>
          </Card>
        ) : filteredAndSortedItems.length === 0 ? (
          <Card className="text-center py-16">
            <CardContent>
              <Search className="h-16 w-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">No items found</h3>
              <p className="text-gray-600">
                Try adjusting your search or filters
              </p>
            </CardContent>
          </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredAndSortedItems.map((item) => (
              <Card key={item.id} className="group hover:shadow-lg transition-shadow">
              <CardHeader className="p-4 pb-2">
                  <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden mb-4">
                    {item.image ? (
                  <img
                        src={item.image}
                        alt={item.name}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                      />
                    ) : (
                      <div className="w-full h-full bg-gray-200 flex items-center justify-center">
                        <Heart className="h-8 w-8 text-gray-400" />
                      </div>
                    )}
                  </div>
                  
                  <div className="flex items-start justify-between">
                    <Badge variant="secondary" className="text-xs">
                      {item.category}
                    </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                      onClick={() => removeFromWishlist(item.id)}
                      className="text-red-500 hover:text-red-700 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              
                <CardContent className="p-4 pt-0">
                  <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2">
                    {item.name}
                  </h3>
                
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-lg font-bold text-green-600">
                      {formatPrice(item.price)}
                    </span>
                    <span className="text-xs text-gray-500">
                      Added {formatDate(item.addedAt)}
                      </span>
                </div>
                
                  <div className="flex space-x-2">
                  <Button
                      variant="outline"
                    size="sm"
                    className="flex-1"
                      onClick={() => router.push(`/product/${item.id}`)}
                  >
                      <Eye className="h-4 w-4 mr-2" />
                      View
                  </Button>
                  <Button
                    size="sm"
                      className="flex-1"
                  >
                      <ShoppingCart className="h-4 w-4 mr-2" />
                      Buy
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
      </div>
    </div>
  )
} 