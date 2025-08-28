"use client"

import { useState, useEffect } from "react"
import { Heart, ShoppingBag, Eye, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import { Card, CardContent } from "@/components/ui/card"
import { useRouter } from "next/navigation"

interface WishlistItem {
  id: string
  name: string
  price: number
  image: string
  category: string
  addedAt: Date
}

interface WishlistIconProps {
  userId?: string
}

export function WishlistIcon({ userId = "default" }: WishlistIconProps) {
  const [wishlistItems, setWishlistItems] = useState<WishlistItem[]>([])
  const [isOpen, setIsOpen] = useState(false)
  const router = useRouter()

  // Load wishlist from localStorage
  useEffect(() => {
    const saved = localStorage.getItem(`wishlist-${userId}`)
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
  }, [userId])

  // Save wishlist to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem(`wishlist-${userId}`, JSON.stringify(wishlistItems))
  }, [wishlistItems, userId])

  const addToWishlist = (item: Omit<WishlistItem, 'addedAt'>) => {
    const newItem: WishlistItem = {
      ...item,
      addedAt: new Date()
    }
    setWishlistItems(prev => {
      const exists = prev.find(existing => existing.id === item.id)
      if (exists) return prev
      return [...prev, newItem]
    })
  }

  const removeFromWishlist = (itemId: string) => {
    setWishlistItems(prev => prev.filter(item => item.id !== itemId))
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR'
    }).format(price)
  }

  const formatDate = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    
    if (days === 0) return "Today"
    if (days === 1) return "Yesterday"
    if (days < 7) return `${days} days ago`
    return date.toLocaleDateString()
  }

  const truncateName = (name: string, maxLength: number = 30) => {
    return name.length > maxLength ? name.substring(0, maxLength) + "..." : name
  }

  return (
    <div className="relative">
      <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" className="relative">
            <Heart className="h-5 w-5" />
            {wishlistItems.length > 0 && (
              <Badge 
                variant="destructive" 
                className="absolute -top-2 -right-2 h-5 w-5 rounded-full p-0 flex items-center justify-center text-xs"
              >
                {wishlistItems.length > 99 ? "99+" : wishlistItems.length}
              </Badge>
            )}
          </Button>
        </DropdownMenuTrigger>
        
        <DropdownMenuContent align="end" className="w-80 p-0">
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Wishlist</h3>
              <Badge variant="secondary">{wishlistItems.length} items</Badge>
            </div>
          </div>
          
          <div className="max-h-96 overflow-y-auto">
            {wishlistItems.length === 0 ? (
              <div className="p-8 text-center text-gray-500">
                <Heart className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                <p>Your wishlist is empty</p>
                <p className="text-sm">Start adding items to see them here</p>
              </div>
            ) : (
              <div className="p-2 space-y-2">
                {wishlistItems.slice(0, 5).map((item) => (
                  <Card key={item.id} className="p-3">
                    <CardContent className="p-0">
                      <div className="flex items-center space-x-3">
                        <div className="w-12 h-12 bg-gray-100 rounded-lg flex-shrink-0">
                          {item.image ? (
                            <img 
                              src={item.image} 
                              alt={item.name}
                              className="w-full h-full object-cover rounded-lg"
                            />
                          ) : (
                            <div className="w-full h-full bg-gray-200 rounded-lg flex items-center justify-center">
                              <ShoppingBag className="h-5 w-5 text-gray-400" />
                            </div>
                          )}
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {truncateName(item.name)}
                          </p>
                          <p className="text-xs text-gray-500">{item.category}</p>
                          <p className="text-sm font-semibold text-green-600">
                            {formatPrice(item.price)}
                          </p>
                          <p className="text-xs text-gray-400">
                            Added {formatDate(item.addedAt)}
                          </p>
                        </div>
                        
                        <div className="flex flex-col space-y-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              router.push(`/product/${item.id}`)
                              setIsOpen(false)
                            }}
                            className="h-8 w-8 p-0"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeFromWishlist(item.id)}
                            className="h-8 w-8 p-0 text-red-500 hover:text-red-700"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
          
          {wishlistItems.length > 0 && (
            <>
              <DropdownMenuSeparator />
              <div className="p-2">
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => {
                    router.push("/wishlist")
                    setIsOpen(false)
                  }}
                >
                  View All Wishlist Items
                </Button>
              </div>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
} 