"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { 
  MessageSquare, 
  Heart, 
  ShoppingCart, 
  Gift, 
  History, 
  Star,
  X,
  ChevronRight,
  Search,
  Filter
} from "lucide-react"

interface ChatHistory {
  id: string
  prompt: string
  timestamp: string
  recommendations: string[]
  recipient_profile?: any
  occasion_info?: any
}

interface WishlistItem {
  id: string
  name: string
  price: number
  image: string
  category: string
  added_at: string
}

export default function HistorySidebar({ 
  isOpen, 
  onClose, 
  onNewChat,
  onViewWishlist,
  onViewCart,
  onViewCategories
}: {
  isOpen: boolean
  onClose: () => void
  onNewChat: () => void
  onViewWishlist: () => void
  onViewCart: () => void
  onViewCategories: () => void
}) {
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([])
  const [wishlist, setWishlist] = useState<WishlistItem[]>([])
  const [activeTab, setActiveTab] = useState<'history' | 'wishlist' | 'cart' | 'categories'>('history')

  // Load chat history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem('gift-chat-history')
    if (savedHistory) {
      setChatHistory(JSON.parse(savedHistory))
    }
    
    const savedWishlist = localStorage.getItem('gift-wishlist')
    if (savedWishlist) {
      setWishlist(JSON.parse(savedWishlist))
    }
  }, [])

  const handleChatClick = (chat: ChatHistory) => {
    // Replay the chat - you can implement this to show previous recommendations
    console.log('Replaying chat:', chat)
  }

  const removeFromWishlist = (itemId: string) => {
    const updatedWishlist = wishlist.filter(item => item.id !== itemId)
    setWishlist(updatedWishlist)
    localStorage.setItem('gift-wishlist', JSON.stringify(updatedWishlist))
  }

  const categories = [
    { name: "Electronics", icon: "📱", count: 150 },
    { name: "Fashion", icon: "👕", count: 89 },
    { name: "Home & Garden", icon: "🏠", count: 234 },
    { name: "Books", icon: "📚", count: 67 },
    { name: "Toys & Games", icon: "🎮", count: 123 },
    { name: "Sports", icon: "⚽", count: 78 },
    { name: "Beauty", icon: "💄", count: 95 },
    { name: "Food & Drinks", icon: "🍫", count: 45 }
  ]

  return (
    <div className={`fixed inset-y-0 right-0 w-80 bg-white border-l border-gray-200 transform transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : 'translate-x-full'} z-50`}>
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold">Gift Assistant</h2>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setActiveTab('history')}
            className={`flex-1 py-3 px-4 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'history' 
                ? 'border-blue-500 text-blue-600' 
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <History className="h-4 w-4 inline mr-2" />
            History
          </button>
          <button
            onClick={() => setActiveTab('wishlist')}
            className={`flex-1 py-3 px-4 text-sm font-medium border-b-2 transition-colors relative ${
              activeTab === 'wishlist' 
                ? 'border-blue-500 text-blue-600' 
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <Heart className="h-4 w-4 inline mr-2" />
            Wishlist
            {wishlist.length > 0 && (
              <Badge className="ml-1 h-5 w-5 p-0 text-xs bg-red-500">
                {wishlist.length}
              </Badge>
            )}
          </button>
          <button
            onClick={() => setActiveTab('cart')}
            className={`flex-1 py-3 px-4 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'cart' 
                ? 'border-blue-500 text-blue-600' 
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <ShoppingCart className="h-4 w-4 inline mr-2" />
            Cart
          </button>
          <button
            onClick={() => setActiveTab('categories')}
            className={`flex-1 py-3 px-4 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'categories' 
                ? 'border-blue-500 text-blue-600' 
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <Filter className="h-4 w-4 inline mr-2" />
            Browse
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === 'history' && (
            <div className="space-y-4">
              <Button 
                onClick={onNewChat} 
                className="w-full bg-blue-600 hover:bg-blue-700 text-white"
              >
                <MessageSquare className="h-4 w-4 mr-2" />
                New Chat
              </Button>
              
              {chatHistory.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  <MessageSquare className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                  <p>No chat history yet</p>
                  <p className="text-sm">Start a conversation to see your history here</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {chatHistory.slice(0, 10).map((chat) => (
                    <Card 
                      key={chat.id} 
                      className="p-3 cursor-pointer hover:bg-gray-50 transition-colors"
                      onClick={() => handleChatClick(chat)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {chat.prompt}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            {new Date(chat.timestamp).toLocaleDateString()}
                          </p>
                          {chat.occasion_info && (
                            <Badge variant="secondary" className="mt-1 text-xs">
                              {chat.occasion_info.occasion}
                            </Badge>
                          )}
                        </div>
                        <ChevronRight className="h-4 w-4 text-gray-400 flex-shrink-0" />
                      </div>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'wishlist' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">Saved Items ({wishlist.length})</h3>
                <Button variant="outline" size="sm" onClick={onViewWishlist}>
                  View All
                </Button>
              </div>
              
              {wishlist.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  <Heart className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                  <p>Your wishlist is empty</p>
                  <p className="text-sm">Save items you like to your wishlist</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {wishlist.slice(0, 5).map((item) => (
                    <Card key={item.id} className="p-3">
                      <div className="flex items-center space-x-3">
                        <img 
                          src={item.image} 
                          alt={item.name}
                          className="w-12 h-12 object-cover rounded"
                        />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {item.name}
                          </p>
                          <p className="text-sm text-gray-500">₹{item.price}</p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFromWishlist(item.id)}
                          className="text-red-500 hover:text-red-700"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'cart' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">Shopping Cart</h3>
                <Button variant="outline" size="sm" onClick={onViewCart}>
                  View Cart
                </Button>
              </div>
              
              <div className="text-center text-gray-500 py-8">
                <ShoppingCart className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>Your cart is empty</p>
                <p className="text-sm">Add items to your cart to see them here</p>
              </div>
            </div>
          )}

          {activeTab === 'categories' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">Browse Categories</h3>
                <Button variant="outline" size="sm" onClick={onViewCategories}>
                  View All
                </Button>
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                {categories.map((category) => (
                  <Card 
                    key={category.name}
                    className="p-3 cursor-pointer hover:bg-gray-50 transition-colors"
                    onClick={() => onViewCategories()}
                  >
                    <div className="text-center">
                      <div className="text-2xl mb-2">{category.icon}</div>
                      <p className="text-sm font-medium text-gray-900">
                        {category.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {category.count} items
                      </p>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 