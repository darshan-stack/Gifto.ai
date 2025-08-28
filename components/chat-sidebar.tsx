"use client"

import { useState, useEffect } from "react"
import { MessageSquare, Clock, Trash2, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent } from "@/components/ui/card"

interface ChatHistory {
  id: string
  prompt: string
  timestamp: Date
  recipient_profile?: any
  occasion_info?: any
}

interface ChatSidebarProps {
  isOpen: boolean
  onClose: () => void
  onLoadConversation: (history: ChatHistory) => void
  currentConversation?: ChatHistory
}

export function ChatSidebar({ isOpen, onClose, onLoadConversation, currentConversation }: ChatSidebarProps) {
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([])
  const [searchTerm, setSearchTerm] = useState("")

  // Load chat history from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("gift-chat-history")
    if (saved) {
      try {
        const history = JSON.parse(saved).map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp)
        }))
        setChatHistory(history)
      } catch (error) {
        console.error("Error loading chat history:", error)
      }
    }
  }, [])

  // Save chat history to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem("gift-chat-history", JSON.stringify(chatHistory))
  }, [chatHistory])

  const addToHistory = (prompt: string, recipient_profile?: any, occasion_info?: any) => {
    const newHistory: ChatHistory = {
      id: Date.now().toString(),
      prompt,
      timestamp: new Date(),
      recipient_profile,
      occasion_info
    }
    setChatHistory(prev => [newHistory, ...prev.slice(0, 49)]) // Keep last 50 conversations
  }

  const deleteHistory = (id: string) => {
    setChatHistory(prev => prev.filter(item => item.id !== id))
  }

  const filteredHistory = chatHistory.filter(item =>
    item.prompt.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const formatTimestamp = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    
    if (days === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } else if (days === 1) {
      return "Yesterday"
    } else if (days < 7) {
      return `${days} days ago`
    } else {
      return date.toLocaleDateString()
    }
  }

  const truncatePrompt = (prompt: string, maxLength: number = 50) => {
    return prompt.length > maxLength ? prompt.substring(0, maxLength) + "..." : prompt
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-y-0 left-0 z-50 w-80 bg-white border-r border-gray-200 shadow-lg">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex items-center space-x-2">
            <MessageSquare className="h-5 w-5 text-blue-600" />
            <h2 className="text-lg font-semibold">Chat History</h2>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            ×
          </Button>
        </div>

        {/* Search */}
        <div className="p-4 border-b border-gray-200">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search conversations..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {/* Chat History List */}
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-2">
            {filteredHistory.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <MessageSquare className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                <p>No conversations yet</p>
                <p className="text-sm">Start chatting to see your history here</p>
              </div>
            ) : (
              filteredHistory.map((item) => (
                <Card
                  key={item.id}
                  className={`cursor-pointer transition-colors hover:bg-gray-50 ${
                    currentConversation?.id === item.id ? "bg-blue-50 border-blue-200" : ""
                  }`}
                  onClick={() => onLoadConversation(item)}
                >
                  <CardContent className="p-3">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {truncatePrompt(item.prompt)}
                        </p>
                        <div className="flex items-center space-x-2 mt-1">
                          <Clock className="h-3 w-3 text-gray-400" />
                          <span className="text-xs text-gray-500">
                            {formatTimestamp(item.timestamp)}
                          </span>
                        </div>
                        {item.occasion_info?.occasion && (
                          <span className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded mt-1">
                            {item.occasion_info.occasion}
                          </span>
                        )}
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          deleteHistory(item.id)
                        }}
                        className="text-gray-400 hover:text-red-500"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200">
          <Button
            variant="outline"
            className="w-full"
            onClick={() => {
              setChatHistory([])
              localStorage.removeItem("gift-chat-history")
            }}
          >
            Clear All History
          </Button>
        </div>
      </div>
    </div>
  )
} 