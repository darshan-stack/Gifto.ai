import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import type { Product } from "./types"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPrice(price: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(price)
}

export function generateMockProducts(query: string): Product[] {
  const productTemplates = [
    {
      categories: ["Electronics"],
      items: [
        {
          name: "Wireless Bluetooth Headphones",
          desc: "Premium noise-canceling headphones with 30-hour battery life",
          image: "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&h=400&fit=crop",
        },
        {
          name: "Smart Fitness Watch",
          desc: "Advanced fitness tracking with heart rate monitor and GPS",
          image: "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&h=400&fit=crop",
        },
        {
          name: "Portable Bluetooth Speaker",
          desc: "Waterproof speaker with 360-degree sound and LED lights",
          image: "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=400&fit=crop",
        },
        {
          name: "Wireless Charging Pad",
          desc: "Fast wireless charging for all Qi-enabled devices",
          image: "https://images.unsplash.com/photo-1586953208448-b95a79798f07?w=400&h=400&fit=crop",
        },
        {
          name: "Smart Home Assistant",
          desc: "Voice-controlled smart speaker with AI assistant",
          image: "https://images.unsplash.com/photo-1543512214-318c7553f230?w=400&h=400&fit=crop",
        },
        {
          name: "Gaming Mechanical Keyboard",
          desc: "RGB backlit mechanical keyboard for gaming enthusiasts",
          image: "https://images.unsplash.com/photo-1541140532154-b024d705b90a?w=400&h=400&fit=crop",
        },
        {
          name: "4K Action Camera",
          desc: "Ultra HD action camera with image stabilization",
          image: "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?w=400&h=400&fit=crop",
        },
        {
          name: "Tablet with Stylus",
          desc: "High-resolution tablet perfect for digital art and productivity",
          image: "https://images.unsplash.com/photo-1544244015-0df4b3ffc6b0?w=400&h=400&fit=crop",
        },
      ],
    },
    {
      categories: ["Home & Garden"],
      items: [
        {
          name: "Essential Oil Diffuser",
          desc: "Ultrasonic aromatherapy diffuser with LED mood lighting",
          image: "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=400&h=400&fit=crop",
        },
        {
          name: "Smart Plant Pot",
          desc: "Self-watering planter with app connectivity",
          image: "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=400&h=400&fit=crop",
        },
        {
          name: "Cozy Throw Blanket",
          desc: "Ultra-soft weighted blanket for relaxation and comfort",
          image: "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400&h=400&fit=crop",
        },
        {
          name: "Decorative Wall Art",
          desc: "Modern abstract canvas print for home decoration",
          image: "https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=400&h=400&fit=crop",
        },
        {
          name: "Ceramic Coffee Mug Set",
          desc: "Handcrafted ceramic mugs with unique glazed finish",
          image: "https://images.unsplash.com/photo-1514228742587-6b1558fcf93a?w=400&h=400&fit=crop",
        },
        {
          name: "LED String Lights",
          desc: "Warm white fairy lights perfect for ambiance",
          image: "https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=400&h=400&fit=crop",
        },
        {
          name: "Succulent Garden Kit",
          desc: "Complete kit with succulents, pots, and care instructions",
          image: "https://images.unsplash.com/photo-1459411621453-7b03977f4bfc?w=400&h=400&fit=crop",
        },
        {
          name: "Bamboo Kitchen Utensil Set",
          desc: "Eco-friendly bamboo cooking utensils with holder",
          image: "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=400&fit=crop",
        },
      ],
    },
    {
      categories: ["Fashion"],
      items: [
        {
          name: "Luxury Silk Scarf",
          desc: "Premium silk scarf with elegant pattern design",
          image: "https://images.unsplash.com/photo-1601924994987-69e26d50dc26?w=400&h=400&fit=crop",
        },
        {
          name: "Leather Crossbody Bag",
          desc: "Genuine leather bag with adjustable strap",
          image: "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=400&fit=crop",
        },
        {
          name: "Designer Sunglasses",
          desc: "UV protection sunglasses with polarized lenses",
          image: "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=400&h=400&fit=crop",
        },
        {
          name: "Cashmere Sweater",
          desc: "Soft cashmere pullover in classic colors",
          image: "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=400&fit=crop",
        },
        {
          name: "Minimalist Watch",
          desc: "Elegant timepiece with leather strap",
          image: "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400&h=400&fit=crop",
        },
        {
          name: "Cozy Knit Beanie",
          desc: "Warm winter hat in soft merino wool",
          image: "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400&h=400&fit=crop",
        },
        {
          name: "Athletic Running Shoes",
          desc: "Lightweight running shoes with cushioned sole",
          image: "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=400&fit=crop",
        },
        {
          name: "Silk Pajama Set",
          desc: "Luxurious silk sleepwear for ultimate comfort",
          image: "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=400&h=400&fit=crop",
        },
      ],
    },
    {
      categories: ["Books"],
      items: [
        {
          name: "Bestselling Novel Collection",
          desc: "Set of award-winning contemporary fiction books",
          image: "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=400&fit=crop",
        },
        {
          name: "Coffee Table Art Book",
          desc: "Beautiful photography book perfect for display",
          image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",
        },
        {
          name: "Personal Development Guide",
          desc: "Inspiring self-help book for personal growth",
          image: "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=400&h=400&fit=crop",
        },
        {
          name: "Cookbook Collection",
          desc: "Gourmet recipes from world-renowned chefs",
          image: "https://images.unsplash.com/photo-1589829085413-56de8ae18c73?w=400&h=400&fit=crop",
        },
        {
          name: "Travel Photography Book",
          desc: "Stunning landscapes from around the world",
          image: "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=400&fit=crop",
        },
        {
          name: "Mindfulness Journal",
          desc: "Guided journal for meditation and reflection",
          image: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
        },
        {
          name: "Classic Literature Set",
          desc: "Timeless classics in beautiful hardcover editions",
          image: "https://images.unsplash.com/photo-1495446815901-a7297e633e8d?w=400&h=400&fit=crop",
        },
        {
          name: "Science Fiction Anthology",
          desc: "Collection of award-winning sci-fi short stories",
          image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",
        },
      ],
    },
    {
      categories: ["Toys"],
      items: [
        {
          name: "Educational Building Blocks",
          desc: "STEM learning toy for creative construction",
          image: "https://images.unsplash.com/photo-1558877385-09c4d8b7b7a9?w=400&h=400&fit=crop",
        },
        {
          name: "Art Supply Kit",
          desc: "Complete set with paints, brushes, and canvas",
          image: "https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=400&h=400&fit=crop",
        },
        {
          name: "Remote Control Drone",
          desc: "Beginner-friendly drone with HD camera",
          image: "https://images.unsplash.com/photo-1473968512647-3e447244af8f?w=400&h=400&fit=crop",
        },
        {
          name: "Puzzle Game Set",
          desc: "Challenging jigsaw puzzles for all ages",
          image: "https://images.unsplash.com/photo-1606092195730-5d7b9af1efc5?w=400&h=400&fit=crop",
        },
        {
          name: "Musical Instrument Toy",
          desc: "Child-friendly keyboard with learning modes",
          image: "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=400&h=400&fit=crop",
        },
        {
          name: "Science Experiment Kit",
          desc: "Safe and fun chemistry set for young scientists",
          image: "https://images.unsplash.com/photo-1532094349884-543bc11b234d?w=400&h=400&fit=crop",
        },
        {
          name: "Plush Stuffed Animal",
          desc: "Soft and cuddly teddy bear made from organic cotton",
          image: "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400&h=400&fit=crop",
        },
        {
          name: "Board Game Collection",
          desc: "Family-friendly games for hours of entertainment",
          image: "https://images.unsplash.com/photo-1606092195730-5d7b9af1efc5?w=400&h=400&fit=crop",
        },
      ],
    },
    {
      categories: ["Sports"],
      items: [
        {
          name: "Yoga Mat Set",
          desc: "Non-slip yoga mat with carrying strap and blocks",
          image: "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=400&h=400&fit=crop",
        },
        {
          name: "Resistance Band Kit",
          desc: "Complete workout bands with door anchor",
          image: "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=400&fit=crop",
        },
        {
          name: "Water Bottle with Infuser",
          desc: "Insulated bottle with fruit infusion chamber",
          image: "https://images.unsplash.com/photo-1523362628745-0c100150b504?w=400&h=400&fit=crop",
        },
        {
          name: "Foam Roller",
          desc: "Muscle recovery tool for post-workout relief",
          image: "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=400&fit=crop",
        },
        {
          name: "Adjustable Dumbbells",
          desc: "Space-saving weights with quick adjustment",
          image: "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=400&fit=crop",
        },
        {
          name: "Running Belt",
          desc: "Lightweight belt for phone and essentials",
          image: "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=400&fit=crop",
        },
        {
          name: "Tennis Racket",
          desc: "Professional-grade racket for all skill levels",
          image: "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400&h=400&fit=crop",
        },
        {
          name: "Cycling Helmet",
          desc: "Lightweight safety helmet with ventilation",
          image: "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=400&fit=crop",
        },
      ],
    },
    {
      categories: ["Beauty"],
      items: [
        {
          name: "Skincare Gift Set",
          desc: "Complete routine with cleanser, serum, and moisturizer",
          image: "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
        },
        {
          name: "Makeup Brush Collection",
          desc: "Professional brushes for flawless application",
          image: "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400&h=400&fit=crop",
        },
        {
          name: "Aromatherapy Bath Bombs",
          desc: "Relaxing bath bombs with essential oils",
          image: "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=400&h=400&fit=crop",
        },
        {
          name: "Hair Styling Tool",
          desc: "Professional hair dryer with multiple attachments",
          image: "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400&h=400&fit=crop",
        },
        {
          name: "Nail Care Kit",
          desc: "Complete manicure set with tools and polish",
          image: "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
        },
        {
          name: "Face Mask Collection",
          desc: "Variety pack of hydrating and purifying masks",
          image: "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
        },
        {
          name: "Perfume Gift Set",
          desc: "Elegant fragrance collection in travel sizes",
          image: "https://images.unsplash.com/photo-1541643600914-78b084683601?w=400&h=400&fit=crop",
        },
        {
          name: "LED Makeup Mirror",
          desc: "Illuminated mirror with adjustable brightness",
          image: "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400&h=400&fit=crop",
        },
      ],
    },
  ]

  const brands = ["Apple", "Samsung", "Nike", "Adidas", "Amazon", "Sony", "Microsoft", "Canon", "Fitbit", "Bose"]
  const allProducts: Product[] = []

  // Generate products from templates
  productTemplates.forEach((template) => {
    template.items.forEach((item, index) => {
      const basePrice = Math.floor(Math.random() * 400) + 30
      const hasDiscount = Math.random() > 0.7

      allProducts.push({
        id: `${template.categories[0].toLowerCase()}-${index + 1}`,
        name: item.name,
        description: item.desc,
        price: basePrice,
        originalPrice: hasDiscount ? Math.floor(basePrice * 1.3) : undefined,
        image: item.image,
        rating: Math.floor(Math.random() * 2) + 4, // 4-5 stars
        reviewCount: Math.floor(Math.random() * 2000) + 50,
        category: template.categories[0],
        brand: brands[Math.floor(Math.random() * brands.length)],
        features: [
          "Premium Quality",
          "Fast Shipping",
          "Gift Wrapping Available",
          "Customer Favorite",
          "Eco-Friendly",
          "Limited Edition",
        ].slice(0, Math.floor(Math.random() * 3) + 2),
        inStock: Math.random() > 0.05,
        fastShipping: Math.random() > 0.3,
      })
    })
  })

  // Shuffle and return 100 products
  const shuffled = allProducts.sort(() => Math.random() - 0.5)
  return shuffled.slice(0, 100)
}
