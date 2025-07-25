import type React from "react"
import type { Metadata } from "next"
import { Inter, Poppins, Playfair_Display } from "next/font/google"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
})

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
  variable: "--font-poppins",
  display: "swap",
})

const playfair = Playfair_Display({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-playfair",
  display: "swap",
})

export const metadata: Metadata = {
  title: "GiftGenius AI - Find Perfect Gifts with AI",
  description:
    "Discover the perfect gift for any occasion with our AI-powered recommendation engine. Get personalized gift suggestions from over 100,000 curated products in seconds.",
  keywords: "gifts, AI, recommendations, shopping, presents, gift ideas, personalized gifts",
  authors: [{ name: "GiftGenius AI Team" }],
  creator: "GiftGenius AI",
  publisher: "GiftGenius AI",
  robots: "index, follow",
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://giftgenius-ai.com",
    title: "GiftGenius AI - Find Perfect Gifts with AI",
    description: "AI-powered gift recommendations for any occasion",
    siteName: "GiftGenius AI",
  },
  twitter: {
    card: "summary_large_image",
    title: "GiftGenius AI - Find Perfect Gifts with AI",
    description: "AI-powered gift recommendations for any occasion",
    creator: "@giftgeniusai",
  },
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${poppins.variable} ${playfair.variable}`}>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#8B5CF6" />
      </head>
      <body className={`${inter.className} antialiased`}>{children}</body>
    </html>
  )
}
