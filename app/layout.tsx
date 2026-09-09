import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'RoadScan AI — Road Inspection',
  description: 'AI-powered road defect detection and inspection system.',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
