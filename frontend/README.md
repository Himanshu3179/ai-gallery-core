# 🎨 Prospo AI Gallery - Frontend

> React + TypeScript frontend for the AI-powered photo gallery with semantic search and facial recognition.

## 🛠 Tech Stack

- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite (with HMR and Fast Refresh)
- **Styling:** CSS with custom masonry grid layout
- **UI Components:** Custom Pin-based gallery interface

## 🚀 Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 📁 Project Structure

- `src/components/` - Reusable UI components (Layout, MasonryGrid, Pin, SearchBar)
- `src/pages/` - Page components (Home, ImageDetail)
- `src/types.ts` - TypeScript type definitions

## 🔌 API Integration

The frontend connects to the FastAPI backend running on `http://localhost:8000` for:
- Semantic search queries
- Image metadata retrieval
- Face recognition results

## 🎨 Features

- **Masonry Grid Layout:** Pinterest-style responsive image grid
- **Semantic Search:** Natural language queries like "cat on sofa" or "beach sunset"
- **Image Details:** Click any image to view full details and AI-generated captions
- **Responsive Design:** Optimized for desktop and mobile viewing
