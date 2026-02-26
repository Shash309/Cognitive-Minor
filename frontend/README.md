# 🌐 CAPE Frontend — Interface & Visualization

This is the primary user interface for **CAPE (Career Path Explorer)**. Built with React and Vite, it focuses on providing a high-performance, interactive, and visually stunning experience for exploring career data.

## ✨ Key Features
- **Dynamic Career Visualizer**: Powered by `reactflow` and `elkjs` for complex graph layouts.
- **Interactive Roadmap**: Real-time path tracing with animated dotted connections.
- **Glassmorphism Design**: Modern UI aesthetic using consistent design tokens.
- **Smooth Animations**: Integrated `framer-motion` for transitions and micro-interactions.
- **Responsive Layout**: Works across different screen sizes.

## 🛠️ Technology Stack
- **Framework**: React 18+
- **Build Tool**: Vite
- **Graph Engine**: React Flow
- **Layout Engine**: ELK.js
- **Animations**: Framer Motion
- **Icons**: FontAwesome 6

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Setup environment**:
   Create a `.env` file if you need to point to a custom backend URL (default is `http://127.0.0.1:5000`):
   ```
   VITE_API_URL=http://localhost:5000
   ```

3. **Run in development**:
   ```bash
   npm run dev
   ```

4. **Build for production**:
   ```bash
   npm run build
   ```
