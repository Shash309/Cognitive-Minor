import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import Dashboard from './components/Dashboard';
import Home from './components/Home';
import CollegeExplorer from './components/CollegeExplorer';
import CareerQuiz from './components/CareerQuiz';
import SkillBuilder from './components/SkillBuilder';
import CareerPathVisualizer from './components/CareerPathVisualizer';
import TimelineTracker from './components/TimelineTracker';
import './App.css';

function App() {
  // Check localStorage for persisted user session
  const [user, setUser] = useState(() => {
    const savedUser = localStorage.getItem('career_app_user');
    return savedUser ? JSON.parse(savedUser) : null;
  });

  // Initialize isLoggedIn based on whether user data exists
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    return !!localStorage.getItem('career_app_user');
  });

  const handleLogin = (userData) => {
    // Save user data to localStorage
    localStorage.setItem('career_app_user', JSON.stringify(userData));
    setUser(userData);
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    // Clear session from localStorage
    localStorage.removeItem('career_app_user');
    setIsLoggedIn(false);
    setUser(null);
  };

  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={!isLoggedIn ? <LandingPage onLogin={handleLogin} /> : <Navigate to="/dashboard" replace />} />
          <Route path="/login" element={!isLoggedIn ? <LandingPage onLogin={handleLogin} /> : <Navigate to="/dashboard" replace />} />

          <Route path="/dashboard" element={isLoggedIn ? <Dashboard user={user} onLogout={handleLogout} /> : <Navigate to="/" replace />}>
            <Route index element={<Home user={user} />} />
            <Route path="colleges" element={<CollegeExplorer />} />
            <Route path="quiz" element={<CareerQuiz />} />
            <Route path="skills" element={<SkillBuilder />} />
            <Route path="visualizer" element={<CareerPathVisualizer />} />
            <Route path="timeline" element={<TimelineTracker />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
