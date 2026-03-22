import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import Dashboard from './components/Dashboard';
import Home from './components/Home';
import CollegeExplorer from './components/CollegeExplorer';
import CareerQuiz from './components/CareerQuiz';
import PsychAssessment from './components/PsychAssessment';
import SkillBuilder from './components/SkillBuilder';
import CareerPathVisualizer from './components/CareerPathVisualizer';
import TimelineTracker from './components/TimelineTracker';
import Profile from './components/Profile';
import QuizResultDetails from './components/QuizResultDetails';
import Results from './components/Results';
import ProtectedRoute from './components/ProtectedRoute';
import VoiceInsight from './components/VoiceInsight';
import CounselorDashboard from './components/CounselorDashboard';
import CounselorHome from './components/CounselorHome';
import StudentReport from './components/StudentReport';
import CounselorChat from './components/CounselorChat';
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

  const isCounselorEnabled = import.meta.env.VITE_ENABLE_COUNSELLOR_FEATURE === "true";
  const isCounselor = user?.role === 'counselor' && isCounselorEnabled;
  const defaultRoute = isCounselor ? '/counselor' : '/dashboard/psychology';

  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={!isLoggedIn ? <LandingPage onLogin={handleLogin} /> : <Navigate to={defaultRoute} replace />} />
          <Route path="/login" element={!isLoggedIn ? <LandingPage onLogin={handleLogin} /> : <Navigate to={defaultRoute} replace />} />

          {/* ─── Student Routes ─── */}
          <Route path="/dashboard" element={isLoggedIn && !isCounselor ? <Dashboard user={user} onLogout={handleLogout} /> : <Navigate to="/" replace />}>
            <Route index element={<Home user={user} />} />
            <Route path="colleges" element={<CollegeExplorer />} />
            <Route
              path="quiz"
              element={
                <ProtectedRoute requirePsych requireVoice>
                  <CareerQuiz />
                </ProtectedRoute>
              }
            />
            <Route path="quiz-result/:attemptId" element={<QuizResultDetails />} />
            <Route path="psychology" element={<PsychAssessment />} />
            <Route
              path="voice"
              element={
                <ProtectedRoute requirePsych>
                  <VoiceInsight />
                </ProtectedRoute>
              }
            />
            <Route
              path="results"
              element={
                <ProtectedRoute requirePsych requireVoice requireQuiz>
                  <Results />
                </ProtectedRoute>
              }
            />
            <Route path="profile" element={<Profile />} />
            <Route path="skills" element={<SkillBuilder />} />
            <Route path="visualizer" element={<CareerPathVisualizer />} />
            <Route path="timeline" element={<TimelineTracker />} />
          </Route>

          {/* ─── Counselor Routes ─── */}
          {isCounselorEnabled && (
            <Route path="/counselor" element={isLoggedIn && isCounselor ? <CounselorDashboard user={user} onLogout={handleLogout} /> : <Navigate to="/" replace />}>
              <Route index element={<CounselorHome />} />
              <Route path="student/:studentEmail" element={<StudentReport />} />
              <Route path="chat/:studentEmail" element={<CounselorChat />} />
            </Route>
          )}
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
