import React, { useEffect, useState } from 'react';
import { Navigate, useLocation, useOutletContext } from 'react-router-dom';

const ProtectedRoute = ({ children, requirePsych = false, requireVoice = false, requireQuiz = false }) => {
  const { user, progress } = useOutletContext() || {};
  const location = useLocation();

  if (!user?.email) {
    return <Navigate to="/dashboard" replace />;
  }

  if (!progress) {
    return null;
  }

  // Route‑level enforcement
  if (requirePsych && !progress.psych_completed && location.pathname !== '/dashboard/psychology') {
    return <Navigate to="/dashboard/psychology" replace />;
  }

  if (requireVoice && !progress.voice_completed && location.pathname !== '/dashboard/voice') {
    return <Navigate to="/dashboard/voice" replace />;
  }

  if (requireQuiz && !progress.quiz_completed && location.pathname !== '/dashboard/quiz') {
    return <Navigate to="/dashboard/quiz" replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;

