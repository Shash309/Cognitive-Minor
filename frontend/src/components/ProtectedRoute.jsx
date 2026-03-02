import React, { useEffect, useState } from 'react';
import { Navigate, useLocation, useOutletContext } from 'react-router-dom';

const ProtectedRoute = ({ children, requirePsych = false, requireVoice = false, requireQuiz = false }) => {
  const { user } = useOutletContext() || {};
  const location = useLocation();
  const [progress, setProgress] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      if (!user?.email) {
        setLoading(false);
        setError('Please sign in to continue.');
        return;
      }
      try {
        const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
        const res = await fetch(
          `${apiBase}/api/user-progress?user_email=${encodeURIComponent(user.email)}`
        );
        const json = await res.json();
        if (!res.ok || json.error) {
          throw new Error(json.error || 'Unable to read onboarding progress.');
        }
        setProgress(json);
        setError('');
      } catch (err) {
        setError(err.message || 'Unable to read onboarding progress.');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [user?.email]);

  if (loading) {
    return null;
  }

  if (error || !progress) {
    return <Navigate to="/dashboard" replace />;
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

