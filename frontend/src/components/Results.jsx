import React, { useEffect, useState } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import './CounselorDashboard.css';

const Results = () => {
  const { user } = useOutletContext() || {};
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);
  const [processingIndex, setProcessingIndex] = useState(0);
  const [counselorRequested, setCounselorRequested] = useState(false);
  const [requestingCounselor, setRequestingCounselor] = useState(false);

  const processingSteps = [
    'Analyzing psychological traits',
    'Analyzing voice patterns',
    'Evaluating academic preferences',
    'Generating career recommendations',
  ];

  useEffect(() => {
    const load = async () => {
      if (!user?.email) {
        setError('Please sign in to view your results.');
        setLoading(false);
        return;
      }
      setLoading(true);
      setError('');
      try {
        const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
        const res = await fetch(
          `${apiBase}/api/career-results?user_email=${encodeURIComponent(user.email)}`
        );
        const json = await res.json();
        if (!res.ok || json.error) {
          throw new Error(json.error || 'Unable to load results.');
        }
        setData(json);
      } catch (err) {
        setError(err.message || 'Something went wrong while loading results.');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [user?.email]);

  useEffect(() => {
    if (!loading) return;
    const id = setInterval(() => {
      setProcessingIndex((i) => (i + 1) % processingSteps.length);
    }, 1400);
    return () => clearInterval(id);
  }, [loading, processingSteps.length]);

  const handleRestart = async () => {
    if (!user?.email) return;
    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
      await fetch(`${apiBase}/api/reset-progress`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_email: user.email }),
      });
      navigate('/dashboard/psychology', { replace: true });
    } catch {
      // soft-fail
      navigate('/dashboard/psychology', { replace: true });
    }
  };

  const handleCounselorRequest = async () => {
    if (!user?.email || requestingCounselor) return;
    setRequestingCounselor(true);
    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
      const res = await fetch(`${apiBase}/api/counseling/request`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_email: user.email }),
      });
      if (res.ok) {
        setCounselorRequested(true);
      }
    } catch {
      // soft fail
    } finally {
      setRequestingCounselor(false);
    }
  };

  if (loading) {
    const pct = Math.round(((processingIndex + 1) / processingSteps.length) * 100);
    return (
      <div className="career-quiz-container">
        <div className="quiz-main-content">
          <div className="ai-processing" role="status" aria-live="polite">
            <div className="ai-processing-title">
              <i className="fas fa-robot" aria-hidden="true" /> AI is processing your career intelligence
            </div>
            <div className="ai-processing-step">
              {processingSteps[processingIndex]}
              <span className="ai-dots" aria-hidden="true" />
            </div>
            <div className="ai-processing-bar" aria-hidden="true">
              <div className="ai-processing-fill" style={{ width: `${pct}%` }} />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="career-quiz-container">
        <div className="quiz-main-content">
          <p className="error-message">{error || 'Unable to load results.'}</p>
        </div>
      </div>
    );
  }

  const top = data.top_recommendation;
  const rankings = data.career_rankings || [];

  return (
    <div className="career-quiz-container show-results">
      <div className="quiz-main-content">
        <div className="quiz-results">
          <h3>Unified Career Result</h3>

          {top && (
            <div className="quiz-explanation">
              <h4>Why this decision?</h4>
              <p>{top.explanation}</p>
              <div className="quiz-contrib-row">
                {typeof top.quiz_component === 'number' && (
                  <span>
                    Quiz:{' '}
                    <span className="quiz-explanation-highlight">
                      {Math.round(top.quiz_component)}%
                    </span>
                  </span>
                )}
                {typeof top.psych_component === 'number' && (
                  <span>
                    Psychological:{' '}
                    <span className="quiz-explanation-highlight">
                      {Math.round(top.psych_component)}%
                    </span>
                  </span>
                )}
                {typeof top.voice_component === 'number' && (
                  <span>
                    Voice:{' '}
                    <span className="quiz-explanation-highlight voice-highlight">
                      {Math.round(top.voice_component)}%
                    </span>
                  </span>
                )}
                {typeof top.confidence_score === 'number' && (
                  <span>
                    Confidence:{' '}
                    <span className="quiz-explanation-highlight">
                      {Math.round(top.confidence_score)}%
                    </span>
                  </span>
                )}
              </div>
            </div>
          )}

          <div className="career-cards">
            {rankings.map((item) => (
              <div key={item.career} className="career-card">
                <div className="career-header">
                  <h4>{item.career}</h4>
                  <span className="career-score">
                    {Math.round(item.final_score)}%
                  </span>
                </div>
                <div className="career-components">
                  <div>
                    <span>Quiz</span>
                    <div className="bar">
                      <div
                        className="fill academic"
                        style={{ width: `${Math.round(item.quiz_component || 0)}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <span>Psychological</span>
                    <div className="bar">
                      <div
                        className="fill psych"
                        style={{ width: `${Math.round(item.psych_component || 0)}%` }}
                      />
                    </div>
                  </div>
                  {typeof item.voice_component === 'number' && (
                    <div>
                      <span>Voice</span>
                      <div className="bar">
                        <div
                          className="fill voice"
                          style={{ width: `${Math.round(item.voice_component || 0)}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Talk to a Counselor */}
          <div className="talk-to-counselor">
            <div className="ttc-icon">🧠</div>
            <div className="ttc-title">Talk to a Counselor</div>
            <div className="ttc-desc">
              Get personalized guidance from an expert career counselor who can interpret your AI results and provide real-world advice.
            </div>
            {counselorRequested ? (
              <div className="ttc-success">
                <i className="fas fa-check-circle" /> Request sent! A counselor will review your profile.
              </div>
            ) : (
              <button
                className="ttc-btn"
                onClick={handleCounselorRequest}
                disabled={requestingCounselor}
              >
                <i className="fas fa-hand-paper" />
                {requestingCounselor ? 'Sending...' : 'Request Counselor Help'}
              </button>
            )}
          </div>

          <button
            type="button"
            className="btn-nav btn-retake"
            onClick={handleRestart}
          >
            <i className="fas fa-redo" /> Start New Career Assessment
          </button>
        </div>
      </div>
    </div>
  );
};

export default Results;

