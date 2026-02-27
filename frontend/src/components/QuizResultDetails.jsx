import React, { useEffect, useState } from 'react';
import { useParams, useOutletContext, useNavigate } from 'react-router-dom';
import './CareerQuiz.css';

const QuizResultDetails = () => {
  const { attemptId } = useParams();
  const { user } = useOutletContext() || {};
  const navigate = useNavigate();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      if (!user?.email) {
        setError('Please sign in to view quiz results.');
        setLoading(false);
        return;
      }
      setLoading(true);
      setError('');
      try {
        const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
        const res = await fetch(
          `${apiBase}/api/quiz-attempt?user_email=${encodeURIComponent(
            user.email
          )}&attempt_id=${encodeURIComponent(attemptId)}`
        );
        const json = await res.json();
        if (!res.ok || json.error) {
          throw new Error(json.error || 'Result not found.');
        }
        setData(json);
      } catch (err) {
        setError(err.message || 'Result not found.');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [attemptId, user?.email]);

  if (loading) {
    return (
      <div className="career-quiz-container">
        <div className="quiz-main-content">
          <p>Loading quiz result…</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="career-quiz-container">
        <div className="quiz-main-content">
          <p className="error-message">{error || 'Result not found.'}</p>
          <button
            type="button"
            className="btn-nav btn-back"
            onClick={() => navigate('/dashboard/profile')}
          >
            ← Back to profile
          </button>
        </div>
      </div>
    );
  }

  const { attempt, career_rankings, top_recommendation } = data;
  const top = top_recommendation;
  const rankings = career_rankings || [];

  return (
    <div className="career-quiz-container">
      <div className="quiz-main-content">
        <button
          type="button"
          className="btn-nav btn-back"
          onClick={() => navigate('/dashboard/profile')}
        >
          ← Back to profile
        </button>

        <div className="quiz-results" style={{ paddingTop: '1.5rem' }}>
          <h3>AI Career Quiz Result</h3>
          {attempt?.top_career && (
            <p>
              <strong>Top career:</strong>{' '}
              <span className="quiz-explanation-highlight">{attempt.top_career}</span>
            </p>
          )}

          {top && (
            <div className="quiz-explanation">
              <h4>Why this career?</h4>
              <p>{top.explanation}</p>
              <div className="quiz-contrib-row">
                {typeof top.quiz_component === 'number' && (
                  <span>
                    Quiz alignment:{' '}
                    <span className="quiz-explanation-highlight">
                      {Math.round(top.quiz_component)}%
                    </span>
                  </span>
                )}
                {typeof top.psych_component === 'number' && (
                  <span>
                    Psychological alignment:{' '}
                    <span className="quiz-explanation-highlight">
                      {Math.round(top.psych_component)}%
                    </span>
                  </span>
                )}
                {typeof attempt?.confidence === 'number' && (
                  <span>
                    Quiz confidence:{' '}
                    <span className="quiz-explanation-highlight">
                      {Math.round(attempt.confidence)}%
                    </span>
                  </span>
                )}
              </div>
              <div className="quiz-badge-row">
                {Array.isArray(top.top_traits) &&
                  top.top_traits.map((trait) => (
                    <span key={trait.trait} className="quiz-badge trait">
                      {trait.trait.replace(/_/g, ' ')} · {Math.round(trait.user_score)}%
                    </span>
                  ))}
                {Array.isArray(top.quiz_signals?.matched_skills) &&
                  top.quiz_signals.matched_skills.map((skill) => (
                    <span key={skill} className="quiz-badge skill">
                      {skill}
                    </span>
                  ))}
              </div>
              {typeof attempt?.academic_percent === 'number' && (
                <div style={{ marginTop: '0.75rem', fontSize: '0.9rem' }}>
                  Academic percentage:{' '}
                  <span className="quiz-explanation-highlight">
                    {attempt.academic_percent.toFixed(1)}%
                  </span>{' '}
                  · Stream:{' '}
                  <span className="quiz-explanation-highlight">
                    {attempt.stream || 'Not specified'}
                  </span>
                </div>
              )}
            </div>
          )}

          {rankings.length > 0 && (
            <div style={{ marginTop: '2rem', textAlign: 'left' }}>
              <h4>Full career ranking</h4>
              <ul style={{ listStyle: 'none', padding: 0, marginTop: '0.75rem' }}>
                {rankings.map((item) => (
                  <li
                    key={item.career}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      padding: '0.4rem 0',
                      borderBottom: '1px solid var(--glass-border)',
                    }}
                  >
                    <span>{item.career}</span>
                    <span className="quiz-explanation-highlight">
                      {Math.round(item.final_score)}%
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default QuizResultDetails;

