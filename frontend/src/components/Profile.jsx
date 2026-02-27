import React, { useEffect, useState } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import './Profile.css';

const Profile = () => {
  const { user } = useOutletContext() || {};
  const navigate = useNavigate();
  const [profile, setProfile] = useState(null);
  const [quizHistory, setQuizHistory] = useState([]);
  const [psychHistory, setPsychHistory] = useState([]);
  const [careerSnapshot, setCareerSnapshot] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      if (!user?.email) {
        setLoading(false);
        return;
      }
      setLoading(true);
      setError('');
      try {
        const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

        const [profileRes, quizRes, psychRes, fusedRes] = await Promise.all([
          fetch(`${apiBase}/api/profile?user_email=${encodeURIComponent(user.email)}`),
          fetch(`${apiBase}/api/quiz-history?user_email=${encodeURIComponent(user.email)}`),
          fetch(`${apiBase}/api/psych-assessment?user_email=${encodeURIComponent(user.email)}`),
          fetch(`${apiBase}/api/career-results?user_email=${encodeURIComponent(user.email)}`),
        ]);

        const profileData = await profileRes.json();
        if (!profileRes.ok || profileData.error) {
          throw new Error(profileData.error || 'Unable to load profile.');
        }

        const quizData = await quizRes.json();
        const psychData = await psychRes.json();
        const fusedData = await fusedRes.json();

        setProfile(profileData);
        if (Array.isArray(quizData.attempts)) {
          setQuizHistory(quizData.attempts);
        }
        if (Array.isArray(psychData.history)) {
          setPsychHistory(psychData.history);
        }
        if (!fusedData.error && Array.isArray(fusedData.career_rankings)) {
          setCareerSnapshot(fusedData);
        }
      } catch (err) {
        setError(err.message || 'Something went wrong while loading profile.');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [user?.email]);

  if (!user?.email) {
    return <p>Please sign in to view your profile.</p>;
  }

  if (loading) {
    return <p>Loading profile…</p>;
  }

  if (error) {
    return <p className="error-message">{error}</p>;
  }

  const personal = profile?.user || {};
  const fusedTop = careerSnapshot?.career_rankings?.[0] || profile?.fused_top;

  const getInitial = () => {
    const name = personal.name || user.name || '';
    return name ? name.charAt(0).toUpperCase() : (user.email || 'U').charAt(0).toUpperCase();
  };

  const computeConfidence = (attempt) => {
    const scores = attempt?.quiz_scores;
    if (!scores || typeof scores !== 'object') return null;
    let maxScore = -1;
    Object.values(scores).forEach((v) => {
      const num = Number(v);
      if (!Number.isNaN(num)) {
        maxScore = Math.max(maxScore, num);
      }
    });
    return maxScore >= 0 ? maxScore : null;
  };

  const quizComponent = fusedTop?.quiz_component;
  const psychComponent = fusedTop?.psych_component;

  return (
    <div className="profile-page">
      <h2 className="profile-header-title">Profile</h2>

      <div className="profile-grid">
        {/* Card 1 – Personal Information */}
        <div className="profile-card">
          <div className="profile-card-header">
            <div>
              <div className="profile-card-title">Personal Information</div>
              <div className="profile-card-accent" />
            </div>
            <button className="profile-edit-btn" type="button" title="Edit profile (coming soon)">
              <i className="fas fa-pen" />
            </button>
          </div>
          <div className="profile-personal-body">
            <div className="profile-avatar-circle">
              {getInitial()}
            </div>
            <div className="profile-personal-fields">
              <div>
                <div className="profile-field-label">Name</div>
                <div className="profile-field-value">{personal.name || user.name || '—'}</div>
              </div>
              <div>
                <div className="profile-field-label">Email</div>
                <div className="profile-field-value">{personal.email || user.email}</div>
              </div>
              {personal.created_at && (
                <div>
                  <div className="profile-field-label">Account created</div>
                  <div className="profile-field-value">
                    {new Date(personal.created_at).toLocaleString()}
                  </div>
                </div>
              )}
              {personal.last_login && (
                <div>
                  <div className="profile-field-label">Last login</div>
                  <div className="profile-field-value">
                    {new Date(personal.last_login).toLocaleString()}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Card 2 – Career Snapshot */}
        <div className="profile-card">
          <div className="profile-card-header">
            <div>
              <div className="profile-card-title">Career Snapshot</div>
              <div className="profile-card-accent" />
            </div>
          </div>

          {fusedTop ? (
            <>
              <div className="career-main-value">{fusedTop.career}</div>
              <div className="career-score-highlight">
                {Math.round(fusedTop.final_score)}%
              </div>
              <div className="career-progress-bar">
                <div
                  className="career-progress-fill"
                  style={{ width: `${Math.round(fusedTop.final_score)}%` }}
                />
              </div>
              <div className="career-breakdown-row">
                <span>
                  <strong>Quiz:</strong>{' '}
                  {quizComponent != null ? `${Math.round(quizComponent)}%` : '—'}
                </span>
                <span>
                  <strong>Psychological:</strong>{' '}
                  {psychComponent != null ? `${Math.round(psychComponent)}%` : '—'}
                </span>
              </div>
            </>
          ) : (
            <p className="profile-mini-value">No fused career data yet.</p>
          )}
        </div>
      </div>

      {/* Card 3 – AI Career Quiz History */}
      <div className="profile-card-full">
        <div className="profile-card-title">AI Career Quiz History</div>
        <div className="profile-card-accent" />
        {quizHistory.length === 0 ? (
          <p className="profile-mini-value" style={{ marginTop: '10px' }}>
            No quiz attempts yet.
          </p>
        ) : (
          <div className="profile-history-list">
            {quizHistory.map((attempt, idx) => {
              const confidence = computeConfidence(attempt);
              return (
                <div key={attempt.timestamp || idx} className="profile-history-item">
                  <div className="profile-history-date">
                    {attempt.timestamp
                      ? new Date(attempt.timestamp).toLocaleString()
                      : 'Unknown'}
                  </div>
                  <div className="profile-history-title">
                    {attempt.top_career || '—'}
                  </div>
                  <div>
                    <span className="profile-mini-label">Stream</span>
                    <div className="profile-mini-value">
                      {attempt.stream || 'Not specified'}
                    </div>
                  </div>
                  {typeof attempt.academic_percent === 'number' && (
                    <div>
                      <span className="profile-mini-label">Academic %</span>
                      <div className="profile-mini-value">
                        {attempt.academic_percent.toFixed(1)}%
                      </div>
                    </div>
                  )}
                  <div className="profile-history-footer">
                    <div>
                      {confidence != null && (
                        <span className="profile-tag profile-tag-red">
                          Confidence {Math.round(confidence)}%
                        </span>
                      )}
                    </div>
                    <button
                      type="button"
                      className="profile-details-btn"
                      onClick={() =>
                        attempt.id &&
                        navigate(`/dashboard/quiz-result/${encodeURIComponent(attempt.id)}`)
                      }
                    >
                      View Details
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Card 4 – Psychological Analysis History */}
      <div className="profile-card-full">
        <div className="profile-card-title">Psychological Assessment History</div>
        <div className="profile-card-accent" />
        {psychHistory.length === 0 ? (
          <p className="profile-mini-value" style={{ marginTop: '10px' }}>
            No psychological assessments yet.
          </p>
        ) : (
          <div className="profile-history-list">
            {psychHistory.map((entry) => (
              <div key={entry.completed_at} className="profile-history-item">
                <div className="profile-history-date">
                  {entry.completed_at
                    ? new Date(entry.completed_at).toLocaleString()
                    : 'Unknown'}
                </div>
                <div className="profile-history-title">
                  {entry.top_career || 'Top career unavailable'}
                </div>
                <div>
                  <span className="profile-mini-label">Decision style</span>
                  <div className="profile-pill-row">
                    {entry.decision_style && (
                      <span className="profile-tag profile-tag-gray">
                        {entry.decision_style}
                      </span>
                    )}
                  </div>
                </div>
                {Array.isArray(entry.dominant_traits) && entry.dominant_traits.length > 0 && (
                  <div>
                    <span className="profile-mini-label">Top strengths</span>
                    <div className="profile-pill-row">
                      {entry.dominant_traits.slice(0, 3).map((t) => (
                        <span key={t.name} className="profile-tag profile-tag-red">
                          {t.display_name || t.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {entry.stability_label && (
                  <div style={{ marginTop: 4 }}>
                    <span className="profile-mini-label">Stability</span>
                    <div className="profile-pill-row">
                      <span className="profile-tag profile-tag-gray">
                        {entry.stability_label}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Profile;

