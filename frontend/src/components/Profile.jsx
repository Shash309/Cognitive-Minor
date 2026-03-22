import React, { useEffect, useRef, useState } from 'react';
import { useOutletContext, useNavigate, useLocation } from 'react-router-dom';
import './Profile.css';
import './CounselorDashboard.css';

const toTitleCase = (value) => {
  if (!value || typeof value !== 'string') return '';
  return value
    .replace(/_/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const Profile = () => {
  const { user } = useOutletContext() || {};
  const navigate = useNavigate();
  const location = useLocation();
  const [profile, setProfile] = useState(null);
  const [quizHistory, setQuizHistory] = useState([]);
  const [psychHistory, setPsychHistory] = useState([]);
  const [careerSnapshot, setCareerSnapshot] = useState(null);
  const [voiceHistory, setVoiceHistory] = useState([]);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [feedbackData, setFeedbackData] = useState([]);
  const [showResultBanner, setShowResultBanner] = useState(
    Boolean(location.state && location.state.highlightLatestResult)
  );
  const isCounselorEnabled = import.meta.env.VITE_ENABLE_COUNSELLOR_FEATURE === "true";
  const latestSessionRef = useRef(null);

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
        if (Array.isArray(profileData.voice_history)) {
          setVoiceHistory(profileData.voice_history);
        }
        if (Array.isArray(profileData.career_sessions)) {
          setSessionHistory(profileData.career_sessions);
        }
        if (!fusedData.error && Array.isArray(fusedData.career_rankings)) {
          setCareerSnapshot(fusedData);
        }

        // Fetch counselor feedback
        try {
          const fbRes = await fetch(
            `${apiBase}/api/counseling/feedback?student_email=${encodeURIComponent(user.email)}`
          );
          const fbData = await fbRes.json();
          if (Array.isArray(fbData.feedback)) {
            setFeedbackData(fbData.feedback);
          }
        } catch {
          // non-fatal
        }
      } catch (err) {
        setError(err.message || 'Something went wrong while loading profile.');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [user?.email]);

  useEffect(() => {
    if (showResultBanner && latestSessionRef.current) {
      latestSessionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [showResultBanner, sessionHistory]);

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
  const fusedTop =
    careerSnapshot?.top_recommendation ||
    careerSnapshot?.career_rankings?.[0] ||
    profile?.fused_top;

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

  const hasCareerResults =
    !!careerSnapshot &&
    Array.isArray(careerSnapshot.career_rankings) &&
    careerSnapshot.career_rankings.length > 0;
  const hasPsychHistory = Array.isArray(psychHistory) && psychHistory.length > 0;
  const hasVoiceHistory = Array.isArray(voiceHistory) && voiceHistory.length > 0;
  const hasQuizHistory = Array.isArray(quizHistory) && quizHistory.length > 0;
  const showAiInsight = hasCareerResults && hasPsychHistory && hasVoiceHistory && hasQuizHistory;

  const latestPsychEntry = hasPsychHistory ? psychHistory[0] : null;
  const decisionStyle =
    latestPsychEntry?.decision_style || careerSnapshot?.decision_style || null;

  const dominantTraits = Array.isArray(latestPsychEntry?.dominant_traits)
    ? latestPsychEntry.dominant_traits.slice(0, 3)
    : [];
  const dominantTraitNames = dominantTraits
    .map((t) => t.display_name || toTitleCase(t.name))
    .filter(Boolean);

  const voiceInsight =
    careerSnapshot?.voice_insight || (hasVoiceHistory ? voiceHistory[0] : null);

  const levelLabel = (score) => {
    if (typeof score !== 'number') return null;
    if (score >= 70) return 'strong';
    if (score >= 45) return 'moderate';
    return 'emerging';
  };

  const motivationLevel = levelLabel(voiceInsight?.motivation_score);
  const confidenceLevel = levelLabel(voiceInsight?.confidence_score);

  const voiceBits = [];
  if (confidenceLevel) {
    voiceBits.push(`${confidenceLevel} vocal confidence`);
  }
  if (motivationLevel) {
    voiceBits.push(`${motivationLevel} motivation when talking about goals`);
  }
  if (voiceInsight?.top_voice_career) {
    voiceBits.push(`signals that align with ${voiceInsight.top_voice_career}`);
  }
  const behavioralSummary = voiceBits.length > 0 ? `${voiceBits.join('. ')}.` : '';

  const topRecommendation = careerSnapshot?.top_recommendation;
  const careerAlignmentExplanation =
    (topRecommendation && topRecommendation.explanation) ||
    (fusedTop?.career
      ? `Your combined quiz, psychological, and voice signals currently align most strongly with ${fusedTop.career}.`
      : '');

  return (
    <div className="profile-page">
      {showResultBanner && (
        <div className="profile-banner-success">
          <span>Your Career Decision is Ready.</span>
          <button
            type="button"
            className="profile-banner-close"
            onClick={() => setShowResultBanner(false)}
          >
            ×
          </button>
        </div>
      )}
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

      {/* Card 5 – Voice Insight History */}
      <div className="profile-card-full">
        <div className="profile-card-title">Voice Insight History</div>
        <div className="profile-card-accent" />
        {voiceHistory.length === 0 ? (
          <p className="profile-mini-value" style={{ marginTop: '10px' }}>
            No voice insights captured yet.
          </p>
        ) : (
          <div className="profile-history-list">
            {voiceHistory.map((entry) => (
              <div key={entry.timestamp} className="profile-history-item">
                <div className="profile-history-date">
                  {entry.timestamp ? new Date(entry.timestamp).toLocaleString() : 'Unknown'}
                </div>
                <div className="profile-history-title">
                  Transcript
                </div>
                <div className="profile-mini-value">
                  {(entry.transcript || '').slice(0, 140)}
                  {(entry.transcript || '').length > 140 ? '…' : ''}
                </div>
                <div className="profile-history-footer">
                  <div>
                    {typeof entry.motivation_score === 'number' && (
                      <span className="profile-tag profile-tag-red">
                        Motivation {Math.round(entry.motivation_score)}%
                      </span>
                    )}
                    {typeof entry.confidence_score === 'number' && (
                      <span className="profile-tag" style={{ marginLeft: '0.5rem' }}>
                        Confidence {Math.round(entry.confidence_score)}%
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Card 6 – Career Decision Results */}
      <div className="profile-card-full">
        <div className="profile-card-title">Career Decision Results</div>
        <div className="profile-card-accent" />
        {sessionHistory.length === 0 ? (
          <p className="profile-mini-value" style={{ marginTop: '10px' }}>
            No completed career assessment sessions yet.
          </p>
        ) : (
          <div className="profile-history-list">
            {sessionHistory.map((session, idx) => {
              const weights = session.weights || {};
              const wQuiz =
                typeof weights.quiz === 'number' ? Math.round(weights.quiz * 100) : null;
              const wPsych =
                typeof weights.psych === 'number' ? Math.round(weights.psych * 100) : null;
              const wVoice =
                typeof weights.voice === 'number' ? Math.round(weights.voice * 100) : null;

              const topCareer = session.top_career;
              let topScore = null;
              if (
                session.final_scores &&
                topCareer &&
                session.final_scores[topCareer] != null
              ) {
                topScore = Math.round(session.final_scores[topCareer]);
              }

              return (
                <div
                  key={session.timestamp}
                  className="profile-history-item"
                  ref={idx === 0 ? latestSessionRef : null}
                >
                  <div className="profile-history-date">
                    {session.timestamp
                      ? new Date(session.timestamp).toLocaleString()
                      : 'Unknown'}
                  </div>
                  <div className="profile-history-title">
                    {topCareer || '—'}
                  </div>
                  <div className="profile-mini-row">
                    {topScore != null && (
                      <span className="profile-mini-label">
                        Final score:&nbsp;
                        <span className="profile-mini-value">{topScore}%</span>
                      </span>
                    )}
                  </div>
                  <div className="profile-mini-row">
                    {typeof session.confidence_score === 'number' && (
                      <span className="profile-tag profile-tag-red">
                        Confidence {Math.round(session.confidence_score)}%
                      </span>
                    )}
                  </div>
                  <div className="profile-mini-row" style={{ marginTop: '0.35rem' }}>
                    <span className="profile-mini-label">Signal contributions</span>
                    <div className="profile-mini-value">
                      <span>Quiz {wQuiz != null ? `${wQuiz}%` : '—'}</span>{' · '}
                      <span>Psych {wPsych != null ? `${wPsych}%` : '—'}</span>{' · '}
                      <span>Voice {wVoice != null ? `${wVoice}%` : '—'}</span>
                    </div>
                  </div>
                  <div className="profile-history-footer" style={{ marginTop: '0.35rem' }}>
                    <button
                      type="button"
                      className="profile-details-btn"
                      onClick={() => navigate('/dashboard/results')}
                    >
                      View Full Results
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {showAiInsight && (
        <div className="profile-card-full">
          <div className="profile-card-title">
            <i className="fas fa-robot" style={{ marginRight: '0.4rem' }} /> AI Insight
          </div>
          <div className="profile-card-accent" />

          {decisionStyle && (
            <div style={{ marginTop: '10px' }}>
              <span className="profile-mini-label">Decision style</span>
              <div className="profile-mini-value">{decisionStyle}</div>
            </div>
          )}

          {dominantTraitNames.length > 0 && (
            <div style={{ marginTop: '10px' }}>
              <span className="profile-mini-label">Dominant traits</span>
              <div className="profile-pill-row">
                {dominantTraitNames.map((name) => (
                  <span key={name} className="profile-tag profile-tag-red">
                    {name}
                  </span>
                ))}
              </div>
            </div>
          )}

          {behavioralSummary && (
            <div style={{ marginTop: '10px' }}>
              <span className="profile-mini-label">Behavioral signals</span>
              <div className="profile-mini-value">{behavioralSummary}</div>
            </div>
          )}

          {careerAlignmentExplanation && (
            <div style={{ marginTop: '10px' }}>
              <span className="profile-mini-label">Career alignment explanation</span>
              <div className="profile-mini-value">
                {careerAlignmentExplanation}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Card 7 – Counselor Feedback */}
      {isCounselorEnabled && feedbackData.length > 0 && (
        <div className="profile-card-full">
          <div className="profile-card-title">
            <i className="fas fa-user-tie" style={{ marginRight: '0.4rem' }} /> Counselor Feedback
          </div>
          <div className="profile-card-accent" />
          <div className="counselor-feedback-section">
            {feedbackData.map((fb, idx) => (
              <div key={idx} className="feedback-card">
                <div className="feedback-counselor">🧠 {fb.counselor_name}</div>
                <div className="feedback-text">{fb.text}</div>
                {fb.timestamp && (
                  <div className="feedback-time">
                    {new Date(fb.timestamp).toLocaleString()}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Profile;

