import React, { useState, useEffect, useMemo } from 'react';
import { useOutletContext } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import './PsychAssessment.css';

const QUESTIONS = [
  // Big Five – Openness
  { id: 'O1', dimension: 'openness', text: 'I actively seek out new experiences, ideas, or perspectives.' },
  { id: 'O2', dimension: 'openness', text: 'I enjoy solving abstract or theoretical problems.', scenario: true },
  { id: 'O3', dimension: 'creativity_preference', text: 'I prefer work where I can create or experiment rather than follow fixed procedures.' },

  // Big Five – Conscientiousness
  { id: 'C1', dimension: 'conscientiousness', text: 'I keep track of my tasks and rarely miss deadlines.' },
  { id: 'C2', dimension: 'conscientiousness', text: 'I plan my work carefully before starting.' },
  { id: 'C3', dimension: 'structure_preference', text: 'I feel more comfortable when there is a clear structure and schedule.' },

  // Big Five – Extraversion
  { id: 'E1', dimension: 'extraversion', text: 'I feel energized after interacting with groups of people.' },
  { id: 'E2', dimension: 'extraversion', text: 'I enjoy being the center of attention in group discussions or presentations.' },
  { id: 'E3', dimension: 'individual_contributor', text: 'I prefer working independently rather than in large teams.', reverse: true },

  // Big Five – Agreeableness
  { id: 'A1', dimension: 'agreeableness', text: 'I try to understand other people’s feelings before making decisions that affect them.' },
  { id: 'A2', dimension: 'agreeableness', text: 'People often describe me as cooperative and easy to work with.' },
  { id: 'A3', dimension: 'leadership_index', text: 'I can balance firm decisions with empathy for others.' },

  // Big Five – Neuroticism (emotional stability)
  { id: 'N1', dimension: 'neuroticism', text: 'Unexpected setbacks make me feel very anxious or overwhelmed.', reverse: true },
  { id: 'N2', dimension: 'neuroticism', text: 'I stay calm and focused even when things do not go as planned.', reverse: false },
  { id: 'N3', dimension: 'stress_tolerance', text: 'I can continue performing well under sustained pressure (e.g., exams, deadlines).' },

  // Decision-making style
  { id: 'D1', dimension: 'analytical_thinking', text: 'When making decisions, I rely more on data, facts, and logic than on intuition.' },
  { id: 'D2', dimension: 'analytical_thinking', text: 'I enjoy breaking complex problems into smaller logical steps.' },
  { id: 'D3', dimension: 'intuitive_preference', text: 'In ambiguous situations, I trust my gut feeling more than detailed analysis.', reverse: false },

  // Risk tolerance
  { id: 'R1', dimension: 'risk_tolerance', text: 'I am comfortable taking calculated risks if the potential reward is high.' },
  { id: 'R2', dimension: 'risk_tolerance', text: 'I prefer stable and predictable paths, even if growth is slower.', reverse: true },

  // Motivation
  { id: 'M1', dimension: 'intrinsic_motivation', text: 'I am motivated by learning and mastery, even without external rewards.' },
  { id: 'M2', dimension: 'intrinsic_motivation', text: 'I can stay engaged in a task simply because I find it interesting.' },
  { id: 'M3', dimension: 'extrinsic_motivation', text: 'Recognition, titles, or salary strongly influence my career choices.' },

  // Leadership vs IC
  { id: 'L1', dimension: 'leadership_index', text: 'I enjoy coordinating people and taking responsibility for team outcomes.' },
  { id: 'L2', dimension: 'leadership_index', text: 'In group projects, I naturally move into a leadership or organizing role.' },
  { id: 'L3', dimension: 'individual_contributor', text: 'I prefer to be an expert contributing deep work rather than managing others.' },

  // Creativity vs structure
  { id: 'CS1', dimension: 'creativity_preference', text: 'I feel energized when brainstorming novel ideas or unconventional solutions.' },
  { id: 'CS2', dimension: 'structure_preference', text: 'I am most productive when there are clear rules, checklists, and guidelines.' },

  // Scenario-based / behavioral frequency
  { id: 'S1', dimension: 'stress_tolerance', text: 'Before a major exam or presentation, I can manage my stress and stay focused on preparation.' },
  { id: 'S2', dimension: 'analytical_thinking', text: 'When friends ask for advice, I often help them think through pros and cons logically.' },
  { id: 'S3', dimension: 'openness', text: 'If I had the option, I would gladly move to a new city or country to pursue an interesting opportunity.' },
];

const LIKERT_OPTIONS = [
  { value: 1, label: 'Strongly Disagree' },
  { value: 2, label: 'Disagree' },
  { value: 3, label: 'Neutral' },
  { value: 4, label: 'Agree' },
  { value: 5, label: 'Strongly Agree' },
];

const RadarChart = ({ profile }) => {
  const traitKeys = useMemo(
    () => [
      { key: 'openness', label: 'Openness' },
      { key: 'conscientiousness', label: 'Conscientiousness' },
      { key: 'extraversion', label: 'Extraversion' },
      { key: 'agreeableness', label: 'Agreeableness' },
      { key: 'neuroticism', label: 'Emotional Stability', invert: true },
      { key: 'analytical_thinking', label: 'Analytical' },
      { key: 'risk_tolerance', label: 'Risk' },
      { key: 'leadership_index', label: 'Leadership' },
    ],
    []
  );

  const size = 260;
  const center = size / 2;
  const radius = size * 0.38;
  const levels = [0.25, 0.5, 0.75, 1];

  const pointsForLevel = (level) =>
    traitKeys
      .map((t, i) => {
        const angle = (2 * Math.PI * i) / traitKeys.length - Math.PI / 2;
        const r = radius * level;
        const x = center + r * Math.cos(angle);
        const y = center + r * Math.sin(angle);
        return `${x},${y}`;
      })
      .join(' ');

  const profilePoints = traitKeys
    .map((t, i) => {
      let value = profile?.[t.key] ?? 0;
      if (t.invert) {
        value = 100 - value;
      }
      const level = Math.max(0, Math.min(1, value / 100));
      const angle = (2 * Math.PI * i) / traitKeys.length - Math.PI / 2;
      const r = radius * level;
      const x = center + r * Math.cos(angle);
      const y = center + r * Math.sin(angle);
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <div className="radar-chart-wrapper">
      <svg width={size} height={size} className="radar-chart">
        <circle cx={center} cy={center} r={radius} className="radar-circle-base" />
        {levels.map((lvl, idx) => (
          <polygon
            key={idx}
            points={pointsForLevel(lvl)}
            className="radar-level"
          />
        ))}
        {traitKeys.map((t, i) => {
          const angle = (2 * Math.PI * i) / traitKeys.length - Math.PI / 2;
          const x = center + radius * Math.cos(angle);
          const y = center + radius * Math.sin(angle);
          const labelX = center + (radius + 18) * Math.cos(angle);
          const labelY = center + (radius + 18) * Math.sin(angle);
          return (
            <g key={t.key}>
              <line
                x1={center}
                y1={center}
                x2={x}
                y2={y}
                className="radar-axis"
              />
              <text
                x={labelX}
                y={labelY}
                textAnchor="middle"
                dominantBaseline="middle"
                className="radar-label"
              >
                {t.label}
              </text>
            </g>
          );
        })}
        <polygon points={profilePoints} className="radar-profile" />
      </svg>
    </div>
  );
};

const PsychAssessment = () => {
  const { t } = useTranslation();
  const { user } = useOutletContext() || {};
  const [answers, setAnswers] = useState({});
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [showRetakeConfirm, setShowRetakeConfirm] = useState(false);
  const [fusion, setFusion] = useState(null);

  const totalQuestions = QUESTIONS.length;
  const progress = ((currentIndex + 1) / totalQuestions) * 100;
  const estimatedMinutes = Math.round(totalQuestions * 0.35); // ~20–25 seconds per question

  const storageKey = user?.email ? `psych_assessment_${user.email}` : null;

  useEffect(() => {
    if (!storageKey) return;
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed.answers) setAnswers(parsed.answers);
      }
    } catch {
      // ignore localStorage errors
    }
  }, [storageKey]);

  useEffect(() => {
    if (!storageKey) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify({ answers }));
    } catch {
      // ignore
    }
  }, [answers, storageKey]);

  const handleChange = (qid, value) => {
    setAnswers((prev) => ({ ...prev, [qid]: value }));
  };

  const handleNext = () => {
    if (currentIndex < totalQuestions - 1) {
      setCurrentIndex((idx) => idx + 1);
    }
  };

  const handleBack = () => {
    if (currentIndex > 0) {
      setCurrentIndex((idx) => idx - 1);
    }
  };

  const allAnswered = useMemo(
    () => QUESTIONS.every((q) => typeof answers[q.id] === 'number'),
    [answers]
  );

  const loadExistingProfile = async () => {
    if (!user?.email) return;
    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

      const res = await fetch(
        `${apiBase}/api/psych-assessment?user_email=${encodeURIComponent(user.email)}`
      );
      if (res.ok) {
        const data = await res.json();
        if (data.profile) {
          setResult(data);
        }
        if (Array.isArray(data.history)) {
          setHistory(data.history);
        }
      }

      const fusedRes = await fetch(
        `${apiBase}/api/career-results?user_email=${encodeURIComponent(user.email)}`
      );
      if (fusedRes.ok) {
        const fusedData = await fusedRes.json();
        if (!fusedData.error) {
          setFusion(fusedData);
        }
      }
    } catch {
      // non-fatal
    }
  };

  useEffect(() => {
    loadExistingProfile();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.email]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!user?.email) {
      setError('Please sign in to save your psychological profile.');
      return;
    }
    if (!allAnswered) {
      setError('Please answer all questions before submitting.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
      const res = await fetch(
        `${apiBase}/api/psych-assessment`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_email: user.email,
            responses: answers,
            retake_reason: null,
          }),
        }
      );
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || 'Unable to compute profile.');
      }
      setResult(data);
      if (Array.isArray(data.history)) {
        setHistory(data.history);
      }

      // Refresh unified fused rankings after new assessment
      const fusedRes = await fetch(
        `${apiBase}/api/career-results?user_email=${encodeURIComponent(user.email)}`
      );
      if (fusedRes.ok) {
        const fusedData = await fusedRes.json();
        if (!fusedData.error) {
          setFusion(fusedData);
        }
      }
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleRetake = () => {
    setResult(null);
    setCurrentIndex(0);
    setAnswers({});
    setShowRetakeConfirm(false);
  };

  const currentQuestion = QUESTIONS[currentIndex];

  if (result?.profile) {
    const dominantTraits = (result.dominant_traits || []).slice(0, 3);
    const fusedRankings = fusion?.career_rankings || [];
    const topCareer = fusedRankings[0];
    const stabilityIndex =
      typeof result.stability_index === 'number' ? result.stability_index : null;
    const stabilityPercent =
      stabilityIndex !== null ? Math.round(stabilityIndex * 100) : null;

    return (
      <div className="psych-container">
        <div className="psych-header">
          <div>
            <h2>{t('psych.title', 'Psychological Analysis')}</h2>
            <p>
              {t(
                'psych.subtitle',
                'Understand your cognitive profile and how it aligns with careers.'
              )}
            </p>
            <div className="psych-title-underline" />
          </div>
        </div>

        <div className="psych-layout">
          <div className="psych-main">
            <section className="psych-panel psych-panel-profile">
              <div className="psych-panel-header">
                <div className="psych-icon-circle">
                  <i className="fas fa-brain" />
                </div>
                <div>
                  <h3>{t('psych.profileOverview', 'Your Cognitive Profile')}</h3>
                  {stabilityPercent !== null && (
                    <p className="stability-text">
                      {t('psych.stabilityLabel', 'Your cognitive profile stability')}:{' '}
                      <span className="stability-strong">
                        {stabilityPercent}%{' '}
                        {result.stability_label || ''}
                      </span>
                    </p>
                  )}
                </div>
              </div>
              <div className="psych-overview-grid">
                <RadarChart profile={result.profile} />
                <div className="psych-summary">
                  <h4>{t('psych.dominantStrengths', 'Dominant strengths')}</h4>
                  <ul>
                    {dominantTraits.map((trait) => (
                      <li key={trait.name}>
                        <span className="trait-name">{trait.display_name}</span>
                        <span className="trait-score">{Math.round(trait.score)} / 100</span>
                      </li>
                    ))}
                  </ul>
                  {result.decision_style && (
                    <p className="decision-style">
                      <strong>{t('psych.decisionStyle', 'Decision-making style')}:</strong>{' '}
                      {result.decision_style}
                    </p>
                  )}
                  {topCareer && (
                    <p className="career-alignment">
                      <strong>{t('psych.topCareerAlignment', 'Top career match')}:</strong>{' '}
                      {topCareer.career} ({Math.round(topCareer.overall_score)}%)
                    </p>
                  )}
                </div>
              </div>
            </section>

            {fusion?.voice_insight && (
              <div className="voice-insight-section">
                  <h4><i className="fas fa-microphone-alt" /> Voice Insight</h4>
                  {fusion.voice_insight.transcript && (
                    <p className="voice-transcript-preview">{fusion.voice_insight.transcript}</p>
                  )}
                  <div className="voice-insight-metrics">
                    {typeof fusion.voice_insight.motivation_score === 'number' && (
                      <span>Motivation: <span className="voice-highlight">{Math.round(fusion.voice_insight.motivation_score)}%</span></span>
                    )}
                    {typeof fusion.voice_insight.confidence_score === 'number' && (
                      <span>Confidence: <span className="voice-highlight">{Math.round(fusion.voice_insight.confidence_score)}%</span></span>
                    )}
                    {fusion.voice_insight.top_voice_career && (
                      <span>Top (voice): <span className="voice-highlight">{fusion.voice_insight.top_voice_career}</span></span>
                    )}
                  </div>
              </div>
            )}

            <section className="psych-panel">
              <h3>{t('psych.careerRecommendations', 'Career recommendations')}</h3>
              <div className="career-cards">
                {fusedRankings.map((item) => {
                  const psychDetail = (result.career_matches || []).find(
                    (c) => c.career === item.career
                  );
                  return (
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
                              style={{
                                width: `${Math.round(item.quiz_component || 0)}%`,
                              }}
                            />
                          </div>
                        </div>
                        <div>
                          <span>Psychological</span>
                          <div className="bar">
                            <div
                              className="fill psych"
                              style={{
                                width: `${Math.round(item.psych_component || 0)}%`,
                              }}
                            />
                          </div>
                        </div>
                        {typeof item.voice_component === 'number' && (
                          <div>
                            <span>Voice</span>
                            <div className="bar">
                              <div
                                className="fill voice"
                                style={{
                                  width: `${Math.round(item.voice_component || 0)}%`,
                                }}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                      {Array.isArray(psychDetail?.skill_gaps) &&
                        psychDetail.skill_gaps.length > 0 && (
                          <div className="career-gaps">
                            <strong>Skill gaps:</strong>
                            <ul>
                              {psychDetail.skill_gaps.map((g) => (
                                <li key={g.name}>
                                  {g.display_name}: {Math.round(g.current)} →{' '}
                                  {Math.round(g.desired)}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                    </div>
                  );
                })}
              </div>
            </section>
          </div>

          <aside className="psych-sidebar">
            <section className="psych-panel">
              <h3>{t('psych.history', 'Profile history')}</h3>
              {history && history.length > 0 ? (
                <ul className="history-list">
                  {history.map((h) => (
                    <li key={h.completed_at}>
                      <span className="history-date">
                        {new Date(h.completed_at).toLocaleDateString()}
                      </span>
                      {h.top_career && (
                        <span className="history-career">{h.top_career}</span>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="history-empty">
                  {t('psych.noHistory', 'You have not completed this assessment yet.')}
                </p>
              )}
            </section>

            <section className="psych-panel">
              <h3>{t('psych.actions', 'Actions')}</h3>
              <p className="estimate">
                {t('psych.estimate', 'Estimated completion time')}: ~{estimatedMinutes} min
              </p>
              <button
                className="btn-retake"
                type="button"
                onClick={() => setShowRetakeConfirm(true)}
              >
                {t('psych.retake', 'Retake assessment')}
              </button>
            </section>
          </aside>

          {showRetakeConfirm && (
            <div className="psych-modal-overlay">
              <div className="psych-modal">
                <h4>{t('psych.retakeTitle', 'Retake assessment?')}</h4>
                <p className="psych-modal-text">
                  {t(
                    'psych.retakeWarning',
                    'Retaking too frequently may affect profile stability tracking. Do you want to continue?'
                  )}
                </p>
                <div className="psych-modal-actions">
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={() => setShowRetakeConfirm(false)}
                  >
                    {t('common.cancel', 'Cancel')}
                  </button>
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={handleRetake}
                  >
                    {t('psych.retakeConfirm', 'Continue Retake')}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="psych-container">
      <div className="psych-header">
        <h2>{t('psych.title', 'Psychological Analysis')}</h2>
        <p>
          {t(
            'psych.intro',
            'Answer a short, science-inspired questionnaire to generate a psychological profile that will be integrated into your career recommendations.'
          )}
        </p>
        <div className="psych-meta">
          <span>
            <i className="fas fa-clock" />{' '}
            {t('psych.estimate', 'Estimated completion time')}: ~{estimatedMinutes} min
          </span>
          <span>
            <i className="fas fa-save" />{' '}
            {t('psych.autoSave', 'Progress is auto-saved on this device.')}
          </span>
        </div>
      </div>

      <form className="psych-form" onSubmit={handleSubmit}>
        <div className="psych-progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>

        <div className="psych-question-card">
          <div className="question-header">
            <span className="question-step">
              Question {currentIndex + 1} of {totalQuestions}
            </span>
            {currentQuestion.scenario && (
              <span className="question-tag">Scenario-based</span>
            )}
          </div>
          <p className="question-text">{currentQuestion.text}</p>

          <div className="likert-scale">
            {LIKERT_OPTIONS.map((opt) => (
              <label key={opt.value} className="likert-option">
                <input
                  type="radio"
                  name={currentQuestion.id}
                  value={opt.value}
                  checked={answers[currentQuestion.id] === opt.value}
                  onChange={() => handleChange(currentQuestion.id, opt.value)}
                />
                <span className="likert-label">{opt.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="psych-navigation">
          <button
            type="button"
            className="btn-nav btn-back"
            onClick={handleBack}
            disabled={currentIndex === 0}
          >
            <i className="fas fa-arrow-left" /> {t('psych.back', 'Back')}
          </button>
          {currentIndex < totalQuestions - 1 ? (
            <button
              type="button"
              className="btn-nav btn-next"
              onClick={handleNext}
            >
              {t('psych.next', 'Next')} <i className="fas fa-arrow-right" />
            </button>
          ) : (
            <button
              type="submit"
              className="btn-nav btn-submit"
              disabled={loading || !allAnswered}
            >
              {loading
                ? t('psych.submitting', 'Calculating profile…')
                : t('psych.submit', 'Complete assessment')}
            </button>
          )}
        </div>

        {error && <p className="error-message">{error}</p>}
      </form>
    </div>
  );
};

export default PsychAssessment;

