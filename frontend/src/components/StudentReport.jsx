import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

const toTitleCase = (value) => {
    if (!value || typeof value !== 'string') return '';
    return value
        .replace(/_/g, ' ')
        .split(' ')
        .filter(Boolean)
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
};

const StudentReport = () => {
    const { studentEmail } = useParams();
    const navigate = useNavigate();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const load = async () => {
            if (!studentEmail) return;
            try {
                const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
                const res = await fetch(
                    `${apiBase}/api/counselor/student-report?student_email=${encodeURIComponent(studentEmail)}`
                );
                const json = await res.json();
                if (!res.ok) throw new Error(json.error || 'Unable to load report.');
                setData(json);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        load();
    }, [studentEmail]);

    if (loading) return <p style={{ color: 'var(--text-muted)' }}>Loading student report...</p>;
    if (error) return <p style={{ color: 'var(--error)' }}>{error}</p>;
    if (!data) return null;

    const { student, psych_profile, dominant_traits, decision_style, voice_insight, quiz, career_rankings } = data;
    const getInitial = () => (student?.name || 'S').charAt(0).toUpperCase();

    // Get main traits for the radar display
    const traitEntries = Object.entries(psych_profile || {})
        .filter(([, v]) => typeof v === 'number')
        .sort((a, b) => b[1] - a[1]);

    const sentimentLabel = (s) => {
        if (s == null) return '—';
        if (s >= 0.7) return 'Positive';
        if (s >= 0.4) return 'Neutral';
        return 'Negative';
    };

    return (
        <div className="student-report">
            <button className="sr-back-btn" onClick={() => navigate('/counselor')}>
                <i className="fas fa-arrow-left" /> Back to Dashboard
            </button>

            <div className="sr-student-header">
                <div className="sr-student-avatar">{getInitial()}</div>
                <div className="sr-student-info">
                    <h2>{student?.name || studentEmail}</h2>
                    <p>{student?.email} · AI Career Intelligence Report</p>
                </div>
            </div>

            {/* Psychological Profile */}
            <div className="sr-section">
                <div className="sr-section-title">
                    <i className="fas fa-brain" /> Psychological Profile
                </div>
                {decision_style && (
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '14px' }}>
                        Decision Style: <strong style={{ color: 'var(--text-main)' }}>{decision_style}</strong>
                    </p>
                )}
                <div className="sr-radar">
                    {traitEntries.slice(0, 9).map(([trait, score]) => (
                        <div key={trait} className="sr-trait-bar">
                            <div className="sr-trait-name">{toTitleCase(trait)}</div>
                            <div className="sr-trait-fill-bg">
                                <div className="sr-trait-fill" style={{ width: `${Math.round(score)}%` }} />
                            </div>
                            <div className="sr-trait-score">{Math.round(score)}%</div>
                        </div>
                    ))}
                </div>
                {dominant_traits && dominant_traits.length > 0 && (
                    <div style={{ marginTop: '14px' }}>
                        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Dominant Traits: </span>
                        {dominant_traits.slice(0, 4).map((t, i) => (
                            <span key={i} className="src-trait-tag" style={{ marginRight: '6px' }}>
                                {t.display_name || t.name || t}
                            </span>
                        ))}
                    </div>
                )}
            </div>

            {/* Voice Insight */}
            <div className="sr-section">
                <div className="sr-section-title">
                    <i className="fas fa-microphone-alt" /> Voice Insight
                </div>
                <div className="sr-voice-grid">
                    <div className="sr-voice-stat">
                        <div className="stat-value">
                            {voice_insight?.confidence_score != null ? Math.round(voice_insight.confidence_score) : '—'}
                        </div>
                        <div className="stat-label">Confidence</div>
                    </div>
                    <div className="sr-voice-stat">
                        <div className="stat-value">
                            {voice_insight?.motivation_score != null ? Math.round(voice_insight.motivation_score) : '—'}
                        </div>
                        <div className="stat-label">Motivation</div>
                    </div>
                    <div className="sr-voice-stat">
                        <div className="stat-value">{sentimentLabel(voice_insight?.sentiment)}</div>
                        <div className="stat-label">Emotional Tone</div>
                    </div>
                </div>
                {voice_insight?.transcript && (
                    <div className="sr-voice-transcript">
                        "{voice_insight.transcript.slice(0, 300)}{voice_insight.transcript.length > 300 ? '...' : ''}"
                    </div>
                )}
            </div>

            {/* Career Quiz Results */}
            <div className="sr-section">
                <div className="sr-section-title">
                    <i className="fas fa-chart-bar" /> Career Rankings
                </div>
                {quiz?.top_career && (
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '14px' }}>
                        Quiz Top Pick: <strong style={{ color: 'var(--primary)' }}>{quiz.top_career}</strong>
                        {quiz.stream && <span> · Stream: {quiz.stream}</span>}
                    </p>
                )}
                <div className="sr-career-list">
                    {(career_rankings || []).map((item, idx) => (
                        <div key={item.career} className="sr-career-item">
                            <div className="sr-career-rank">{idx + 1}</div>
                            <div className="sr-career-name">{item.career}</div>
                            <div className="sr-career-score">{Math.round(item.score)}%</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* AI Insight */}
            {data.explanation && (
                <div className="sr-section">
                    <div className="sr-section-title">
                        <i className="fas fa-robot" /> AI Insight
                    </div>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                        {data.explanation}
                    </p>
                </div>
            )}

            <button
                className="sr-consult-btn"
                onClick={() => navigate(`/counselor/chat/${encodeURIComponent(studentEmail)}`)}
            >
                <i className="fas fa-comment-dots" /> Start Consultation
            </button>
        </div>
    );
};

export default StudentReport;
