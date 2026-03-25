import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
const getToken = () => localStorage.getItem('admin_token');
const authHeaders = () => ({
  'Content-Type': 'application/json',
  Authorization: `Bearer ${getToken()}`,
});

const InfoRow = ({ label, value }) => (
  <div className="admin-detail-row">
    <span className="admin-detail-label">{label}</span>
    <span className="admin-detail-value">{value || '—'}</span>
  </div>
);

const AdminUserDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API}/admin/user/${id}`, { headers: authHeaders() });
        if (res.status === 403) { navigate('/admin/login'); return; }
        if (!res.ok) { setData(null); setLoading(false); return; }
        setData(await res.json());
      } catch {
        console.error('Error loading user detail');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [id, navigate]);

  if (loading) {
    return (
      <div className="admin-page-loader">
        <div className="admin-page-spinner" />
        <p>Loading user details…</p>
      </div>
    );
  }

  if (!data || !data.user) {
    return (
      <div className="admin-detail-empty">
        <h2>User not found</h2>
        <button className="admin-text-btn" onClick={() => navigate('/admin/dashboard/users')}>← Back to Users</button>
      </div>
    );
  }

  const u = data.user;
  const quizHistory = data.quiz_history || [];
  const voiceHistory = data.voice_history || [];
  const psych = data.psych_profile;
  const career = data.career_fused;
  const progress = data.progress;

  return (
    <div className="admin-user-detail-page">
      <button className="admin-back-btn" onClick={() => navigate('/admin/dashboard/users')}>
        ← Back to Users
      </button>

      {/* User Header */}
      <div className="admin-detail-header">
        <div className="admin-detail-avatar">{(u.name || '?').charAt(0).toUpperCase()}</div>
        <div>
          <h1>{u.name}</h1>
          <p>{u.email}</p>
        </div>
      </div>

      {/* Basic Info */}
      <div className="admin-detail-section">
        <h2>Basic Information</h2>
        <div className="admin-detail-grid">
          <InfoRow label="ID" value={u.id} />
          <InfoRow label="Phone" value={u.phone} />
          <InfoRow label="Age" value={u.age} />
          <InfoRow label="Gender" value={u.gender} />
          <InfoRow label="Location" value={u.location} />
          <InfoRow label="Registered" value={u.created_at ? new Date(u.created_at).toLocaleDateString() : '—'} />
        </div>
      </div>

      {/* Progress */}
      {progress && (
        <div className="admin-detail-section">
          <h2>Evaluation Progress</h2>
          <div className="admin-progress-chips">
            <span className={`admin-chip ${progress.psych_completed ? 'done' : ''}`}>
              {progress.psych_completed ? '✅' : '⬜'} Psychology
            </span>
            <span className={`admin-chip ${progress.voice_completed ? 'done' : ''}`}>
              {progress.voice_completed ? '✅' : '⬜'} Voice Insight
            </span>
            <span className={`admin-chip ${progress.quiz_completed ? 'done' : ''}`}>
              {progress.quiz_completed ? '✅' : '⬜'} Career Quiz
            </span>
          </div>
        </div>
      )}

      {/* Psych Profile */}
      {psych && typeof psych === 'object' && (
        <div className="admin-detail-section">
          <h2>Psychological Profile</h2>
          <div className="admin-detail-grid">
            {Object.entries(psych)
              .filter(([, v]) => typeof v === 'number')
              .sort(([, a], [, b]) => b - a)
              .map(([k, v]) => (
                <div className="admin-trait-bar" key={k}>
                  <div className="admin-trait-label">{k.replace(/_/g, ' ')}</div>
                  <div className="admin-trait-track">
                    <div className="admin-trait-fill" style={{ width: `${Math.min(v, 100)}%` }} />
                  </div>
                  <span className="admin-trait-val">{typeof v === 'number' ? v.toFixed(0) : v}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Career Fused Results */}
      {career && (
        <div className="admin-detail-section">
          <h2>Career Recommendations</h2>
          {career.career_rankings ? (
            <div className="admin-career-rankings">
              {career.career_rankings.map((c, i) => (
                <div className="admin-career-rank" key={c.career || i}>
                  <span className="admin-rank-num">#{i + 1}</span>
                  <span className="admin-rank-career">{c.career}</span>
                  <span className="admin-rank-score">
                    {typeof c.final_score === 'number' ? c.final_score.toFixed(1) : c.final_score}%
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="admin-empty-sm">No career rankings available</p>
          )}
        </div>
      )}

      {/* Quiz History */}
      <div className="admin-detail-section">
        <h2>Quiz History ({quizHistory.length})</h2>
        {quizHistory.length === 0 ? (
          <p className="admin-empty-sm">No quiz attempts</p>
        ) : (
          <div className="admin-history-list">
            {quizHistory.slice(0, 5).map((q, i) => (
              <div className="admin-history-item" key={i}>
                <span className="admin-history-ts">{q.timestamp ? new Date(q.timestamp).toLocaleString() : 'Unknown'}</span>
                <span className="admin-history-detail">
                  {q.predicted_career || q.top_career || 'N/A'}
                  {q.confidence ? ` (${q.confidence}% confidence)` : ''}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Voice History */}
      <div className="admin-detail-section">
        <h2>Voice Analysis History ({voiceHistory.length})</h2>
        {voiceHistory.length === 0 ? (
          <p className="admin-empty-sm">No voice analyses</p>
        ) : (
          <div className="admin-history-list">
            {voiceHistory.slice(0, 5).map((v, i) => (
              <div className="admin-history-item" key={i}>
                <span className="admin-history-ts">{v.timestamp ? new Date(v.timestamp).toLocaleString() : 'Unknown'}</span>
                <span className="admin-history-detail">
                  {v.transcript ? `"${v.transcript.substring(0, 80)}${v.transcript.length > 80 ? '…' : ''}"` : 'No transcript'}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default AdminUserDetail;
