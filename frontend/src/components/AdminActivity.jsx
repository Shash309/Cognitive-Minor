import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
const getToken = () => localStorage.getItem('admin_token');
const authHeaders = () => ({
  'Content-Type': 'application/json',
  Authorization: `Bearer ${getToken()}`,
});

const AdminActivity = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API}/admin/activity`, { headers: authHeaders() });
        if (res.status === 403) { navigate('/admin/login'); return; }
        const data = await res.json();
        setLogs(Array.isArray(data) ? data : []);
      } catch {
        console.error('Failed to load activity');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [navigate]);

  if (loading) {
    return (
      <div className="admin-page-loader">
        <div className="admin-page-spinner" />
        <p>Loading activity…</p>
      </div>
    );
  }

  const typeIcon = (type) => {
    switch (type) {
      case 'registration': return '🆕';
      case 'login': return '🔑';
      case 'error': return '⚠️';
      default: return '📌';
    }
  };

  return (
    <div className="admin-activity-page">
      <div className="admin-page-header">
        <h1>System Activity</h1>
        <p>{logs.length} recorded event{logs.length !== 1 ? 's' : ''}</p>
      </div>

      <div className="admin-timeline">
        {logs.length === 0 ? (
          <div className="admin-empty">No activity recorded yet</div>
        ) : (
          logs.map((log, i) => (
            <div className="admin-timeline-item" key={i}>
              <div className="admin-timeline-dot">{typeIcon(log.type)}</div>
              <div className="admin-timeline-content">
                <div className="admin-timeline-title">
                  <span className="admin-timeline-type">{log.type}</span>
                  <span className="admin-timeline-user">{log.name || log.email}</span>
                </div>
                <div className="admin-timeline-meta">
                  <span>{log.email}</span>
                  <span className="admin-timeline-time">
                    {log.timestamp && log.timestamp !== 'Unknown'
                      ? new Date(log.timestamp).toLocaleString()
                      : 'Unknown time'}
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AdminActivity;
