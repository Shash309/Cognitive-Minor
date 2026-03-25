import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
const getToken = () => localStorage.getItem('admin_token');
const authHeaders = () => ({
  'Content-Type': 'application/json',
  Authorization: `Bearer ${getToken()}`,
});

const StatCard = ({ icon, label, value, color }) => (
  <div className="admin-stat-card" style={{ '--stat-accent': color }}>
    <div className="admin-stat-icon">{icon}</div>
    <div className="admin-stat-body">
      <span className="admin-stat-value">{value ?? '—'}</span>
      <span className="admin-stat-label">{label}</span>
    </div>
  </div>
);

const AdminHome = () => {
  const [stats, setStats] = useState(null);
  const [recentUsers, setRecentUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const load = async () => {
      try {
        const [statsRes, usersRes] = await Promise.all([
          fetch(`${API}/admin/stats`, { headers: authHeaders() }),
          fetch(`${API}/admin/users`, { headers: authHeaders() }),
        ]);

        if (statsRes.status === 403 || usersRes.status === 403) {
          navigate('/admin/login');
          return;
        }

        const statsData = await statsRes.json();
        const usersData = await usersRes.json();

        setStats(statsData);
        setRecentUsers(Array.isArray(usersData) ? usersData.slice(0, 8) : []);
      } catch {
        console.error('Failed to load admin data');
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
        <p>Loading dashboard…</p>
      </div>
    );
  }

  return (
    <div className="admin-home">
      <div className="admin-page-header">
        <h1>Dashboard Overview</h1>
        <p>Platform metrics and quick insights</p>
      </div>

      {/* Stats Grid */}
      <div className="admin-stats-grid">
        <StatCard icon="👥" label="Total Users" value={stats?.total_users} color="#6366f1" />
        <StatCard icon="🧠" label="Quiz Attempts" value={stats?.total_quiz_attempts} color="#8b5cf6" />
        <StatCard icon="🎙️" label="Voice Analyses" value={stats?.total_voice_analyses} color="#ec4899" />
        <StatCard icon="📊" label="Psych Assessments" value={stats?.total_psych_assessments} color="#10b981" />
        <StatCard icon="🎯" label="Career Recommendations" value={stats?.total_career_recommendations} color="#f59e0b" />
        <StatCard icon="🟢" label="Active Users" value={stats?.active_users} color="#06b6d4" />
      </div>

      {/* Recent Users */}
      <div className="admin-section">
        <div className="admin-section-header">
          <h2>Recent Users</h2>
          <button className="admin-text-btn" onClick={() => navigate('/admin/dashboard/users')}>
            View All →
          </button>
        </div>

        <div className="admin-table-wrapper">
          <table className="admin-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Phone</th>
                <th>Location</th>
                <th>Age</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {recentUsers.length === 0 ? (
                <tr><td colSpan="6" className="admin-empty">No users yet</td></tr>
              ) : (
                recentUsers.map((u) => (
                  <tr key={u.id || u.email}>
                    <td className="admin-cell-name">
                      <div className="admin-user-avatar-sm">{(u.name || '?').charAt(0).toUpperCase()}</div>
                      {u.name}
                    </td>
                    <td>{u.email}</td>
                    <td>{u.phone || '—'}</td>
                    <td>{u.location || '—'}</td>
                    <td>{u.age || '—'}</td>
                    <td>
                      <button
                        className="admin-view-btn"
                        onClick={() => navigate(`/admin/dashboard/user/${u.id}`)}
                      >
                        Details
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AdminHome;
