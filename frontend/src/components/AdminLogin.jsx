import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './AdminLogin.css';

const AdminLogin = ({ onAdminLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
    try {
      const res = await fetch(`${apiBase}/admin/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.error || 'Login failed');
        setLoading(false);
        return;
      }

      // Store admin session separately
      localStorage.setItem('admin_token', data.token);
      localStorage.setItem('admin_user', JSON.stringify(data.admin));
      onAdminLogin(data.admin, data.token);
      navigate('/admin/dashboard');
    } catch {
      setError('Server error. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="admin-login-container">
      {/* Ambient glow orbs */}
      <div className="admin-orb admin-orb-1" />
      <div className="admin-orb admin-orb-2" />

      <div className="admin-login-card">
        <div className="admin-login-header">
          <div className="admin-shield-icon">🛡️</div>
          <h1>Admin Portal</h1>
          <p>Cognitive Career Decision Intelligence — Management Console</p>
        </div>

        <form onSubmit={handleSubmit} className="admin-login-form">
          {error && <div className="admin-error-banner">{error}</div>}

          <div className="admin-input-group">
            <label htmlFor="admin-email">Email</label>
            <input
              id="admin-email"
              type="email"
              placeholder="admin@cognitive.ai"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoFocus
            />
          </div>

          <div className="admin-input-group">
            <label htmlFor="admin-password">Password</label>
            <input
              id="admin-password"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button type="submit" className="admin-login-btn" disabled={loading}>
            {loading ? (
              <span className="admin-spinner" />
            ) : (
              'Access Dashboard'
            )}
          </button>
        </form>

        <div className="admin-login-footer">
          <span>Restricted access — authorized personnel only</span>
        </div>
      </div>
    </div>
  );
};

export default AdminLogin;
