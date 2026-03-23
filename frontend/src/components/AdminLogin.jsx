import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Admin.css'; // We'll create a minimal Admin.css later or reuse generic one

const AdminLogin = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const navigate = useNavigate();

  const handleAdminLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
      const res = await fetch(`${apiBase}/api/admin/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Login failed');
      }

      // Instead of storing in career_app_user, let's keep admin isolated slightly if possible, 
      // or we can reuse handleLogin if the wrapper supports role check.
      // The App.jsx handleLogin already supports arbitrary user data.
      onLogin({ ...data.user, token: data.token });
      navigate('/admin/dashboard');

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="admin-login-container">
      <div className="admin-login-panel">
        <div className="admin-header">
          <h2>Admin Secure Login</h2>
          <p>Restricted area. Authorized personnel only.</p>
        </div>
        <form onSubmit={handleAdminLogin}>
          {error && <div className="admin-error">{error}</div>}
          <div className="admin-input-group">
            <label>Admin Email</label>
            <input 
              type="email" 
              required 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="admin@cognitive.com"
            />
          </div>
          <div className="admin-input-group">
            <label>Password</label>
            <input 
              type="password" 
              required 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </div>
          <button type="submit" disabled={isLoading} className="admin-btn">
            {isLoading ? 'Authenticating...' : 'Secure Login'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default AdminLogin;
