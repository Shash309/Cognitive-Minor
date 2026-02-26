// frontend/src/components/Header.jsx
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { useTranslation } from 'react-i18next';
import { changeLanguage } from '../features/i18n';

const formatUsername = (user) => {
  if (!user) return 'User';
  const raw = typeof user === 'string' ? user : user.name || 'User';
  const username = raw.includes('@') ? raw.split('@')[0] : raw;
  return username.charAt(0).toUpperCase() + username.slice(1);
};

const Header = ({ user, onLogout, unreadCount = 0, onNotificationClick }) => {
  const { t, i18n } = useTranslation();
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const handleLangChange = (e) => {
    changeLanguage(e.target.value);
  };

  return (
    <header className="header-shell">
      <div className="header-left">
        <span className="header-date">
          {currentTime.toLocaleDateString(undefined, { weekday: 'short', month: 'long', day: 'numeric' })}
        </span>
        <span className="header-time">
          {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>

      <div className="header-right">
        <select
          value={i18n.language || 'en'}
          onChange={handleLangChange}
          style={{
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid var(--glass-border)',
            color: 'var(--text-main)',
            padding: '6px 10px',
            borderRadius: '8px',
            outline: 'none',
            cursor: 'pointer'
          }}
        >
          <option value="en" style={{ color: '#000' }}>English</option>
          <option value="hi" style={{ color: '#000' }}>हिंदी</option>
          <option value="ur" style={{ color: '#000' }}>اردو</option>
        </select>

        <button className="btn-icon" onClick={onNotificationClick}>
          <i className="fas fa-bell"></i>
          {unreadCount > 0 && <span className="notification-badge">{unreadCount}</span>}
        </button>

        <div className="user-profile">
          <div className="avatar-circle">
            {formatUsername(user).charAt(0)}
          </div>
          <span className="user-name">{formatUsername(user)}</span>
        </div>

        <button onClick={onLogout} className="btn-icon" title="Logout" style={{ borderColor: 'var(--error)', color: 'var(--error)' }}>
          <i className="fas fa-sign-out-alt"></i>
        </button>
      </div>
    </header>


  );
};

Header.propTypes = {
  user: PropTypes.oneOfType([PropTypes.string, PropTypes.object]),
  onLogout: PropTypes.func.isRequired,
  unreadCount: PropTypes.number,
  onNotificationClick: PropTypes.func,
};

export default Header;
