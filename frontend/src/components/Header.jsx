import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { changeLanguage } from '../features/i18n';

const formatUsername = (user) => {
  if (!user) return 'User';
  const raw = typeof user === 'string' ? user : user.name || 'User';
  const username = raw.includes('@') ? raw.split('@')[0] : raw;
  return username.charAt(0).toUpperCase() + username.slice(1);
};

const getInitial = (user) => {
  const name = formatUsername(user);
  return name.charAt(0).toUpperCase();
};

const Header = ({ user, onLogout, unreadCount = 0, onNotificationClick }) => {
  const { i18n } = useTranslation();
  const navigate = useNavigate();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [showUserMenu, setShowUserMenu] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Close user menu on outside click
  useEffect(() => {
    const handleClick = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setShowUserMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const handleLangChange = (e) => {
    changeLanguage(e.target.value);
  };

  const dateStr = currentTime.toLocaleDateString(undefined, { weekday: 'short', month: 'long', day: 'numeric' });
  const timeStr = currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <header className="header-shell">
      {/* Left: Date & Time - Very Subtle */}
      <div className="header-left">
        <div className="header-datetime">
          <span className="header-date">{dateStr}</span>
          <span className="header-time-separator">|</span>
          <span className="header-time">{timeStr}</span>
        </div>
      </div>

      {/* Center: (Removed, handled in Home view) */}
      <div className="header-center"></div>

      {/* Right: Controls */}
      <div className="header-right">
        {/* Language Selector */}
        <div className="lang-wrapper">
          <select
            value={i18n.language || 'en'}
            onChange={handleLangChange}
            className="lang-select-minimal"
          >
            <option value="en">English</option>
            <option value="hi">हिंदी</option>
            <option value="ur">اردو</option>
          </select>
        </div>

        {/* Notification Bell */}
        <button className="icon-btn-refined" onClick={onNotificationClick} title="Notifications">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 0 1-3.46 0" />
          </svg>
          {unreadCount > 0 && <span className="notif-dot" />}
        </button>

        {/* User Pill */}
        <div className="user-menu-wrapper" ref={menuRef}>
          <button
            className="user-pill-refined"
            onClick={() => setShowUserMenu(!showUserMenu)}
          >
            <div className="pill-avatar">{getInitial(user)}</div>
            <span className="pill-name">{formatUsername(user)}</span>
          </button>

          {showUserMenu && (
            <div className="user-dropdown-refined">
              <div className="dropdown-header">
                <div className="dropdown-avatar-lg">{getInitial(user)}</div>
                <div className="dropdown-user-details">
                  <p className="dropdown-name">{formatUsername(user)}</p>
                  <p className="dropdown-email">{typeof user === 'object' ? user.email : user}</p>
                </div>
              </div>
              <div className="dropdown-menu-items">
                <button
                  className="dropdown-menu-btn"
                  onClick={() => {
                    navigate('/dashboard/profile');
                    setShowUserMenu(false);
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="3" />
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 8 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 3.6 15a1.65 1.65 0 0 0-1.51-1H2a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 3.6 8a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 8 3.6a1.65 1.65 0 0 0 1-1.51V2a2 2 0 1 1 4 0v.09A1.65 1.65 0 0 0 16 3.6a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 8a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                  </svg>
                  View Profile
                </button>
                <hr className="dropdown-divider" />
                <button className="dropdown-menu-btn" onClick={onLogout}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                    <polyline points="16 17 21 12 16 7" />
                    <line x1="21" y1="12" x2="9" y2="12" />
                  </svg>
                  Sign Out
                </button>
              </div>
            </div>
          )}
        </div>
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
