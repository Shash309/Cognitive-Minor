import React, { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import { Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import Header from './Header';
import Sidebar from './Sidebar';
import './Dashboard.css';

const Notifications = ({ notifications, onClear }) => (
  <div className="notifications-panel">
    <div className="notifications-header">
      <h3>Notifications</h3>
      <button onClick={onClear}>Mark all as read</button>
    </div>
    <div className="notifications-list">
      {notifications.length > 0 ? (
        notifications.map((item, index) => (
          <div key={index} className="notification-item">
            <i className="fas fa-calendar-alt"></i>
            <div className="notification-content">
              <p><strong>{item.title}</strong></p>
              <span>Deadline: {new Date(item.date).toLocaleDateString()}</span>
            </div>
          </div>
        ))
      ) : (
        <p className="no-notifications">No new notifications</p>
      )}
    </div>
  </div>
);
Notifications.propTypes = { notifications: PropTypes.array.isRequired, onClear: PropTypes.func.isRequired };

const Dashboard = ({ user, onLogout }) => {
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showNotifications, setShowNotifications] = useState(false);
  const location = useLocation();

  const fetchNotifications = useCallback(() => {
    const eventsData = [
      { title: 'JEE Main 2026 (Session 1) Application', date: '2025-11-22T21:00:00' },
      { title: 'GATE 2026 Registration (with late fee)', date: '2025-10-11T23:59:59' },
      { title: "Prime Minister's Scholarship Scheme (PMSS)", date: '2025-11-30T23:59:59' },
      { title: 'NEET UG 2026 Application', date: '2026-03-07T21:00:00' },
    ];
    const upcomingEvents = eventsData.filter(event => new Date(event.date) > new Date());
    setNotifications(upcomingEvents);
    setUnreadCount(upcomingEvents.length);
  }, []);

  useEffect(() => { fetchNotifications(); }, [fetchNotifications]);

  const handleClearNotifications = () => { setUnreadCount(0); };

  return (
    <div className="app-shell">
      {/* Fixed Sidebar */}
      <div className="sidebar-shell">
        <Sidebar />
      </div>

      {/* Main Content Area */}
      <div className="main-shell">
        <Header
          user={user}
          onLogout={onLogout}
          unreadCount={unreadCount}
          onNotificationClick={() => setShowNotifications(!showNotifications)}
        />

        {/* Absolute Notification Panel */}
        {showNotifications && (
          <Notifications notifications={notifications} onClear={handleClearNotifications} />
        )}

        <div className="content-shell">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              style={{ minHeight: '100%', padding: '2.5rem' }}
            >
              <Outlet context={{ user }} />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>

  );
};

Dashboard.propTypes = { user: PropTypes.object, onLogout: PropTypes.func.isRequired };

export default Dashboard;

