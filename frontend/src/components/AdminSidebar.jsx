import React from 'react';
import { NavLink } from 'react-router-dom';
import './AdminDashboard.css';

const AdminSidebar = ({ admin, onLogout }) => {
  return (
    <div className="admin-sidebar">
      <div className="admin-sidebar-header">
        <div className="admin-sidebar-logo">🛡️</div>
        <h3>Admin Console</h3>
      </div>

      <nav className="admin-sidebar-nav">
        <ul>
          <li>
            <NavLink to="/admin/dashboard" end className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="fas fa-chart-pie"></i>
              <span>Dashboard</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/admin/dashboard/users" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="fas fa-users"></i>
              <span>Users</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/admin/dashboard/activity" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="fas fa-history"></i>
              <span>Activity</span>
            </NavLink>
          </li>
        </ul>
      </nav>

      <div className="admin-sidebar-footer">
        <div className="admin-sidebar-user">
          <div className="admin-avatar">
            {(admin?.name || 'A').charAt(0).toUpperCase()}
          </div>
          <div className="admin-sidebar-user-info">
            <span className="admin-user-name">{admin?.name || 'Admin'}</span>
            <span className="admin-user-role">Administrator</span>
          </div>
        </div>
        <button className="admin-logout-btn" onClick={onLogout} title="Logout">
          <i className="fas fa-sign-out-alt"></i>
        </button>
      </div>
    </div>
  );
};

export default AdminSidebar;
