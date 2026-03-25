import React from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import AdminSidebar from './AdminSidebar';
import './AdminDashboard.css';

const AdminDashboard = ({ admin, onLogout }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem('admin_token');
    localStorage.removeItem('admin_user');
    onLogout();
    navigate('/admin/login');
  };

  return (
    <div className="admin-shell">
      <AdminSidebar admin={admin} onLogout={handleLogout} />
      <div className="admin-main">
        <Outlet context={{ admin }} />
      </div>
    </div>
  );
};

export default AdminDashboard;
