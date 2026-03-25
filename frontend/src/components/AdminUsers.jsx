import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
const getToken = () => localStorage.getItem('admin_token');
const authHeaders = () => ({
  'Content-Type': 'application/json',
  Authorization: `Bearer ${getToken()}`,
});

const AdminUsers = () => {
  const [users, setUsers] = useState([]);
  const [filtered, setFiltered] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API}/admin/users`, { headers: authHeaders() });
        if (res.status === 403) { navigate('/admin/login'); return; }
        const data = await res.json();
        setUsers(Array.isArray(data) ? data : []);
        setFiltered(Array.isArray(data) ? data : []);
      } catch {
        console.error('Failed to fetch users');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [navigate]);

  useEffect(() => {
    const q = search.toLowerCase().trim();
    if (!q) { setFiltered(users); return; }
    setFiltered(
      users.filter(
        (u) =>
          (u.name || '').toLowerCase().includes(q) ||
          (u.email || '').toLowerCase().includes(q) ||
          (u.location || '').toLowerCase().includes(q) ||
          (u.phone || '').includes(q)
      )
    );
  }, [search, users]);

  const handleExport = async () => {
    try {
      const res = await fetch(`${API}/admin/export/users`, { headers: authHeaders() });
      if (!res.ok) return;
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'users_export.csv';
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      alert('Export failed');
    }
  };

  if (loading) {
    return (
      <div className="admin-page-loader">
        <div className="admin-page-spinner" />
        <p>Loading users…</p>
      </div>
    );
  }

  return (
    <div className="admin-users-page">
      <div className="admin-page-header">
        <div>
          <h1>User Management</h1>
          <p>{users.length} registered user{users.length !== 1 ? 's' : ''}</p>
        </div>
        <button className="admin-export-btn" onClick={handleExport}>
          <i className="fas fa-download"></i> Export CSV
        </button>
      </div>

      {/* Search */}
      <div className="admin-search-bar">
        <i className="fas fa-search"></i>
        <input
          type="text"
          placeholder="Search by name, email, location, or phone…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        {search && (
          <button className="admin-clear-search" onClick={() => setSearch('')}>✕</button>
        )}
      </div>

      {/* Table */}
      <div className="admin-table-wrapper">
        <table className="admin-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Email</th>
              <th>Phone</th>
              <th>Age</th>
              <th>Gender</th>
              <th>Location</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr><td colSpan="8" className="admin-empty">No users match your search</td></tr>
            ) : (
              filtered.map((u) => (
                <tr key={u.id || u.email}>
                  <td className="admin-cell-id">{u.id}</td>
                  <td className="admin-cell-name">
                    <div className="admin-user-avatar-sm">{(u.name || '?').charAt(0).toUpperCase()}</div>
                    {u.name}
                  </td>
                  <td>{u.email}</td>
                  <td>{u.phone || '—'}</td>
                  <td>{u.age || '—'}</td>
                  <td>{u.gender || '—'}</td>
                  <td>{u.location || '—'}</td>
                  <td>
                    <button
                      className="admin-view-btn"
                      onClick={() => navigate(`/admin/dashboard/user/${u.id}`)}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AdminUsers;
