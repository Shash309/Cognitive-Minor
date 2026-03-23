import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Admin.css';

const AdminDashboard = ({ user, onLogout }) => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [activity, setActivity] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedUser, setSelectedUser] = useState(null);
  const [userDetails, setUserDetails] = useState(null);

  const navigate = useNavigate();
  const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

  useEffect(() => {
    if (!user || user.role !== 'admin' || !user.token) {
      navigate('/admin/login');
      return;
    }
    fetchData();
  }, [user]);

  const fetchWithAuth = async (url) => {
    const res = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${user.token}`
      }
    });
    if (!res.ok) {
      // If unauthorized, logout
      if (res.status === 403 || res.status === 401) {
        onLogout();
        navigate('/admin/login');
      }
      throw new Error('Request failed');
    }
    return res.json();
  };

  const fetchData = async () => {
    setLoading(true);
    try {
      const [statsData, usersData, activityData] = await Promise.all([
        fetchWithAuth(`${apiBase}/api/admin/stats`),
        fetchWithAuth(`${apiBase}/api/admin/users`),
        fetchWithAuth(`${apiBase}/api/admin/activity`)
      ]);
      setStats(statsData);
      setUsers(usersData);
      setActivity(activityData);
    } catch (error) {
      console.error('Error fetching admin data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchUserDetails = async (emailOrId) => {
    try {
      const details = await fetchWithAuth(`${apiBase}/api/admin/user/${encodeURIComponent(emailOrId)}`);
      setUserDetails(details);
    } catch (error) {
      console.error('Error fetching user details:', error);
    }
  };

  const downloadCSV = () => {
    if (!users.length) return;
    const headers = ['ID', 'Name', 'Email', 'Phone', 'Age', 'Gender', 'Location', 'Created At', 'Last Login'];
    const csvContent = [
      headers.join(','),
      ...users.map(u => 
        [u.id, `"${u.name}"`, u.email, u.phone, u.age, u.gender, `"${u.location}"`, u.created_at, u.last_login].join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", "exported_users.csv");
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (loading) {
    return <div className="admin-loading">Loading secure dashboard...</div>;
  }

  const filteredUsers = users.filter((u) => {
    const search = searchTerm.toLowerCase();
    return (u.name?.toLowerCase().includes(search) || u.email?.toLowerCase().includes(search));
  });

  return (
    <div className="admin-layout">
      {/* Sidebar */}
      <aside className="admin-sidebar">
        <div className="admin-sidebar-header">
          <h2>Cognitive Admin</h2>
          <span className="admin-badge">Protected</span>
        </div>
        <nav className="admin-nav">
          <button className={activeTab === 'dashboard' ? 'active' : ''} onClick={() => setActiveTab('dashboard')}>📈 Dashboard</button>
          <button className={activeTab === 'users' ? 'active' : ''} onClick={() => { setActiveTab('users'); setSelectedUser(null); }}>👤 Users</button>
          <button className={activeTab === 'activity' ? 'active' : ''} onClick={() => setActiveTab('activity')}>⚙️ Activity</button>
        </nav>
        <div className="admin-sidebar-footer">
          <button className="admin-logout-btn" onClick={() => { onLogout(); navigate('/admin/login'); }}>Security Logout</button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="admin-main">
        <header className="admin-main-header">
          <h1>{activeTab === 'dashboard' ? 'Platform Overview' : activeTab === 'users' ? 'User Monitoring' : 'System Activity'}</h1>
        </header>

        <div className="admin-content-area">
          {activeTab === 'dashboard' && stats && (
            <div className="admin-grid-stats">
              <div className="admin-stat-card">
                <h3>Total Users</h3>
                <div className="admin-stat-value">{stats.total_users}</div>
              </div>
              <div className="admin-stat-card">
                <h3>Active Logins</h3>
                <div className="admin-stat-value">{stats.active_users}</div>
              </div>
              <div className="admin-stat-card">
                <h3>Quiz Attempts</h3>
                <div className="admin-stat-value">{stats.total_quiz_attempts}</div>
              </div>
              <div className="admin-stat-card">
                <h3>Voice Analyses</h3>
                <div className="admin-stat-value">{stats.total_voice_analyses}</div>
              </div>
            </div>
          )}

          {activeTab === 'users' && !selectedUser && (
            <div className="admin-panel">
              <div className="admin-panel-actions">
                <input 
                  type="text" 
                  className="admin-search-input" 
                  placeholder="Search by name or email..." 
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                />
                <button onClick={downloadCSV} className="admin-btn admin-btn-export">Export CSV</button>
              </div>
              
              <div className="admin-table-container">
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Name</th>
                      <th>Email</th>
                      <th>Age</th>
                      <th>Location</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredUsers.map(u => (
                      <tr key={u.id}>
                        <td>{u.id}</td>
                        <td>{u.name}</td>
                        <td>{u.email}</td>
                        <td>{u.age}</td>
                        <td>{u.location}</td>
                        <td>
                          <button 
                            className="admin-btn-small" 
                            onClick={() => {
                              setSelectedUser(u);
                              fetchUserDetails(u.email);
                            }}
                          >
                            View Insights
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {filteredUsers.length === 0 && <p className="admin-empty">No users found.</p>}
              </div>
            </div>
          )}

          {activeTab === 'users' && selectedUser && (
            <div className="admin-panel user-detail-panel">
              <button className="admin-back-btn" onClick={() => setSelectedUser(null)}>← Back to Users</button>
              <h2>{selectedUser.name} <span className="admin-subtext">({selectedUser.email})</span></h2>
              
              {!userDetails ? (
                <p>Loading deep insights...</p>
              ) : (
                <div className="admin-user-details-grid">
                  <div className="admin-detail-card">
                    <h3>👤 Basic Info</h3>
                    <p><strong>Phone:</strong> {userDetails.user.phone}</p>
                    <p><strong>Gender:</strong> {userDetails.user.gender}</p>
                    <p><strong>Age:</strong> {userDetails.user.age}</p>
                    <p><strong>Location:</strong> {userDetails.user.location}</p>
                    <p><strong>Joined:</strong> {new Date(userDetails.user.created_at).toLocaleString()}</p>
                  </div>
                  
                  <div className="admin-detail-card">
                    <h3>🧠 AI Quiz Status</h3>
                    <p>
                      {userDetails.quiz_history?.length 
                        ? `Attempts: ${userDetails.quiz_history.length}` 
                        : 'No quiz taken yet.'}
                    </p>
                    {userDetails.quiz_history?.length > 0 && userDetails.quiz_history[0].quiz_scores && (
                      <div className="admin-mini-list">
                        <strong>Latest Scores:</strong>
                        {Object.entries(userDetails.quiz_history[0].quiz_scores).slice(0,3).map(([k,v]) => (
                          <div key={k}>{k}: {v}%</div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="admin-detail-card">
                    <h3>🎙️ Voice Analysis Status</h3>
                    <p>
                      {userDetails.voice_history?.length 
                        ? `Analyses: ${userDetails.voice_history.length}` 
                        : 'No voice analysis yet.'}
                    </p>
                    {userDetails.voice_history?.length > 0 && (
                      <div className="admin-mini-list">
                        <strong>Latest Transcript:</strong>
                        <p className="admin-transcript-snippet">"{userDetails.voice_history[0].transcript.substring(0, 50)}..."</p>
                      </div>
                    )}
                  </div>

                  <div className="admin-detail-card">
                    <h3>🏆 Career Recommendations</h3>
                    {userDetails.career_recommendations?.career_rankings ? (
                       <div className="admin-mini-list">
                       {userDetails.career_recommendations.career_rankings.slice(0,3).map((cr, idx) => (
                         <div key={idx}>{idx + 1}. {cr.career} - {cr.final_score.toFixed(1)}</div>
                       ))}
                     </div>
                    ) : (
                      <p>Fusion complete: None yet.</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'activity' && (
             <div className="admin-panel">
               <div className="admin-table-container">
                 <table className="admin-table">
                   <thead>
                     <tr>
                       <th>Time</th>
                       <th>Type</th>
                       <th>User</th>
                       <th>Email</th>
                     </tr>
                   </thead>
                   <tbody>
                     {activity.map(a => (
                       <tr key={a.id}>
                         <td>{new Date(a.timestamp).toLocaleString()}</td>
                         <td><span className={`admin-tag admin-tag-${a.type}`}>{a.type.toUpperCase()}</span></td>
                         <td>{a.user}</td>
                         <td>{a.email}</td>
                       </tr>
                     ))}
                   </tbody>
                 </table>
                 {activity.length === 0 && <p className="admin-empty">No activity found.</p>}
               </div>
             </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default AdminDashboard;
