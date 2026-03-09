import React, { useEffect, useState } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';

const CounselorHome = () => {
    const { user } = useOutletContext() || {};
    const navigate = useNavigate();
    const [students, setStudents] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const load = async () => {
            try {
                const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
                const res = await fetch(`${apiBase}/api/counselor/students`);
                const json = await res.json();
                setStudents(json.students || []);
            } catch {
                // ignore
            } finally {
                setLoading(false);
            }
        };
        load();
        // Refresh every 15 seconds
        const id = setInterval(load, 15000);
        return () => clearInterval(id);
    }, []);

    const getInitial = (name) => (name || 'S').charAt(0).toUpperCase();

    if (loading) {
        return (
            <div className="counselor-empty">
                <p>Loading student requests...</p>
            </div>
        );
    }

    return (
        <div>
            <div className="counselor-header">
                <h2>Counselor Dashboard</h2>
                <p>Welcome back, {user?.name || 'Counselor'}. Monitor student career assessments and provide expert guidance.</p>
            </div>

            {/* Stats */}
            <div className="counselor-stats">
                <div className="stat-card">
                    <div className="stat-value">{students.length}</div>
                    <div className="stat-label">Student Requests</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">
                        {students.filter(s => s.status === 'pending').length}
                    </div>
                    <div className="stat-label">Pending Reviews</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{user?.specialization || '—'}</div>
                    <div className="stat-label">Your Specialization</div>
                </div>
            </div>

            {students.length === 0 ? (
                <div className="counselor-empty">
                    <div className="counselor-empty-icon">📋</div>
                    <h3>No student requests yet</h3>
                    <p>Students who complete their career assessment and request counseling will appear here.</p>
                </div>
            ) : (
                <div className="student-cards-grid">
                    {students.map((student) => (
                        <div
                            key={student.email}
                            className="student-request-card"
                            onClick={() => navigate(`/counselor/student/${encodeURIComponent(student.email)}`)}
                        >
                            <div className="src-header">
                                <div className="src-avatar">{getInitial(student.name)}</div>
                                <div>
                                    <div className="src-name">{student.name}</div>
                                    <div className="src-email">{student.email}</div>
                                </div>
                            </div>

                            <div className="src-career-row">
                                <div>
                                    <div className="src-career-label">Recommended Career</div>
                                    <div className="src-career-value">{student.top_career || '—'}</div>
                                </div>
                                {student.confidence_score != null && (
                                    <div className="src-confidence">
                                        {student.confidence_score}% confidence
                                    </div>
                                )}
                            </div>

                            {student.dominant_traits && student.dominant_traits.length > 0 && (
                                <div className="src-traits">
                                    {student.dominant_traits.map((trait, i) => (
                                        <span key={i} className="src-trait-tag">{trait}</span>
                                    ))}
                                </div>
                            )}

                            <div className="src-status">
                                <span className="src-status-dot" />
                                {student.status === 'pending' ? 'Awaiting Review' : student.status}
                                {student.requested_at && (
                                    <span> · {new Date(student.requested_at).toLocaleDateString()}</span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default CounselorHome;
