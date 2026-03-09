import React from 'react';
import { NavLink } from 'react-router-dom';

const CounselorSidebar = () => {
    return (
        <>
            <div className="sidebar-header">
                <i className="fas fa-brain sidebar-logo-icon"></i>
                <h3>Career Counselor</h3>
            </div>

            <nav className="sidebar-nav">
                <ul>
                    <li>
                        <NavLink to="/counselor" end className={({ isActive }) => (isActive ? 'active' : '')}>
                            <i className="fas fa-home"></i> Dashboard
                        </NavLink>
                    </li>
                    <li>
                        <NavLink to="/counselor" end className={({ isActive }) => (isActive ? 'active' : '')}>
                            <i className="fas fa-users"></i> Student Requests
                        </NavLink>
                    </li>
                </ul>
            </nav>

            <div className="counselor-sidebar-badge">
                <div className="csb-icon">🧠</div>
                <div className="csb-label">AI + Human</div>
                <div className="csb-sub">Career Intelligence</div>
            </div>
        </>
    );
};

export default CounselorSidebar;
