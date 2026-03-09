import React, { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import { Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import Header from './Header';
import CounselorSidebar from './CounselorSidebar';
import './CounselorDashboard.css';

const MotionDiv = motion.div;

const CounselorDashboard = ({ user, onLogout }) => {
    const location = useLocation();

    return (
        <div className="app-shell">
            <div className="sidebar-shell">
                <CounselorSidebar />
            </div>
            <div className="main-shell">
                <Header
                    user={user}
                    onLogout={onLogout}
                    unreadCount={0}
                    onNotificationClick={() => { }}
                />
                <div className="content-shell">
                    <AnimatePresence mode="wait">
                        <MotionDiv
                            key={location.pathname}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            transition={{ duration: 0.25, ease: "easeOut" }}
                            style={{ minHeight: '100%', padding: '2.5rem' }}
                        >
                            <Outlet context={{ user }} />
                        </MotionDiv>
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
};

CounselorDashboard.propTypes = { user: PropTypes.object, onLogout: PropTypes.func.isRequired };

export default CounselorDashboard;
