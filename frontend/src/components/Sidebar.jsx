import React from 'react';
import { useTranslation } from 'react-i18next';
import { NavLink } from 'react-router-dom';

const Sidebar = () => {
  const { t } = useTranslation();

  return (
    <>
      <div className="sidebar-header">
        <i className="fas fa-compass sidebar-logo-icon"></i>
        <h3>{t('common.appName', 'Career Explorer')}</h3>
      </div>

      <nav className="sidebar-nav">
        <ul>
          <li>
            <NavLink to="/dashboard" end className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-home"></i> {t('common.dashboard', 'Dashboard')}
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/colleges" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-university"></i> {t('common.exploreColleges', 'Explore Colleges')}
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/quiz" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-tasks"></i> {t('common.aiCareerQuiz', 'AI Career Quiz')}
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/skills" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-lightbulb"></i> {t('common.skillBuilder', 'Skill Builder')}
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/visualizer" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-project-diagram"></i> {t('common.careerVisualizer', 'Career Visualizer')}
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/timeline" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-calendar-alt"></i> {t('common.timelineTracker', 'Timeline Tracker')}
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/psychology" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-brain"></i> {t('common.psychologicalAnalysis', 'Psychological Analysis')}
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/voice" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-microphone-alt"></i> {t('common.voiceInsight', 'Voice Insight')}
            </NavLink>
          </li>
        </ul>
      </nav>
    </>

  );
};

export default Sidebar;

