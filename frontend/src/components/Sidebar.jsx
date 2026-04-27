import React from 'react';
import { useTranslation } from 'react-i18next';
import { NavLink } from 'react-router-dom';

const Sidebar = ({ progress }) => {
  const { t } = useTranslation();

  const psychDone = progress?.psych_completed;
  const voiceDone = progress?.voice_completed;

  const lockVoice = !psychDone;
  const lockQuiz = !psychDone || !voiceDone;

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
              <i className="fas fa-home"></i> <span className="sidebar-text">{t('common.dashboard', 'Dashboard')}</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/colleges" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-university"></i> <span className="sidebar-text">{t('common.exploreColleges', 'Explore Colleges')}</span>
            </NavLink>
          </li>
          <li>
            {lockQuiz ? (
              <div className="sidebar-item-locked" title="Complete previous steps to unlock the AI Career Quiz.">
                <i className="fas fa-tasks"></i> <span className="sidebar-text">{t('common.aiCareerQuiz', 'AI Career Quiz')}</span>{' '}
                <i className="fas fa-lock lock-icon"></i>
              </div>
            ) : (
              <NavLink to="/dashboard/quiz" className={({ isActive }) => (isActive ? 'active' : '')}>
                <i className="fas fa-tasks"></i> <span className="sidebar-text">{t('common.aiCareerQuiz', 'AI Career Quiz')}</span>
              </NavLink>
            )}
          </li>
          <li>
            <NavLink to="/dashboard/skills" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-lightbulb"></i> <span className="sidebar-text">{t('common.skillBuilder', 'Skill Builder')}</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/visualizer" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-project-diagram"></i> <span className="sidebar-text">{t('common.careerVisualizer', 'Career Visualizer')}</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/timeline" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-calendar-alt"></i> <span className="sidebar-text">{t('common.timelineTracker', 'Timeline Tracker')}</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard/psychology" className={({ isActive }) => (isActive ? 'active' : '')}>
              <i className="fas fa-brain"></i> <span className="sidebar-text">{t('common.psychologicalAnalysis', 'Psychological Analysis')}</span>
            </NavLink>
          </li>
          <li>
            {lockVoice ? (
              <div className="sidebar-item-locked" title="Complete the Psychological Assessment to unlock Voice Insight.">
                <i className="fas fa-microphone-alt"></i> <span className="sidebar-text">{t('common.voiceInsight', 'Voice Insight')}</span>{' '}
                <i className="fas fa-lock lock-icon"></i>
              </div>
            ) : (
              <NavLink to="/dashboard/voice" className={({ isActive }) => (isActive ? 'active' : '')}>
                <i className="fas fa-microphone-alt"></i> <span className="sidebar-text">{t('common.voiceInsight', 'Voice Insight')}</span>
              </NavLink>
            )}
          </li>
        </ul>
      </nav>
    </>

  );
};

export default Sidebar;

