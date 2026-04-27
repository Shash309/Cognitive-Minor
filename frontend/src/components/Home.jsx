import React, { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import PropTypes from 'prop-types';
import { motion } from 'framer-motion';
import { useOutletContext } from 'react-router-dom';

const MotionDiv = motion.div;

const Home = ({ user }) => {
    const { t } = useTranslation();
    const { progress } = useOutletContext() || {};

    const psychDone = Boolean(progress?.psych_completed);
    const voiceDone = Boolean(progress?.voice_completed);
    const quizDone = Boolean(progress?.quiz_completed);

    const progressPct = useMemo(() => {
        const total = 3;
        const completed = [psychDone, voiceDone, quizDone].filter(Boolean).length;
        return Math.round((completed / total) * 100);
    }, [psychDone, voiceDone, quizDone]);

    const features = [
        {
            key: 'colleges',
            path: '/dashboard/colleges',
            icon: 'fas fa-university',
            title: t('common.exploreColleges'),
            description: t('dashboard.exploreCollegesDesc', 'Search and filter colleges across India by state and rank.'),
        },
        {
            key: 'quiz',
            path: '/dashboard/quiz',
            icon: 'fas fa-tasks',
            title: t('common.aiCareerQuiz'),
            description: t('dashboard.aiCareerQuizDesc', 'Answer questions to get a personalized career recommendation.'),
        },
        {
            key: 'skills',
            path: '/dashboard/skills',
            icon: 'fas fa-lightbulb',
            title: t('common.skillBuilder'),
            description: t('dashboard.skillBuilderDesc', 'Discover the key skills required for your chosen career path.'),
        },
        {
            key: 'visualizer',
            path: '/dashboard/visualizer',
            icon: 'fas fa-project-diagram',
            title: t('common.careerVisualizer'),
            description: t('dashboard.careerVisualizerDesc', 'Visually explore the connections between subjects, degrees, and careers.'),
        },
        {
            key: 'timeline',
            path: '/dashboard/timeline',
            icon: 'fas fa-calendar-alt',
            title: t('common.timelineTracker'),
            description: t('dashboard.timelineTrackerDesc', 'Stay updated on all important admission and scholarship dates.'),
        },
        {
            key: 'psychology',
            path: '/dashboard/psychology',
            icon: 'fas fa-brain',
            title: t('common.psychologicalAnalysis', 'Psychological Analysis'),
            description: t('dashboard.psychologicalAnalysisDesc', 'Understand your cognitive profile and how it aligns with different careers.'),
        },
    ];

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1
            }
        }
    };

    const itemVariants = {
        hidden: { y: 20, opacity: 0 },
        visible: {
            y: 0,
            opacity: 1
        }
    };

    return (
        <MotionDiv
            className="home-view"
            initial="hidden"
            animate="visible"
            variants={containerVariants}
        >
            <div className="career-overview">
                <div className="career-overview-hero">
                    <div className="career-overview-title">
                        Career Intelligence Overview
                    </div>
                    <div className="career-overview-subtitle">
                        Welcome back, {user?.name || 'Explorer'}. Track your progress and use AI-powered insights to complete your career evaluation journey.
                    </div>
                    <div className="career-overview-badges">
                        <span className="pill-badge">
                            <i className="fas fa-bolt accent" aria-hidden="true" /> Progress <span className="accent">{progressPct}%</span>
                        </span>
                        <span className="pill-badge">
                            <i className="fas fa-shield-alt accent" aria-hidden="true" /> Onboarding status{' '}
                            <span className="accent">{quizDone ? 'Complete' : 'In progress'}</span>
                        </span>
                    </div>
                </div>

                <div className="career-overview-side">
                    <div className="overview-card">
                        <div className="overview-card-header">
                            <div className="overview-card-title">
                                <i className="fas fa-route" aria-hidden="true" />
                                Career Intelligence Progress
                            </div>
                            <div className="overview-progress-meta">{progressPct}%</div>
                        </div>
                        <div className="overview-progress-bar" aria-hidden="true">
                            <div className="overview-progress-fill" style={{ width: `${progressPct}%` }} />
                        </div>

                        {progressPct === 100 ? (
                            <div className="overview-checklist">
                                <div className="overview-check-item">
                                    <div className="overview-check-left">
                                        <span className={`overview-check-dot ${psychDone ? 'done' : ''}`} />
                                        Psychological Analysis
                                    </div>
                                    <div className={`overview-check-status ${psychDone ? 'done' : ''}`}>
                                        {psychDone ? 'Completed' : 'Pending'}
                                    </div>
                                </div>
                                <div className="overview-check-item">
                                    <div className="overview-check-left">
                                        <span className={`overview-check-dot ${voiceDone ? 'done' : ''}`} />
                                        Voice Insight
                                    </div>
                                    <div className={`overview-check-status ${voiceDone ? 'done' : ''}`}>
                                        {voiceDone ? 'Completed' : 'Pending'}
                                    </div>
                                </div>
                                <div className="overview-check-item">
                                    <div className="overview-check-left">
                                        <span className={`overview-check-dot ${quizDone ? 'done' : ''}`} />
                                        AI Career Quiz
                                    </div>
                                    <div className={`overview-check-status ${quizDone ? 'done' : ''}`}>
                                        {quizDone ? 'Completed' : 'Pending'}
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="overview-action-centered">
                                <Link to="/dashboard/psychology" className="discover-direction-btn">
                                    <span className="btn-glow-effect"></span>
                                    <span className="btn-content">Discover your direction <i className="fas fa-arrow-right"></i></span>
                                </Link>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="feature-cards-container">
                {features.map((feature) => (
                    <motion.div
                        key={feature.key}
                        variants={itemVariants}
                        whileTap={{ scale: 0.98 }}
                    >
                        <Link to={feature.path} className="feature-card" style={{ textDecoration: 'none', color: 'inherit', display: 'block' }}>
                            <div className="card-icon"><i className={feature.icon}></i></div>
                            <div className="card-content">
                                <h2>{feature.title}</h2>
                                <p>{feature.description}</p>
                            </div>
                            <span className="card-arrow">&rarr;</span>
                        </Link>
                    </motion.div>
                ))}
            </div>
        </MotionDiv>
    );
};

Home.propTypes = {
    user: PropTypes.object
};

export default Home;
