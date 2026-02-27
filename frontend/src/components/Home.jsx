import React from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import PropTypes from 'prop-types';
import { motion } from 'framer-motion';

const Home = ({ user }) => {
    const { t } = useTranslation();

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
        <motion.div
            className="home-view"
            initial="hidden"
            animate="visible"
            variants={containerVariants}
        >
            <header className="home-header">
                <h1>{t('dashboard.welcomeBackUser', { name: user?.name || 'Explorer' })}</h1>
                <p>{t('dashboard.welcomeSub', 'Your journey to a successful career continues here. What would you like to do today?')}</p>
            </header>

            <div className="feature-cards-container">
                {features.map((feature) => (
                    <motion.div
                        key={feature.key}
                        variants={itemVariants}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
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
        </motion.div>
    );
};

Home.propTypes = {
    user: PropTypes.object
};

export default Home;
