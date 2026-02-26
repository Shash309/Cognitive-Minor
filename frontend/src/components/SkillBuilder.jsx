import React from 'react';
import './SkillBuilder.css';
import { useTranslation } from 'react-i18next';

const SkillBuilder = () => {
  const { t } = useTranslation();

  const skills = [
    { key: 'problemSolving', icon: 'fas fa-puzzle-piece' },
    { key: 'communication', icon: 'fas fa-comments' },
    { key: 'dsa', icon: 'fas fa-sitemap' },
    { key: 'projectManagement', icon: 'fas fa-tasks' },
    { key: 'cloud', icon: 'fas fa-cloud' },
  ];

  return (
    <div className="skill-builder-container">
      <div className="skill-builder-header">
        <h2>{t('skills.title')}</h2>
        <p>{t('skills.subtitle')}</p>
      </div>
      <div className="skills-list">
        {skills.map((skill, index) => (
          <div 
            key={skill.key} 
            className="skill-card"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="skill-icon">
              <i className={skill.icon}></i>
            </div>
            <h3>{t(`skills.items.${skill.key}.name`)}</h3>
            <p>{t(`skills.items.${skill.key}.description`)}</p>
            <button className="btn-learn-more">{t('common.learnMore')}</button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SkillBuilder;