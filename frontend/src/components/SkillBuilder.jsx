import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import './SkillBuilder.css';

// Curated content for each skill
const skillDetails = {
  problemSolving: {
    title: "Problem Solving",
    icon: "fas fa-puzzle-piece",
    overview: "Problem solving is the ability to identify complex problems, review related information, develop and evaluate options, and implement solutions. It's the #1 skill employers look for across all engineering and tech disciplines.",
    tips: [
      "Break down large complex problems into smaller, manageable sub-problems.",
      "Practice edge-case thinking: What happens if the input is empty? What if it's massive?",
      "Explain your thought process out loud (Rubber Duck Debugging)."
    ],
    resources: [
      { name: "LeetCode: Top Interview Questions", url: "https://leetcode.com/" },
      { name: "HackerRank: Problem Solving Path", url: "https://www.hackerrank.com/" },
      { name: "Project Euler", url: "https://projecteuler.net/" }
    ]
  },
  communication: {
    title: "Communication",
    icon: "fas fa-comments",
    overview: "In tech, writing code is only half the job. The other half is explaining your code, writing documentation, advocating for your ideas, and collaborating effectively with product managers and designers.",
    tips: [
      "Write concise, descriptive Pull Request (PR) summaries.",
      "Practice the 'BLUF' method (Bottom Line Up Front) when writing emails.",
      "Always assume positive intent during code reviews."
    ],
    resources: [
      { name: "Coursera: Effective Communication", url: "https://www.coursera.org/learn/communication" },
      { name: "Google Developer Documentation Guidelines", url: "https://developers.google.com/style" }
    ]
  },
  dsa: {
    title: "Data Structures & Algorithms",
    icon: "fas fa-sitemap",
    overview: "Data Structures and Algorithms (DSA) are the building blocks of efficient software. Understanding which structures to use (Arrays, HashMaps, Trees, Graphs) dictates how well your application will scale.",
    tips: [
      "Don't just memorize algorithms; understand the 'Why' behind their time complexity (Big O).",
      "Master HashMaps and Arrays first—they solve 70% of standard problems.",
      "Learn to recognize patterns (Sliding Window, Two Pointers, DFS/BFS)."
    ],
    resources: [
      { name: "NeetCode Roadmap", url: "https://neetcode.io/roadmap" },
      { name: "Cracking the Coding Interview (Book)", url: "https://www.crackingthecodinginterview.com/" },
      { name: "VisuAlgo: Visualizing DSA", url: "https://visualgo.net/en" }
    ]
  },
  projectManagement: {
    title: "Project Management",
    icon: "fas fa-tasks",
    overview: "Agile, Scrum, Kanban—knowing how to manage tasks, estimate time, and deliver features on schedule is critical as you move from junior to mid-level and senior roles.",
    tips: [
      "Learn how to break features down into 1-2 day 'tickets' or 'stories'.",
      "Always pad your time estimates by 20% for unexpected bugs.",
      "Understand the difference between Agile and Waterfall methodologies."
    ],
    resources: [
      { name: "Atlassian Agile Coach", url: "https://www.atlassian.com/agile" },
      { name: "Google Project Management Certificate", url: "https://grow.google/certificates/project-management/" }
    ]
  },
  cloud: {
    title: "Cloud Computing",
    icon: "fas fa-cloud",
    overview: "Modern applications don't run on local servers. They scale dynamically on platforms like AWS, Microsoft Azure, and Google Cloud. Understanding cloud infrastructure, serverless, and deployment is essential.",
    tips: [
      "Understand the Shared Responsibility Model of cloud security.",
      "Deploy a simple web app using Vercel or Netlify to learn PaaS.",
      "Learn the basics of Docker containers and CI/CD pipelines."
    ],
    resources: [
      { name: "AWS Cloud Practitioner Training", url: "https://aws.amazon.com/training/" },
      { name: "Microsoft Azure Fundamentals", url: "https://learn.microsoft.com/en-us/training/paths/az-900-describe-cloud-concepts/" },
      { name: "A Cloud Guru", url: "https://acloudguru.com/" }
    ]
  }
};

const SkillBuilder = () => {
  const { t } = useTranslation();
  const [selectedSkill, setSelectedSkill] = useState(null);

  const skills = [
    { key: 'problemSolving', icon: 'fas fa-puzzle-piece' },
    { key: 'communication', icon: 'fas fa-comments' },
    { key: 'dsa', icon: 'fas fa-sitemap' },
    { key: 'projectManagement', icon: 'fas fa-tasks' },
    { key: 'cloud', icon: 'fas fa-cloud' },
  ];

  const handleLearnMore = (key) => {
    setSelectedSkill(skillDetails[key]);
  };

  const closeModal = () => {
    setSelectedSkill(null);
  };

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
            <button
              className="btn-learn-more"
              onClick={() => handleLearnMore(skill.key)}
            >
              {t('common.learnMore')}
            </button>
          </div>
        ))}
      </div>

      {/* Modal Overlay */}
      <AnimatePresence>
        {selectedSkill && (
          <motion.div
            className="skill-modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeModal}
          >
            <motion.div
              className="skill-modal-content"
              initial={{ y: 50, opacity: 0, scale: 0.95 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 20, opacity: 0, scale: 0.95 }}
              onClick={(e) => e.stopPropagation()} // Prevent close when clicking inside
            >
              <button className="skill-modal-close" onClick={closeModal}>
                <i className="fas fa-times"></i>
              </button>

              <div className="skill-modal-header">
                <div className="modal-icon-wrapper">
                  <i className={selectedSkill.icon}></i>
                </div>
                <h2>{selectedSkill.title}</h2>
              </div>

              <div className="skill-modal-body">
                <div className="modal-section">
                  <h3>Why it Matters</h3>
                  <p>{selectedSkill.overview}</p>
                </div>

                <div className="modal-section">
                  <h3>Actionable Tips</h3>
                  <ul className="modal-tips-list">
                    {selectedSkill.tips.map((tip, idx) => (
                      <li key={idx}>
                        <i className="fas fa-check-circle tip-icon"></i>
                        <span>{tip}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="modal-section">
                  <h3>Recommended Resources</h3>
                  <div className="resource-links">
                    {selectedSkill.resources.map((res, idx) => (
                      <a
                        key={idx}
                        href={res.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="resource-card"
                      >
                        <span className="resource-name">{res.name}</span>
                        <i className="fas fa-external-link-alt"></i>
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SkillBuilder;