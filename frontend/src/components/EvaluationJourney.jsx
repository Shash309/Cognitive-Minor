import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import './EvaluationJourney.css';

const STEPS = [
  { key: 'psych', label: 'Psychological Analysis', icon: 'fas fa-brain' },
  { key: 'voice', label: 'Voice Insight', icon: 'fas fa-microphone-alt' },
  { key: 'quiz', label: 'AI Career Quiz', icon: 'fas fa-tasks' },
  { key: 'results', label: 'Results', icon: 'fas fa-chart-line' },
];

export default function EvaluationJourney({ currentStep, progress }) {
  const completion = useMemo(() => {
    const psych = Boolean(progress?.psych_completed);
    const voice = Boolean(progress?.voice_completed);
    const quiz = Boolean(progress?.quiz_completed);
    return {
      psych,
      voice,
      quiz,
      results: quiz,
    };
  }, [progress]);

  return (
    <div className="eval-journey" role="region" aria-label="Career Evaluation Journey">
      <div className="eval-journey-header">
        <div className="eval-journey-title">Career Evaluation Journey</div>
        <div className="eval-journey-subtitle">Track your progress across the core evaluation modules.</div>
      </div>

      <div className="eval-journey-steps" role="list">
        {STEPS.map((step, idx) => {
          const isActive = currentStep === step.key;
          const isDone = Boolean(completion[step.key]);
          return (
            <div
              key={step.key}
              className={[
                'eval-step',
                isActive ? 'active' : '',
                isDone ? 'done' : '',
              ].filter(Boolean).join(' ')}
              role="listitem"
            >
              <div className="eval-step-index" aria-hidden="true">
                {isDone ? <i className="fas fa-check" /> : idx + 1}
              </div>
              <div className="eval-step-main">
                <div className="eval-step-label">
                  <i className={step.icon} aria-hidden="true" /> {step.label}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

EvaluationJourney.propTypes = {
  currentStep: PropTypes.oneOf(['psych', 'voice', 'quiz', 'results']),
  progress: PropTypes.object,
};

