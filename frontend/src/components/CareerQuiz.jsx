import React, { useState, useRef, useEffect } from 'react';
import './CareerQuiz.css';
import { useTranslation } from 'react-i18next';
import { useOutletContext } from 'react-router-dom';

// FINAL: All questions with their full list of options
// selectionType: "single" | "multiple" strictly controls behaviour
const questions = [
    {
        id: 'Q1',
        question: 'What are your favorite subjects?',
        selectionType: 'multiple',
        options: ['Accountancy', 'Biology', 'Business Studies', 'Chemistry', 'Computer Science', 'Design', 'Economics', 'Fine Arts', 'History', 'Maths', 'Physics', 'Political Science', 'Psychology'],
    },
    {
        id: 'Q2',
        question: 'Which activities do you enjoy most?',
        selectionType: 'multiple',
        options: ['Coding', 'Debating', 'Designing', 'Drawing', 'Experiments', 'Helping Others', 'Organizing Events', 'Public Speaking', 'Reading', 'Research', 'Solving Puzzles', 'Sports', 'Writing'],
    },
    {
        id: 'Q3',
        question: 'What do you consider your strongest skills?',
        selectionType: 'multiple',
        options: ['Analysis', 'Communication', 'Creativity', 'Design Thinking', 'Financial Analysis', 'Leadership', 'Presentation', 'Problem Solving', 'Programming', 'Research', 'Teamwork', 'Writing'],
    },
    {
        id: 'Q4',
        question: 'Which work style suits you better?',
        selectionType: 'single',
        options: ['Practical', 'Theoretical', 'Both'],
    },
    {
        id: 'Q5',
        question: 'What type of workplace do you prefer?',
        selectionType: 'multiple',
        options: ['Classroom', 'Corporate Office', 'Creative Studio', 'Government Office', 'NGO', 'Outdoors', 'Research Lab', 'Startup'],
    },
    {
        id: 'Q6',
        question: 'Are you ready for competitive exams?',
        selectionType: 'single',
        options: ['Yes', 'No', 'Maybe'],
    },
    {
        id: 'Q7',
        question: 'Where would you prefer to study/work?',
        selectionType: 'single',
        options: ['India', 'Abroad', 'Flexible'],
    },
    {
        id: 'Q8',
        question: 'What career values matter most to you?',
        selectionType: 'multiple',
        options: ['Job Security', 'Creativity & Freedom', 'Balanced'],
    },
    {
        id: 'Q9',
        question: 'What is your long-term career goal?',
        selectionType: 'single',
        options: ['Artist', 'Civil Servant', 'Data Scientist', 'Designer', 'Doctor', 'Engineer', 'Entrepreneur', 'Lawyer', 'Manager', 'Scientist', 'Teacher'],
    },
];


const optionKeyMap = {
    'Accountancy': 'accountancy', 'Biology': 'biology', 'Business Studies': 'businessStudies', 'Chemistry': 'chemistry', 'Computer Science': 'computerScience', 'Design': 'design', 'Economics': 'economics', 'Fine Arts': 'fineArts', 'History': 'history', 'Maths': 'maths', 'Physics': 'physics', 'Political Science': 'politicalScience', 'Psychology': 'psychology',
    'Coding': 'coding', 'Debating': 'debating', 'Designing': 'designing', 'Drawing': 'drawing', 'Experiments': 'experiments', 'Helping Others': 'helpingOthers', 'Organizing Events': 'organizingEvents', 'Public Speaking': 'publicSpeaking', 'Reading': 'reading', 'Research': 'research', 'Solving Puzzles': 'solvingPuzzles', 'Sports': 'sports', 'Writing': 'writing',
    'Analysis': 'analysis', 'Communication': 'communication', 'Creativity': 'creativity', 'Design Thinking': 'designThinking', 'Financial Analysis': 'financialAnalysis', 'Leadership': 'leadership', 'Presentation': 'presentation', 'Problem Solving': 'problemSolving', 'Programming': 'programming', 'Teamwork': 'teamwork',
    'Practical': 'practical', 'Theoretical': 'theoretical', 'Both': 'both',
    'Classroom': 'classroom', 'Corporate Office': 'corporateOffice', 'Creative Studio': 'creativeStudio', 'Government Office': 'governmentOffice', 'NGO': 'ngo', 'Outdoors': 'outdoors', 'Research Lab': 'researchLab', 'Startup': 'startup',
    'Yes': 'yes', 'No': 'no', 'Maybe': 'maybe',
    'India': 'india', 'Abroad': 'abroad', 'Flexible': 'flexible',
    'Job Security': 'jobSecurity', 'Creativity & Freedom': 'creativityFreedom', 'Balanced': 'balanced',
    'Artist': 'artist', 'Civil Servant': 'civilServant', 'Data Scientist': 'dataScientist', 'Designer': 'designer', 'Doctor': 'doctor', 'Engineer': 'engineer', 'Entrepreneur': 'entrepreneur', 'Lawyer': 'lawyer', 'Manager': 'manager', 'Scientist': 'scientist', 'Teacher': 'teacher'
};

const CareerQuiz = () => {
    const { t } = useTranslation();
    const { user } = useOutletContext() || {};
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
    const [answers, setAnswers] = useState(() => {
        const initialAnswers = {};
        questions.forEach(q => { initialAnswers[q.id] = []; });
        return initialAnswers;
    });
    const [stream, setStream] = useState('');
    const [academicPercentInput, setAcademicPercentInput] = useState('');
    const [academicError, setAcademicError] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [animation, setAnimation] = useState('slide-in');
    const [psychRequiredMessage, setPsychRequiredMessage] = useState('');

    const totalSteps = questions.length + 1; // +1 for academic background step
    const isAcademicStep = currentQuestionIndex === questions.length;
    const currentQuestion = !isAcademicStep ? questions[currentQuestionIndex] : null;
    const progress = ((currentQuestionIndex + 1) / totalSteps) * 100;

    // Enforce psychological assessment prerequisite
    useEffect(() => {
        const checkPsychStatus = async () => {
            if (!user?.email) return;
            try {
                const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
                const res = await fetch(
                    `${apiBase}/api/psych-status?user_email=${encodeURIComponent(user.email)}`
                );
                const data = await res.json();
                if (!res.ok || data.error) {
                    throw new Error(data.error || 'Unable to check psychological status.');
                }
                if (!data.completed) {
                    setPsychRequiredMessage(
                        'You must complete the Psychological Assessment before accessing the AI Career Quiz.'
                    );
                }
            } catch (err) {
                console.error(err);
            }
        };

        checkPsychStatus();
    }, [user?.email]);

    const translateOption = (option) => {
        const key = optionKeyMap[option];
        return key ? t(`quiz.options.${key}`, option) : option;
    };

    const handleSelectionChange = (questionId, option, selectionType) => {
        setAnswers(prevAnswers => {
            const currentSelection = prevAnswers[questionId] || [];
            let newSelection;

            if (selectionType === 'single') {
                // Only allow one selection; clicking again deselects
                newSelection = currentSelection.includes(option) ? [] : [option];
            } else {
                // Multiple selection toggle
                newSelection = currentSelection.includes(option)
                    ? currentSelection.filter(item => item !== option)
                    : [...currentSelection, option];
            }

            return { ...prevAnswers, [questionId]: newSelection };
        });
    };

    const handleNext = () => {
        if (currentQuestionIndex < totalSteps - 1) {
            setAnimation('slide-out');
            setTimeout(() => {
                setCurrentQuestionIndex(prev => prev + 1);
                setAnimation('slide-in');
            }, 300);
        }
    };

    const handleBack = () => {
        if (currentQuestionIndex > 0) {
            setAnimation('slide-out-back');
            setTimeout(() => {
                setCurrentQuestionIndex(prev => prev - 1);
                setAnimation('slide-in-back');
            }, 300);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        setAcademicError("");

        try {
            // Convert structured answers into a single descriptive string for the NLP model
            // Example: "Favorite subjects: Math, Physics. Skills: Analysis. ..."
            // Validate academic background inputs
            const trimmedStream = (stream || '').trim();
            const percentValue = parseFloat(academicPercentInput);
            if (!trimmedStream) {
                throw new Error('Please select your academic stream.');
            }
            if (Number.isNaN(percentValue) || percentValue < 0 || percentValue > 100) {
                setAcademicError('Please enter a valid percentage between 0 and 100.');
                throw new Error('Invalid academic percentage.');
            }

            const academicPercent = percentValue;

            // Convert structured answers into a single descriptive string for the NLP model
            const answersTextParts = Object.entries(answers)
                .map(([key, value]) => {
                    const question = questions.find(q => q.id === key);
                    const qText = question ? question.question : key;
                    const valText = Array.isArray(value) ? value.join(", ") : value;
                    return `${qText}: ${valText}`;
                });

            answersTextParts.push(`Academic stream: ${trimmedStream}`);
            answersTextParts.push(`Academic percentage: ${academicPercent}`);

            const answersText = answersTextParts.join(". ");

            const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

            const res = await fetch(`${apiBase}/api/quiz/submit`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    features: answersText,
                    answers_text: answersText,
                    structured_answers: answers,
                    user_email: user?.email || null,
                    academic_percent: academicPercent,
                    stream: trimmedStream,
                }),
            });

            const data = await res.json();
            if (data.error) throw new Error(data.error);

            // Prefer unified fused results from the dedicated endpoint when we have a user
            if (user?.email) {
                const fusedRes = await fetch(
                    `${apiBase}/api/career-results?user_email=${encodeURIComponent(user.email)}`
                );
                const fusedData = await fusedRes.json();
                if (fusedData.error) {
                    throw new Error(fusedData.error);
                }
                setResult(fusedData);
            } else {
                // Fallback for anonymous users: use fusion included in submit response (if any)
                setResult({
                    career_rankings: data.career_rankings || [],
                    quiz_scores: data.quiz_scores || {},
                    psych_scores: data.psych_scores || {},
                });
            }
        } catch (err) {
            console.error(err);
            if (!academicError) {
                setError(err.message || "Something went wrong. Please try again.");
            }
        } finally {
            setLoading(false);
        }
    };


    const handleReset = () => {
        const initialAnswers = {};
        questions.forEach(q => { initialAnswers[q.id] = []; });
        setAnswers(initialAnswers);
        setStream('');
        setAcademicPercentInput('');
        setAcademicError('');
        setCurrentQuestionIndex(0);
        setResult(null);
    };

    if (psychRequiredMessage) {
        const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
        // Redirect handled by Dashboard routing; here we simply show the guard message.
        return (
            <div className="career-quiz-container">
                <div className="quiz-main-content">
                    <div className="quiz-card">
                        <div className="question-content">
                            <h2 className="question-text">
                                {psychRequiredMessage}
                            </h2>
                            <p className="multi-select-note">
                                Please complete the Psychological Assessment first from the sidebar.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    if (result && Array.isArray(result.career_rankings)) {
        const top = result.top_recommendation;
        const topTraits = top?.top_traits || [];
        const matchedSkills = top?.quiz_signals?.matched_skills || [];
        const mlProb = top?.quiz_signals?.ml_probability;
        const confidenceScore = top?.confidence_score;
        const alsoCloseCareer = top?.also_close_career;

        return (
            <div className="career-quiz-container show-results">
                <div className="quiz-results">
                    <h3>{t('quiz.yourRecommendationTitle')}</h3>
                    <p><strong>{t('quiz.recommendedPath')}</strong></p>

                    {result.voice_insight && (
                        <div className="voice-insight-section">
                            <h4><i className="fas fa-microphone-alt" /> Voice Insight</h4>
                            {result.voice_insight.transcript && (
                                <p className="voice-transcript-preview">{result.voice_insight.transcript}</p>
                            )}
                            <div className="voice-insight-metrics">
                                {typeof result.voice_insight.motivation_score === 'number' && (
                                    <span>Motivation: <span className="quiz-explanation-highlight">{Math.round(result.voice_insight.motivation_score)}%</span></span>
                                )}
                                {typeof result.voice_insight.confidence_score === 'number' && (
                                    <span>Confidence: <span className="quiz-explanation-highlight">{Math.round(result.voice_insight.confidence_score)}%</span></span>
                                )}
                                {result.voice_insight.top_voice_career && (
                                    <span>Top (voice): <span className="quiz-explanation-highlight">{result.voice_insight.top_voice_career}</span></span>
                                )}
                            </div>
                        </div>
                    )}

                    {top && (
                        <div className="quiz-explanation">
                            <h4>Why this career?</h4>
                            <p>
                                {top.explanation}
                            </p>
                            <div className="quiz-contrib-row">
                                {typeof top.quiz_component === 'number' && (
                                    <span>
                                        Quiz:{' '}
                                        <span className="quiz-explanation-highlight">
                                            {Math.round(top.quiz_component)}%
                                        </span>
                                    </span>
                                )}
                                {typeof top.psych_component === 'number' && (
                                    <span>
                                        Psych:{' '}
                                        <span className="quiz-explanation-highlight">
                                            {Math.round(top.psych_component)}%
                                        </span>
                                    </span>
                                )}
                                {typeof top.voice_component === 'number' && (
                                    <span>
                                        Voice:{' '}
                                        <span className="quiz-explanation-highlight voice-highlight">
                                            {Math.round(top.voice_component)}%
                                        </span>
                                    </span>
                                )}
                                {typeof mlProb === 'number' && (
                                    <span>
                                        Model confidence:{' '}
                                        <span className="quiz-explanation-highlight">
                                            {Math.round(mlProb * 100)}%
                                        </span>
                                    </span>
                                )}
                            </div>
                            {(topTraits.length > 0 || matchedSkills.length > 0) && (
                                <div className="quiz-badge-row">
                                    {topTraits.map(trait => (
                                        <span key={trait.trait} className="quiz-badge trait">
                                            {trait.trait.replace(/_/g, ' ')} · {Math.round(trait.user_score)}%
                                        </span>
                                    ))}
                                    {matchedSkills.map(skill => (
                                        <span key={skill} className="quiz-badge skill">
                                            {skill}
                                        </span>
                                    ))}
                                </div>
                            )}
                            {typeof confidenceScore === 'number' && (
                                <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                    Confidence gap vs next career:{' '}
                                    <span className="quiz-explanation-highlight">
                                        {confidenceScore.toFixed(1)} pts
                                    </span>
                                    {alsoCloseCareer && confidenceScore < 10 && (
                                        <> – also close to {alsoCloseCareer}</>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    <div className="career-cards">
                        {result.career_rankings.map((item) => (
                            <div key={item.career} className="career-card">
                                <div className="career-header">
                                    <h4>{item.career}</h4>
                                    <span className="career-score">
                                        {Math.round(item.final_score)}%
                                    </span>
                                </div>
                                <div className="career-components">
                                    <div>
                                        <span>Quiz</span>
                                        <div className="bar">
                                            <div
                                                className="fill academic"
                                                style={{
                                                    width: `${Math.round(item.quiz_component || 0)}%`,
                                                }}
                                            />
                                        </div>
                                    </div>
                                    <div>
                                        <span>Psychological</span>
                                        <div className="bar">
                                            <div
                                                className="fill psych"
                                                style={{
                                                    width: `${Math.round(item.psych_component || 0)}%`,
                                                }}
                                            />
                                        </div>
                                    </div>
                                    {typeof item.voice_component === 'number' && (
                                        <div>
                                            <span>Voice</span>
                                            <div className="bar">
                                                <div
                                                    className="fill voice"
                                                    style={{
                                                        width: `${Math.round(item.voice_component || 0)}%`,
                                                    }}
                                                />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>

                    <button onClick={handleReset} className="btn-nav btn-retake">
                        <i className="fas fa-redo"></i> {t('quiz.retakeQuiz')}
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="career-quiz-container">
            <div className="quiz-main-content">
                <div className="quiz-progress-bar">
                    <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                </div>

                <div className="quiz-top-nav">
                    {[...questions, { id: 'ACADEMIC' }].map((q, index) => (
                        <div
                            key={q.id}
                            className={`nav-item-top ${currentQuestionIndex === index ? 'active' : ''} ${
                                index < questions.length
                                    ? (answers[q.id]?.length > 0 ? 'answered' : '')
                                    : (stream && academicPercentInput ? 'answered' : '')
                            }`}
                        >
                            {index < questions.length
                                ? (answers[q.id]?.length > 0 ? <i className="fas fa-check-circle"></i> : index + 1)
                                : (stream && academicPercentInput ? <i className="fas fa-check-circle"></i> : totalSteps)}
                        </div>
                    ))}
                </div>

                <div className="quiz-card" key={currentQuestionIndex}>
                    <div className={`question-content ${animation}`}>
                        {!isAcademicStep && (
                            <>
                                <h2 className="question-text">
                                    {t(`quiz.questions.${currentQuestion.id}.text`, currentQuestion.question)}
                                </h2>
                                {currentQuestion.selectionType === 'multiple' && (
                                    <p className="multi-select-note">
                                        You can select multiple options.
                                    </p>
                                )}
                                <div className="options-grid">
                                    {currentQuestion.options.map(option => (
                                        <button
                                            key={option}
                                            className={`option-card ${answers[currentQuestion.id]?.includes(option) ? 'selected' : ''}`}
                                            onClick={() =>
                                                handleSelectionChange(
                                                    currentQuestion.id,
                                                    option,
                                                    currentQuestion.selectionType
                                                )
                                            }
                                        >
                                            {translateOption(option)}
                                        </button>
                                    ))}
                                </div>
                            </>
                        )}
                        {isAcademicStep && (
                            <>
                                <h2 className="question-text">
                                    What is your academic background?
                                </h2>
                                <div className="academic-section">
                                    <h3>Part A – Stream selection</h3>
                                    <p className="single-select-note">
                                        Choose your primary stream (single selection).
                                    </p>
                                    <div className="options-grid">
                                        {['Arts', 'Science', 'Commerce'].map(option => (
                                            <button
                                                key={option}
                                                className={`option-card ${stream === option ? 'selected' : ''}`}
                                                onClick={() =>
                                                    setStream(prev =>
                                                        prev === option ? '' : option
                                                    )
                                                }
                                            >
                                                {option}
                                            </button>
                                        ))}
                                    </div>

                                    <h3>Part B – Percentage</h3>
                                    <label className="academic-percent-label">
                                        Enter your percentage in previous class
                                    </label>
                                    <input
                                        type="number"
                                        min="0"
                                        max="100"
                                        step="0.1"
                                        value={academicPercentInput}
                                        onChange={e => {
                                            setAcademicPercentInput(e.target.value);
                                            setAcademicError('');
                                        }}
                                        className={`academic-percent-input ${academicError ? 'has-error' : ''}`}
                                    />
                                    {academicError && (
                                        <p className="error-message">{academicError}</p>
                                    )}
                                </div>
                            </>
                        )}
                    </div>
                </div>

                <div className="quiz-navigation">
                    <button onClick={handleBack} className="btn-nav btn-back" disabled={currentQuestionIndex === 0}>
                        <i className="fas fa-arrow-left"></i> {t('quiz.back')}
                    </button>
                    {currentQuestionIndex < totalSteps - 1 ? (
                        <button onClick={handleNext} className="btn-nav btn-next">
                            {t('quiz.next')} <i className="fas fa-arrow-right"></i>
                        </button>
                    ) : (
                        <button onClick={handleSubmit} className="btn-nav btn-submit" disabled={loading}>
                            {loading ? t('quiz.predicting') : t('quiz.submit')}
                        </button>
                    )}
                </div>
                {error && <p className="error-message">{error}</p>}
            </div>
        </div>
    );
};

export default CareerQuiz;