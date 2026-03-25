import React, { useState, useEffect } from 'react';
import './CareerQuiz.css';
import { useTranslation } from 'react-i18next';
import { useOutletContext, useNavigate } from 'react-router-dom';

const CareerQuiz = () => {
    const { t } = useTranslation();
    const { user, progress, refreshProgress } = useOutletContext() || {};
    const navigate = useNavigate();

    const [staticQuestions, setStaticQuestions] = useState([]);
    const [adaptiveQuestions, setAdaptiveQuestions] = useState([]);
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);

    const [staticAnswers, setStaticAnswers] = useState({});
    const [adaptiveAnswers, setAdaptiveAnswers] = useState({});

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(true);
    const [isFetchingAdaptive, setIsFetchingAdaptive] = useState(false);
    const [error, setError] = useState("");
    const [animation, setAnimation] = useState('slide-in');
    const [psychRequiredMessage, setPsychRequiredMessage] = useState('');
    const [isAdaptivePhase, setIsAdaptivePhase] = useState(false);

    const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

    useEffect(() => {
        if (!user?.email) return;

        if (!progress?.psych_completed || !progress?.voice_completed) {
            setPsychRequiredMessage('You must complete both the Psychological and Voice Assessments before accessing the AI Career Quiz.');
            setLoading(false);
            return;
        }

        const fetchStatic = async () => {
            try {
                const res = await fetch(`${apiBase}/api/quiz/static`);
                const text = await res.text();
                let data;
                try {
                    data = JSON.parse(text);
                } catch (err) {
                    console.error("RAW RESPONSE (Static):", text);
                    throw new Error("Backend did not return valid JSON for static questions.");
                }
                
                if (data.error) throw new Error(data.error);
                setStaticQuestions(data);
                
                const initials = {};
                data.forEach(q => { initials[q.id] = []; });
                setStaticAnswers(initials);
            } catch (err) {
                setError("Failed to load quiz. Please try again.");
            } finally {
                setLoading(false);
            }
        };

        fetchStatic();
    }, [user?.email, progress]);

    const activeQuestions = isAdaptivePhase ? adaptiveQuestions : staticQuestions;
    const currentAnswers = isAdaptivePhase ? adaptiveAnswers : staticAnswers;
    const setCurrentAnswers = isAdaptivePhase ? setAdaptiveAnswers : setStaticAnswers;
    const currentQuestion = activeQuestions[currentQuestionIndex];
    const isEndOfPhase = currentQuestionIndex >= activeQuestions.length - 1;

    const totalSteps = staticQuestions.length + (adaptiveQuestions.length || 3);
    const currentOverallIndex = isAdaptivePhase ? staticQuestions.length + currentQuestionIndex : currentQuestionIndex;
    const calcProgress = ((currentOverallIndex + 1) / totalSteps) * 100;

    const handleSelectionChange = (questionId, option, selectionType) => {
        setCurrentAnswers(prev => {
            const currentSelection = prev[questionId] || [];
            let newSelection;

            if (selectionType === 'single') {
                newSelection = currentSelection.includes(option) ? [] : [option];
            } else {
                newSelection = currentSelection.includes(option)
                    ? currentSelection.filter(item => item !== option)
                    : [...currentSelection, option];
            }

            return { ...prev, [questionId]: newSelection };
        });
    };

    const fetchAdaptive = async () => {
        setIsFetchingAdaptive(true);
        setError("");
        try {
            const res = await fetch(`${apiBase}/api/quiz/adaptive`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_email: user.email, static_answers: staticAnswers }),
            });
            const text = await res.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch (err) {
                console.error("RAW RESPONSE (Adaptive):", text);
                throw new Error("Backend did not return valid JSON for adaptive questions.");
            }
            if (data.error) throw new Error(data.error);
            
            setAdaptiveQuestions(data);
            const initials = {};
            data.forEach(q => { initials[q.id] = []; });
            setAdaptiveAnswers(initials);
            
            setIsAdaptivePhase(true);
            setCurrentQuestionIndex(0);
            setAnimation('slide-in');
        } catch (err) {
            setError(err.message || "Failed to load adaptive questions.");
        } finally {
            setIsFetchingAdaptive(false);
        }
    };

    const handleSubmit = async () => {
        setLoading(true);
        setError("");
        try {
            const res = await fetch(`${apiBase}/api/quiz/submit`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    static_answers: staticAnswers,
                    adaptive_answers: adaptiveAnswers,
                    user_email: user.email
                }),
            });
            const text = await res.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch (err) {
                console.error("RAW RESPONSE (Submit):", text);
                throw new Error("Backend did not return valid JSON for quiz submission.");
            }
            if (data.error) throw new Error(data.error);

            if (refreshProgress) await refreshProgress();
            
            if (user?.email) {
                navigate('/dashboard/profile', {
                    replace: true,
                    state: { highlightLatestResult: true },
                });
            } else {
                setResult(data);
            }
        } catch (err) {
            setError(err.message || "Something went wrong. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const handleNext = () => {
        setAnimation('slide-out');
        setTimeout(() => {
            if (isEndOfPhase) {
                if (!isAdaptivePhase) {
                    fetchAdaptive();
                } else {
                    handleSubmit();
                }
            } else {
                setCurrentQuestionIndex(prev => prev + 1);
                setAnimation('slide-in');
            }
        }, 300);
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

    const handleReset = () => {
        setIsAdaptivePhase(false);
        setCurrentQuestionIndex(0);
        setAdaptiveQuestions([]);
        setResult(null);
        setStaticAnswers(prev => {
            const reset = {};
            Object.keys(prev).forEach(k => reset[k] = []);
            return reset;
        });
    };

    if (psychRequiredMessage) {
        return (
            <div className="career-quiz-container">
                <div className="quiz-main-content">
                    <div className="quiz-card">
                        <div className="question-content">
                            <h2 className="question-text">{psychRequiredMessage}</h2>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    if (loading || isFetchingAdaptive) {
        return (
            <div className="career-quiz-container">
                <div className="quiz-main-content">
                    <div className="quiz-card" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '300px' }}>
                        <i className="fas fa-spinner fa-spin fa-3x" style={{ color: 'var(--brand)' }}></i>
                    </div>
                </div>
            </div>
        );
    }

    if (result) {
        return (
            <div className="career-quiz-container show-results">
                <div className="quiz-results">
                    <h3>Your Recommendation</h3>
                    <div className="career-cards">
                        {result.career_rankings?.map((item) => (
                            <div key={item.career} className="career-card" style={{border: '1px solid var(--border)', padding: '1rem', borderRadius: '8px', marginBottom: '1rem'}}>
                                <h4>{item.career}</h4>
                                <span className="career-score" style={{color: 'var(--brand)', fontWeight: 'bold'}}>{Math.round(item.final_score)}%</span>
                            </div>
                        ))}
                    </div>
                    {result.top_recommendation && (
                        <div style={{marginTop: '2rem', padding: '1.5rem', background: 'var(--bg-secondary)', borderRadius: '12px'}}>
                            <h4>Top Pick: {result.top_recommendation.career}</h4>
                            <p>{result.top_recommendation.explanation}</p>
                            <span style={{opacity: 0.8, fontSize: '0.9rem'}}>Confidence: {Math.round(result.top_recommendation.confidence_score)}%</span>
                        </div>
                    )}
                    <button onClick={handleReset} className="btn-nav btn-retake">
                        <i className="fas fa-redo"></i> Retake Quiz
                    </button>
                </div>
            </div>
        );
    }

    if (!currentQuestion) return null;

    const answeredCount = Object.values(currentAnswers).filter(arr => arr && arr.length > 0).length;

    return (
        <div className="career-quiz-container">
            <div className="quiz-main-content">
                <div className="quiz-progress-bar">
                    <div className="progress-fill" style={{ width: `${calcProgress}%` }}></div>
                </div>

                <div className="quiz-top-nav">
                    {activeQuestions.map((q, index) => (
                        <div
                            key={q.id}
                            className={`nav-item-top ${currentQuestionIndex === index ? 'active' : ''} ${
                                currentAnswers[q.id]?.length > 0 ? 'answered' : ''
                            }`}
                        >
                            {currentAnswers[q.id]?.length > 0 ? <i className="fas fa-check-circle"></i> : index + 1}
                        </div>
                    ))}
                </div>

                <div className="quiz-card" key={currentQuestionIndex}>
                    <div className={`question-content ${animation}`}>
                        <h2 className="question-text">
                            {currentQuestion.question}
                        </h2>
                        {currentQuestion.selectionType === 'multiple' && (
                            <p className="multi-select-note">You can select multiple options.</p>
                        )}
                        <div className="options-grid">
                            {currentQuestion.options.map(option => {
                                const selected = currentAnswers[currentQuestion.id]?.includes(option);
                                return (
                                    <button
                                        key={option}
                                        className={`option-card ${selected ? 'selected' : ''}`}
                                        onClick={() => handleSelectionChange(currentQuestion.id, option, currentQuestion.selectionType)}
                                    >
                                        <span className="option-label">{option}</span>
                                        {selected && <i className="fas fa-check option-check" aria-hidden="true" />}
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>

                <div className="quiz-navigation">
                    <button onClick={handleBack} className="btn-nav btn-back" disabled={currentQuestionIndex === 0 && !isAdaptivePhase}>
                        <i className="fas fa-arrow-left"></i> {t('quiz.back', 'Back')}
                    </button>
                    <button onClick={handleNext} className="btn-nav btn-next">
                        {isEndOfPhase && isAdaptivePhase ? (
                            loading ? t('quiz.predicting', 'Predicting...') : t('quiz.submit', 'Submit')
                        ) : (
                            isEndOfPhase && !isAdaptivePhase ? 'Next Phase' : t('quiz.next', 'Next')
                        )} 
                        <i className={isEndOfPhase && isAdaptivePhase ? '' : "fas fa-arrow-right"}></i>
                    </button>
                </div>
                {error && <p className="error-message">{error}</p>}
            </div>
        </div>
    );
};

export default CareerQuiz;