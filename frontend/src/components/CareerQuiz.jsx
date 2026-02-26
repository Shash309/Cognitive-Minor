import React, { useState, useRef, useEffect } from 'react';
import './CareerQuiz.css';
import { useTranslation } from 'react-i18next';

// FINAL: All questions with their full list of options
const questions = [
    { id: 'Q1', text: 'What are your favorite subjects?', type: 'multi-select', options: ['Accountancy', 'Biology', 'Business Studies', 'Chemistry', 'Computer Science', 'Design', 'Economics', 'Fine Arts', 'History', 'Maths', 'Physics', 'Political Science', 'Psychology'] },
    { id: 'Q2', text: 'Which activities do you enjoy most?', type: 'multi-select', options: ['Coding', 'Debating', 'Designing', 'Drawing', 'Experiments', 'Helping Others', 'Organizing Events', 'Public Speaking', 'Reading', 'Research', 'Solving Puzzles', 'Sports', 'Writing'] },
    { id: 'Q3', text: 'What do you consider your strongest skills?', type: 'multi-select', options: ['Analysis', 'Communication', 'Creativity', 'Design Thinking', 'Financial Analysis', 'Leadership', 'Presentation', 'Problem Solving', 'Programming', 'Research', 'Teamwork', 'Writing'] },
    { id: 'Q4', text: 'Which work style suits you better?', type: 'multi-select', options: ['Practical', 'Theoretical', 'Both'] },
    { id: 'Q5', text: 'What type of workplace do you prefer?', type: 'multi-select', options: ['Classroom', 'Corporate Office', 'Creative Studio', 'Government Office', 'NGO', 'Outdoors', 'Research Lab', 'Startup'] },
    { id: 'Q6', text: 'Are you ready for competitive exams?', type: 'multi-select', options: ['Yes', 'No', 'Maybe'] },
    { id: 'Q7', text: 'Where would you prefer to study/work?', type: 'multi-select', options: ['India', 'Abroad', 'Flexible'] },
    { id: 'Q8', text: 'What career values matter most to you?', type: 'multi-select', options: ['Job Security', 'Creativity & Freedom', 'Balanced'] },
    { id: 'Q9', text: 'What is your long-term career goal?', type: 'multi-select', options: ['Artist', 'Civil Servant', 'Data Scientist', 'Designer', 'Doctor', 'Engineer', 'Entrepreneur', 'Lawyer', 'Manager', 'Scientist', 'Teacher'] },
    { id: 'Q10', text: 'What is your academic background (with %)?', type: 'multi-select', options: ['Arts-41%', 'Arts-42%', 'Arts-43%', 'Arts-44%', 'Arts-45%', 'Arts-46%', 'Arts-47%', 'Arts-48%', 'Arts-49%', 'Arts-50%', 'Arts-51%', 'Arts-54%', 'Arts-55%', 'Arts-56%', 'Arts-57%', 'Arts-58%', 'Arts-59%', 'Arts-60%', 'Arts-61%', 'Arts-62%', 'Arts-63%', 'Arts-64%', 'Arts-65%', 'Arts-66%', 'Arts-67%', 'Arts-68%', 'Arts-69%', 'Arts-70%', 'Arts-71%', 'Arts-72%', 'Arts-73%', 'Arts-74%', 'Arts-75%', 'Arts-76%', 'Arts-77%', 'Arts-78%', 'Arts-79%', 'Arts-81%', 'Arts-82%', 'Arts-83%', 'Arts-84%', 'Arts-85%', 'Arts-86%', 'Arts-87%', 'Arts-88%', 'Arts-89%', 'Arts-90%', 'Arts-91%', 'Arts-92%', 'Commerce-45%', 'Commerce-46%', 'Commerce-47%', 'Commerce-48%', 'Commerce-49%', 'Commerce-50%', 'Commerce-51%', 'Commerce-52%', 'Commerce-53%', 'Commerce-54%', 'Commerce-55%', 'Commerce-56%', 'Commerce-57%', 'Commerce-58%', 'Commerce-59%', 'Commerce-60%', 'Commerce-61%', 'Commerce-62%', 'Commerce-63%', 'Commerce-64%', 'Commerce-65%', 'Commerce-66%', 'Commerce-67%', 'Commerce-68%', 'Commerce-69%', 'Commerce-70%', 'Commerce-71%', 'Commerce-72%', 'Commerce-73%', 'Commerce-74%', 'Commerce-75%', 'Commerce-76%', 'Commerce-77%', 'Commerce-78%', 'Commerce-79%', 'Commerce-80%', 'Commerce-81%', 'Commerce-82%', 'Commerce-83%', 'Commerce-84%', 'Commerce-85%', 'Commerce-86%', 'Commerce-87%', 'Commerce-88%', 'Commerce-89%', 'Commerce-90%', 'Commerce-91%', 'Commerce-92%', 'Commerce-93%', 'Commerce-94%', 'Commerce-95%', 'Science-50%', 'Science-51%', 'Science-52%', 'Science-53%', 'Science-54%', 'Science-55%', 'Science-56%', 'Science-57%', 'Science-58%', 'Science-59%', 'Science-60%', 'Science-61%', 'Science-62%', 'Science-63%', 'Science-64%', 'Science-65%', 'Science-66%', 'Science-67%', 'Science-68%', 'Science-69%', 'Science-70%', 'Science-71%', 'Science-72%', 'Science-73%', 'Science-74%', 'Science-75%', 'Science-76%', 'Science-77%', 'Science-78%', 'Science-79%', 'Science-80%', 'Science-81%', 'Science-82%', 'Science-83%', 'Science-84%', 'Science-85%', 'Science-86%', 'Science-87%', 'Science-88%', 'Science-89%', 'Science-90%', 'Science-91%', 'Science-92%', 'Science-93%', 'Science-94%', 'Science-95%', 'Science-96%', 'Science-97%', 'Science-98%', 'Science-99%', 'Vocational-43%', 'Vocational-44%', 'Vocational-45%', 'Vocational-46%', 'Vocational-47%', 'Vocational-50%', 'Vocational-52%', 'Vocational-53%', 'Vocational-55%', 'Vocational-56%', 'Vocational-57%', 'Vocational-58%', 'Vocational-60%', 'Vocational-61%', 'Vocational-62%', 'Vocational-65%', 'Vocational-68%', 'Vocational-69%', 'Vocational-70%', 'Vocational-71%', 'Vocational-72%', 'Vocational-73%', 'Vocational-74%', 'Vocational-75%', 'Vocational-76%', 'Vocational-78%', 'Vocational-79%', 'Vocational-83%', 'Vocational-88%', 'Vocational-89%'] },
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
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
    const [answers, setAnswers] = useState(() => {
        const initialAnswers = {};
        questions.forEach(q => { initialAnswers[q.id] = []; });
        return initialAnswers;
    });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [animation, setAnimation] = useState('slide-in');

    const currentQuestion = questions[currentQuestionIndex];
    const progress = ((currentQuestionIndex + 1) / questions.length) * 100;

    const translateOption = (option) => {
        const key = optionKeyMap[option];
        return key ? t(`quiz.options.${key}`, option) : option;
    };

    const handleSelectionChange = (questionId, option) => {
        setAnswers(prevAnswers => {
            const currentSelection = prevAnswers[questionId];
            const newSelection = currentSelection.includes(option)
                ? currentSelection.filter(item => item !== option)
                : [...currentSelection, option];
            return { ...prevAnswers, [questionId]: newSelection };
        });
    };

    const handleNext = () => {
        if (currentQuestionIndex < questions.length - 1) {
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

        try {
            // Convert structured answers into a single descriptive string for the NLP model
            // Example: "Favorite subjects: Math, Physics. Skills: Analysis. ..."
            const answersText = Object.entries(answers)
                .map(([key, value]) => {
                    // key is like "Q1", "Q2". value is array or string.
                    // Find the question text
                    const question = questions.find(q => q.id === key);
                    const qText = question ? question.text : key;
                    const valText = Array.isArray(value) ? value.join(", ") : value;
                    return `${qText}: ${valText}`;
                })
                .join(". ");

            console.log("Sending text to backend:", answersText);

            const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000'}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features: answersText }),
            });

            const data = await res.json();
            if (data.error) throw new Error(data.error);
            setResult({ recommendation: data.career });
        } catch (err) {
            console.error(err);
            setError(err.message || "Something went wrong. Please try again.");
        } finally {
            setLoading(false);
        }
    };


    const handleReset = () => {
        const initialAnswers = {};
        questions.forEach(q => { initialAnswers[q.id] = []; });
        setAnswers(initialAnswers);
        setCurrentQuestionIndex(0);
        setResult(null);
    };

    if (result) {
        return (
            <div className="career-quiz-container show-results">
                <div className="quiz-results">
                    <h3>{t('quiz.yourRecommendationTitle')}</h3>
                    <p><strong>{t('quiz.recommendedPath')}</strong> {result.recommendation}</p>
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
                    {questions.map((q, index) => (
                        <div key={q.id} className={`nav-item-top ${currentQuestionIndex === index ? 'active' : ''} ${answers[q.id]?.length > 0 ? 'answered' : ''}`}>
                            {answers[q.id]?.length > 0 ? <i className="fas fa-check-circle"></i> : index + 1}
                        </div>
                    ))}
                </div>

                <div className="quiz-card" key={currentQuestionIndex}>
                    <div className={`question-content ${animation}`}>
                        <h2 className="question-text">{t(`quiz.questions.${currentQuestion.id}.text`, currentQuestion.text)}</h2>
                        <div className="options-grid">
                            {currentQuestion.options.map(option => (
                                <button
                                    key={option}
                                    className={`option-card ${answers[currentQuestion.id]?.includes(option) ? 'selected' : ''}`}
                                    onClick={() => handleSelectionChange(currentQuestion.id, option)}
                                >
                                    {translateOption(option)}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="quiz-navigation">
                    <button onClick={handleBack} className="btn-nav btn-back" disabled={currentQuestionIndex === 0}>
                        <i className="fas fa-arrow-left"></i> {t('quiz.back')}
                    </button>
                    {currentQuestionIndex < questions.length - 1 ? (
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