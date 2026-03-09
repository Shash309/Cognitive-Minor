import React, { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate, useOutletContext } from 'react-router-dom';

const CounselorChat = () => {
    const { studentEmail } = useParams();
    const { user } = useOutletContext() || {};
    const navigate = useNavigate();
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [sending, setSending] = useState(false);
    const [studentName, setStudentName] = useState('');
    const [topCareer, setTopCareer] = useState('');
    const messagesEndRef = useRef(null);

    const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

    const loadMessages = async () => {
        if (!studentEmail) return;
        try {
            const res = await fetch(
                `${apiBase}/api/counseling/messages?student_email=${encodeURIComponent(studentEmail)}`
            );
            const json = await res.json();
            setMessages(json.messages || []);
        } catch {
            // ignore
        }
    };

    const loadStudentInfo = async () => {
        if (!studentEmail) return;
        try {
            const res = await fetch(
                `${apiBase}/api/counselor/student-report?student_email=${encodeURIComponent(studentEmail)}`
            );
            const json = await res.json();
            setStudentName(json.student?.name || studentEmail.split('@')[0]);
            if (json.career_rankings && json.career_rankings.length > 0) {
                setTopCareer(json.career_rankings[0].career);
            }
        } catch {
            setStudentName(studentEmail.split('@')[0]);
        }
    };

    useEffect(() => {
        loadStudentInfo();
        loadMessages();
        // Poll for new messages every 5 seconds
        const id = setInterval(loadMessages, 5000);
        return () => clearInterval(id);
    }, [studentEmail]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || sending) return;
        setSending(true);
        try {
            await fetch(`${apiBase}/api/counseling/messages`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    student_email: studentEmail,
                    sender_role: 'counselor',
                    sender_name: user?.name || 'Counselor',
                    text: input.trim(),
                }),
            });
            setInput('');
            await loadMessages();
        } catch {
            // ignore
        } finally {
            setSending(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const getInitial = () => (studentName || 'S').charAt(0).toUpperCase();

    return (
        <div className="counselor-chat">
            <div className="chat-header">
                <button className="sr-back-btn" onClick={() => navigate(`/counselor/student/${encodeURIComponent(studentEmail)}`)}>
                    <i className="fas fa-arrow-left" />
                </button>
                <div className="chat-header-avatar">{getInitial()}</div>
                <div>
                    <div className="chat-header-name">{studentName}</div>
                    {topCareer && <div className="chat-header-career">Recommended: {topCareer}</div>}
                </div>
            </div>

            <div className="chat-messages">
                {messages.length === 0 ? (
                    <div className="chat-empty">
                        <p>No messages yet. Start the consultation by sending a message.</p>
                    </div>
                ) : (
                    messages.map((msg, idx) => (
                        <div key={idx} className={`chat-bubble ${msg.sender_role}`}>
                            <div className="chat-bubble-sender">
                                {msg.sender_role === 'counselor' ? (msg.sender_name || 'You') : studentName}
                            </div>
                            <div>{msg.text}</div>
                            <div className="chat-bubble-time">
                                {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
                            </div>
                        </div>
                    ))
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-area">
                <input
                    type="text"
                    placeholder="Type your advice or response..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={sending}
                />
                <button className="chat-send-btn" onClick={handleSend} disabled={sending || !input.trim()}>
                    {sending ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    );
};

export default CounselorChat;
