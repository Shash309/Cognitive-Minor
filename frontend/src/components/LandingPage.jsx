import React, { useState } from 'react';
import './LandingPage.css';
import { useTranslation } from 'react-i18next';

const LandingPage = ({ onLogin }) => {
  const [isSignInActive, setIsSignInActive] = useState(true);
  const [selectedRole, setSelectedRole] = useState('student');
  const { t } = useTranslation();

  const isCounselorEnabled = import.meta.env.VITE_ENABLE_COUNSELLOR_FEATURE === "true";

  // Counselor extra fields
  const [experience, setExperience] = useState('');
  const [specialization, setSpecialization] = useState('');
  const [linkedin, setLinkedin] = useState('');

  const handleAuth = async (e, type) => {
    e.preventDefault();
    const form = e.target;

    const email = form.email.value;
    const name = form.name ? form.name.value : 'User';

    // We only strictly require password if the form actually has it displayed
    const password = form.password ? form.password.value : '';
    const confirmPassword = form.confirmPassword ? form.confirmPassword.value : '';

    if (selectedRole === 'counselor') {
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
      try {
        if (type === 'register') {
          const res = await fetch(`${apiBase}/api/counselor/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              name,
              email,
              password,
              years_of_experience: experience,
              specialization,
              linkedin,
            }),
          });
          const data = await res.json();
          if (!res.ok) {
            alert(data.error || 'Registration failed.');
            return;
          }
          onLogin({ ...data, role: 'counselor' });
        } else {
          const res = await fetch(`${apiBase}/api/counselor/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
          });
          const data = await res.json();
          if (!res.ok) {
            alert(data.error || 'Invalid credentials.');
            return;
          }
          onLogin({ ...data, role: 'counselor' });
        }
      } catch {
        alert('Server error. Please try again.');
      }
      return;
    }

    // Student auth
    if (type === 'register') {
      if (confirmPassword && password !== confirmPassword) {
        alert('Passwords do not match');
        return;
      }
      const phone = form.phone ? form.phone.value : '';
      const age = form.age ? form.age.value : '';
      const gender = form.gender ? form.gender.value : '';
      const location = form.location ? form.location.value : '';

      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

      try {
        const res = await fetch(`${apiBase}/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, email, password, phone, age, gender, location })
        });

        const data = await res.json();

        if (!res.ok) {
          console.error("Backend Error Response:", data);
          alert(data.error || 'Server validation failed for Registration');
          return;
        }

        console.log("Registration Success Response:", data);

        // Keep local cache up to date for backwards UI compatibility
        const users = JSON.parse(localStorage.getItem('career_app_users') || '{}');
        users[email] = { name, email, role: 'student' };
        localStorage.setItem('career_app_users', JSON.stringify(users));

        onLogin({ email, name, role: 'student' });

      } catch (err) {
        console.error("Registration fetch error:", err);
        alert("Failed to reach the backend server. Make sure Flask is running!");
      }
    } else {
      // Student Login
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
      try {
        const res = await fetch(`${apiBase}/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password })
        });
        
        const data = await res.json();
        
        if (!res.ok) {
          alert(data.error || 'Invalid credentials.');
          return;
        }
        
        onLogin({ email: data.user.email, name: data.user.name, role: 'student' });
      } catch (err) {
        console.error("Login fetch error:", err);
        alert("Failed to reach the backend server. Make sure Flask is running!");
      }
    }
  };

  return (
    <div className="landing-container">
      <div className={`auth-card-wrapper ${isSignInActive ? '' : 'right-panel-active'}`}>
        {/* ─── Sign Up Form ─── */}
        <div className="form-container sign-up-container">
          <form onSubmit={(e) => handleAuth(e, 'register')}>
            <h1>{t('landing.createAccount')}</h1>

            {/* Role Selector */}
            <div className="role-selector">
              <button
                type="button"
                className={`role-card ${selectedRole === 'student' ? 'selected' : ''}`}
                onClick={() => setSelectedRole('student')}
              >
                <span className="role-icon">🎓</span>
                <span className="role-label">Student</span>
                <span className="role-desc">AI-powered career recommendations</span>
              </button>
              {isCounselorEnabled && (
                <button
                  type="button"
                  className={`role-card ${selectedRole === 'counselor' ? 'selected' : ''}`}
                  onClick={() => setSelectedRole('counselor')}
                >
                  <span className="role-icon">🧠</span>
                  <span className="role-label">Counselor</span>
                  <span className="role-desc">Guide students with AI insights</span>
                </button>
              )}
            </div>

            <input type="text" name="name" placeholder={t('landing.name') || 'Name'} required />
            <input type="email" name="email" placeholder={t('landing.email') || 'Email'} required />

            {selectedRole === 'student' && (
              <div className="student-fields-grid">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>
                  <span style={{ whiteSpace: 'nowrap', opacity: 0.9 }}>+91</span>
                  <input
                    type="tel"
                    name="phone"
                    placeholder="10-digit number"
                    required
                    inputMode="numeric"
                    pattern="^[0-9]{10}$"
                    maxLength={10}
                    style={{ width: 'auto', flex: 1 }}
                  />
                </div>
                <input type="number" name="age" placeholder="Age" min="10" max="100" required />

                <select name="gender" className="specialization-select" required>
                  <option value="">Select Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>

                <input type="text" name="location" placeholder="Location (City)" required />
              </div>
            )}

            {selectedRole === 'counselor' && (
              <div className="counselor-fields">
                <input
                  type="number"
                  min="0"
                  max="50"
                  placeholder="Years of Experience"
                  value={experience}
                  onChange={(e) => setExperience(e.target.value)}
                />
                <select
                  value={specialization}
                  onChange={(e) => setSpecialization(e.target.value)}
                  className="specialization-select"
                >
                  <option value="">Select Specialization</option>
                  <option value="Career Counseling">Career Counseling</option>
                  <option value="Psychology">Psychology</option>
                  <option value="Academic Guidance">Academic Guidance</option>
                  <option value="University Admissions">University Admissions</option>
                </select>
                <input
                  type="url"
                  placeholder="LinkedIn Profile (optional)"
                  value={linkedin}
                  onChange={(e) => setLinkedin(e.target.value)}
                />
              </div>
            )}

            {/* Render password strictly for all users (must be filled before confirm password) */}
            <input type="password" name="password" placeholder={t('landing.password') || 'Password'} required />
            <input
              type="password"
              name="confirmPassword"
              placeholder="Confirm Password"
              required
            />

            <button type="submit">{t('landing.signUp')}</button>
          </form>
        </div>

        {/* ─── Sign In Form ─── */}
        <div className="form-container sign-in-container">
          <form onSubmit={(e) => handleAuth(e, 'login')}>
            <h1>{t('landing.signInTitle')}</h1>

            {/* Role Selector */}
            <div className="role-selector">
              <button
                type="button"
                className={`role-card ${selectedRole === 'student' ? 'selected' : ''}`}
                onClick={() => setSelectedRole('student')}
              >
                <span className="role-icon">🎓</span>
                <span className="role-label">Student</span>
              </button>
              {isCounselorEnabled && (
                <button
                  type="button"
                  className={`role-card ${selectedRole === 'counselor' ? 'selected' : ''}`}
                  onClick={() => setSelectedRole('counselor')}
                >
                  <span className="role-icon">🧠</span>
                  <span className="role-label">Counselor</span>
                </button>
              )}
            </div>

            <input type="email" name="email" placeholder={t('landing.email')} required />
            <input type="password" name="password" placeholder={t('landing.password')} required />
            <button type="submit">{t('landing.signIn')}</button>
          </form>
        </div>

        {/* ─── Overlay ─── */}
        <div className="overlay-container">
          <div className="overlay">
            <div className="overlay-panel overlay-left">
              <h1>{t('landing.welcomeBack')}</h1>
              <p>{t('landing.keepConnected')}</p>
              <button className="ghost" onClick={() => setIsSignInActive(true)}>{t('landing.signIn')}</button>
            </div>
            <div className="overlay-panel overlay-right">
              <h1>{t('landing.helloFriend')}</h1>
              <p>{t('landing.startJourney')}</p>
              <button className="ghost" onClick={() => setIsSignInActive(false)}>{t('landing.signUp')}</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;