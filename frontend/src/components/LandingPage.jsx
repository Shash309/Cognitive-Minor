import React, { useState } from 'react';
import './LandingPage.css';
import { useTranslation } from 'react-i18next';

const LandingPage = ({ onLogin }) => {
  const [isSignInActive, setIsSignInActive] = useState(true);
  const { t } = useTranslation();

  const handleAuth = (e, type) => {
    e.preventDefault();
    const form = e.target;
    
    // Auth logic using localStorage user registry
    const email = form.email.value;
    const password = form.password.value;
    const name = form.name ? form.name.value : 'User';

    // Fetch existing users from localStorage
    const users = JSON.parse(localStorage.getItem('career_app_users') || '{}');

    if (type === 'register') {
      if (users[email]) {
        alert(t('User already exists. Please sign in.'));
        return;
      }
      users[email] = { name, email, password };
      localStorage.setItem('career_app_users', JSON.stringify(users));
      onLogin({ email, name });
    } else { // login
      const user = users[email];
      if (!user || user.password !== password) {
        alert(t('Invalid credentials. Please try again.'));
        return;
      }
      onLogin({ email, name: user.name });
    }
  };

  return (
    <div className="landing-container">
      <div className={`auth-card-wrapper ${isSignInActive ? '' : 'right-panel-active'}`}>
        <div className="form-container sign-up-container">
          <form onSubmit={(e) => handleAuth(e, 'register')}>
            <h1>{t('landing.createAccount')}</h1>
            <div className="social-container">
              <a href="#" className="social"><i className="fab fa-google-plus-g"></i></a>
              <a href="#" className="social"><i className="fab fa-facebook-f"></i></a>
              <a href="#" className="social"><i className="fab fa-github"></i></a>
              <a href="#" className="social"><i className="fab fa-linkedin-in"></i></a>
            </div>
            <span>{t('landing.orUseEmail')}</span>
            <input type="text" name="name" placeholder={t('landing.name')} required />
            <input type="email" name="email" placeholder={t('landing.email')} required />
            <input type="password" name="password" placeholder={t('landing.password')} required />
            <button type="submit">{t('landing.signUp')}</button>
          </form>
        </div>

        <div className="form-container sign-in-container">
          <form onSubmit={(e) => handleAuth(e, 'login')}>
            <h1>{t('landing.signInTitle')}</h1>
            <div className="social-container">
              <a href="#" className="social"><i className="fab fa-google-plus-g"></i></a>
              <a href="#" className="social"><i className="fab fa-facebook-f"></i></a>
              <a href="#" className="social"><i className="fab fa-github"></i></a>
              <a href="#" className="social"><i className="fab fa-linkedin-in"></i></a>
            </div>
            <span>{t('landing.orUseAccount')}</span>
            <input type="email" name="email" placeholder={t('landing.email')} required />
            <input type="password" name="password" placeholder={t('landing.password')} required />
            <a href="#" style={{ fontSize: '0.8rem', marginTop: '8px', color: 'var(--text-muted)' }}>{t('landing.forgotPassword')}</a>
            <button type="submit">{t('landing.signIn')}</button>
          </form>
        </div>

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