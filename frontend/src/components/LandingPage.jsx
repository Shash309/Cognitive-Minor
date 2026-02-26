import React, { useState } from 'react';
import './LandingPage.css';
import { useTranslation } from 'react-i18next';

const LandingPage = ({ onLogin }) => {
  const [isSignInActive, setIsSignInActive] = useState(true);
  const { t } = useTranslation();

  const handleAuth = (e, type) => {
    e.preventDefault();
    const form = e.target;
    // Simple mock auth logic for UI demo
    const email = form.email ? form.email.value : '';
    onLogin(email || (type === 'login' ? 'Explorer' : 'New User'));
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