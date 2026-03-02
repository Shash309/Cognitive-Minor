import React, { createContext, useContext, useEffect, useState } from 'react';

const THEME_KEY = 'user_theme_preference';

const ThemeContext = createContext({
  theme: 'auto',
  resolvedTheme: 'light',
  setTheme: () => {},
});

const getAutoTheme = () => {
  const now = new Date();
  const hour = now.getHours();
  // 07:00–18:59 → light, otherwise dark
  if (hour >= 7 && hour < 19) {
    return 'light';
  }
  return 'dark';
};

export const ThemeProvider = ({ children }) => {
  const [theme, setThemeState] = useState('auto');
  const [resolvedTheme, setResolvedTheme] = useState('light');

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(THEME_KEY);
      if (stored === 'light' || stored === 'dark' || stored === 'auto') {
        setThemeState(stored);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    const effective = theme === 'auto' ? getAutoTheme() : theme;
    setResolvedTheme(effective);

    const root = document.documentElement;
    root.classList.remove('theme-light', 'theme-dark');
    root.classList.add(effective === 'dark' ? 'theme-dark' : 'theme-light');
  }, [theme]);

  const setTheme = (next) => {
    setThemeState(next);
    try {
      window.localStorage.setItem(THEME_KEY, next);
    } catch {
      // ignore
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);

