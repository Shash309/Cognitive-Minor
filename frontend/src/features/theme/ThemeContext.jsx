import React, { useEffect, useState } from 'react';
import { THEME_KEY, ThemeContext, getAutoTheme } from './themeContextBase';

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
    root.setAttribute('data-theme', effective);
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

