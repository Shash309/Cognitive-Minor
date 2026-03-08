import { createContext } from 'react';

export const THEME_KEY = 'user_theme_preference';

export const ThemeContext = createContext({
  theme: 'auto',
  resolvedTheme: 'light',
  setTheme: () => {},
});

export const getAutoTheme = () => {
  const now = new Date();
  const hour = now.getHours();
  // 07:00–18:59 → light, otherwise dark
  return hour >= 7 && hour < 19 ? 'light' : 'dark';
};

