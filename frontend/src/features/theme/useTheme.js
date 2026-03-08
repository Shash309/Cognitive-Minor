import { useContext } from 'react';
import { ThemeContext } from './themeContextBase';

export const useTheme = () => useContext(ThemeContext);

