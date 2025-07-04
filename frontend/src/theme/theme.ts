// Corrected import statements
import { createTheme } from '@mui/material/styles';
import type { PaletteOptions } from '@mui/material/styles';

// Define our color palettes with explicit typing
const darkPalette: PaletteOptions = {
  mode: 'dark',
  primary: { main: '#4A5DFF' },
  secondary: { main: '#FFC837' },
  background: {
    default: '#101118',
    paper: '#1A1C26',
  },
  text: {
    primary: '#E1E1E6',
    secondary: '#8D909F',
  },
  success: { main: '#00D09B' },
  error: { main: '#FF455F' },
  divider: '#2D2F3D',
};

const lightPalette: PaletteOptions = {
  mode: 'light',
  primary: { main: '#4A5DFF' },
  secondary: { main: '#FFC837' },
  background: {
    default: '#F7F7FA',
    paper: '#FFFFFF',
  },
  text: {
    primary: '#1A1C26',
    secondary: '#6B7280',
  },
  success: { main: '#00D09B' },
  error: { main: '#FF455F' },
  divider: '#E5E7EB',
};

// The function to create a theme based on the mode
export const getAppTheme = (mode: 'light' | 'dark') =>
  createTheme({
    palette: mode === 'dark' ? darkPalette : lightPalette,
    typography: {
      fontFamily: ['Inter', 'sans-serif'].join(','),
      h1: { fontSize: '24px', fontWeight: 700 },
      h2: { fontSize: '18px', fontWeight: 600 },
      body1: { fontSize: '14px', fontWeight: 400 },
      button: { textTransform: 'none', fontWeight: 600 },
    },
    components: {
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundImage: 'none', // Important for MUI v5
                }
            }
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                }
            }
        }
    }
  });