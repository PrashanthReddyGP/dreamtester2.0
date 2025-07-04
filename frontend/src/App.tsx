import React, { useState, useMemo } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom'; 
import { getAppTheme } from './theme/theme';
import { MainLayout } from './layouts/MainLayout';
import { StrategyLab } from './pages/StrategyLab';
import { AnalysisHub } from './pages/AnalysisHub';
import { AnimatedPage } from './components/common/AnimatedPage';
import { loader } from '@monaco-editor/react';
import * as monaco from 'monaco-editor';

// This configures monaco's loader and theme before any component mounts.
loader.init().then((monacoInstance) => {
  // We get the colors from our MUI theme generator directly
  const darkTheme = getAppTheme('dark');

  monacoInstance.editor.defineTheme('app-dark-theme', {
    base: 'vs-dark',
    inherit: true,
    rules: [],
    colors: {
      'editor.background': darkTheme.palette.background.paper,
      // You can add more theme customizations here if you want
    },
  });
});


function App() {
  const [mode, setMode] = useState<'light' | 'dark'>('dark');

  const toggleTheme = () => {
    setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
  };

  // Memoize the theme to prevent re-creation on every render
  const theme = useMemo(() => getAppTheme(mode), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <Router> {/* Changed to Router for clarity */}
        <Routes>
          <Route element={<MainLayout mode={mode} toggleTheme={toggleTheme} />}>
            {/* Redirect the root path to the Strategy Lab */}
            <Route path="/" element={<Navigate to="/lab" replace />} />
            
            <Route path="/lab" element={<AnimatedPage><StrategyLab /></AnimatedPage>} />
            <Route path="/analysis" element={<AnimatedPage><AnalysisHub /></AnimatedPage>} />
            <Route path="/automation" element={<AnimatedPage><div>Automation Page</div></AnimatedPage>} />
          </Route>
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;