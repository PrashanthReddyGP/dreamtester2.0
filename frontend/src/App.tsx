import React, { useState, useMemo, useEffect, useRef } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { getAppTheme } from './theme/theme';
import { MainLayout } from './layouts/MainLayout';
import { StrategyLab } from './pages/StrategyLab';
import { AnalysisHub } from './pages/AnalysisHub';
import { AnimatedPage } from './components/common/AnimatedPage';
import { loader } from '@monaco-editor/react';
import AppContext from './context/AppContext';
import type { SettingsState, ApiKeySet } from './context/types';
import { clearLatestBacktestResult, getLatestBacktestResult } from './services/api'; // Import the service
import type { BacktestResultPayload } from './services/api';

loader.init().then((monacoInstance) => {
  const darkTheme = getAppTheme('dark');
  monacoInstance.editor.defineTheme('app-dark-theme', {
    base: 'vs-dark',
    inherit: true,
    rules: [],
    colors: { 'editor.background': darkTheme.palette.background.paper },
  });
});

const API_URL = 'http://127.0.0.1:8000'; // Your FastAPI backend URL

function App() {
  const [mode, setMode] = useState<'light' | 'dark'>('dark');

  const [settings, setSettings] = useState<SettingsState>({});
  const [isLoading, setIsLoading] = useState(true);

  const [error, setError] = useState<string | null>(null);

  const toggleTheme = () => {
    setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
  };

  const theme = useMemo(() => getAppTheme(mode), [mode]);

  const loadInitialData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const exchanges = ['binance']; 

      const promises = exchanges.map((ex: string) =>
        fetch(`${API_URL}/api/keys/${ex}`).then(res =>
          res.ok ? res.json() as Promise<ApiKeySet> : null
        )
      );

      const results = await Promise.all(promises);
      
      const newSettings: SettingsState = {};
      
      results.forEach(keys => {
        if (keys) {
          newSettings[keys.exchange] = { apiKey: keys.apiKey, apiSecret: keys.apiSecret };
        }
      });

      setSettings(newSettings);

    } catch (err) {
      console.error("Failed to fetch initial settings:", err);
      setError("Could not connect to the backend.");
    } finally {
      setIsLoading(false);
    }
  };

  const saveApiKeys = async (exchange: string, apiKey: string, apiSecret: string) => {
    try {
      const response = await fetch(`${API_URL}/api/keys/${exchange}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiKey, apiSecret }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to save keys on the server.');
      }
      
      // After saving, reload the data to ensure UI is in sync
      await loadInitialData(); 
      return await response.json();

    } catch (err) {
      console.error("Error saving keys:", err);
      throw err; 
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  const [latestBacktest, setLatestBacktest] = useState<BacktestResultPayload | null>(null);
  const [isBacktestLoading, setIsBacktestLoading] = useState(false); // Start as false
  const [backtestError, setBacktestError] = useState<string | null>(null);

  const isPollingRef = useRef(false);

  const fetchLatestResults = () => {
    console.log("Starting to poll for latest backtest results...");

    // Guard clause to prevent multiple concurrent polling loops
    if (isPollingRef.current) {
      console.log("Polling is already active. Exiting.");
      return;
    }
    
    // Set the initial state for the loading process
    setIsBacktestLoading(true);
    setBacktestError(null);
    setLatestBacktest(null);

    isPollingRef.current = true;

    const poll = async () => {
        // We define a variable to check if the component is still mounted and waiting
        // This is a way to access the "current" state inside the async function
        if (!isPollingRef.current) {
          console.log("Polling was cancelled.");
          return;
        }

        try {
            const result = await getLatestBacktestResult();
            
            if (result) {
                // SUCCESS: We got the data
                console.log("Successfully fetched backtest results.");
                setLatestBacktest(result);
                setIsBacktestLoading(false);
                isPollingRef.current = false; 
            } else {
              // No result yet, poll again
              setTimeout(poll, 5000);
            }

        } catch (err) {
            console.error("Polling failed with an error:", err);
            setBacktestError("Failed to load backtest results.");
            setIsBacktestLoading(false);
            isPollingRef.current = false;
        }
    };

    poll(); // Start the polling
  };

  const clearLatestBacktest = () => {
    console.log("Clearing latest backtest data from global state...");
    setLatestBacktest(null);
    setBacktestError(null);
    setIsBacktestLoading(true); // Set to true to immediately show the loading spinner on the Analysis page
    isPollingRef.current = false;
  };

  const contextValue = {
    settings,
    isLoading,
    error,
    saveApiKeys,
    latestBacktest,
    isBacktestLoading,
    backtestError,
    fetchLatestResults,
    clearLatestBacktest,
  };

  return (
    <AppContext.Provider value={contextValue}>
      <ThemeProvider theme={theme}>
        <Router>
          <Routes>
            <Route element={<MainLayout mode={mode} toggleTheme={toggleTheme} />}>
              <Route path="/" element={<Navigate to="/lab" replace />} />
              <Route path="/lab" element={<AnimatedPage><StrategyLab /></AnimatedPage>} />
              <Route path="/analysis" element={<AnimatedPage><AnalysisHub /></AnimatedPage>} />
              <Route path="/automation" element={<div  style={{display:'flex', justifyContent:'center', alignItems:'center', width:'100%'}}><AnimatedPage>Automation Page</AnimatedPage></div>} />
            </Route>
          </Routes>
        </Router>
      </ThemeProvider>
    </AppContext.Provider>
  );
}

export default App;