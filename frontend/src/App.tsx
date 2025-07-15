import React, { useState, useMemo, useEffect } from 'react';
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
import { TerminalContextProvider, useTerminal } from './context/TerminalContext';
import { AnalysisContextProvider, useAnalysis } from './context/AnalysisContext';
import { websocketService } from './services/websocketService';



// --- 1. THE NEW WebSocketManager COMPONENT ---
// This component does not render any UI. Its sole purpose is to listen
// to the WebSocket service and route data to the correct context.
const WebSocketManager: React.FC = () => {
    const { addLog, setIsConnected } = useTerminal();
    const { addResult, markComplete } = useAnalysis();

    useEffect(() => {
        // Subscribe to the central WebSocket service
        const subscription = websocketService.subscribe((data: any) => {
            if (!data || !data.type) return; // Add a safety check
            const { type, payload } = data;

            // This switch statement is the router for all incoming data
            switch (type) {
                case 'heartbeat':
                  // This message's only job is to generate traffic.
                  // We receive it and do absolutely nothing with it.
                  // You can log it for debugging if you want.
                  // console.log('Heartbeat received.'); 
                  break;
                // --- END OF CHANGE ---

                case 'log':
                    addLog(payload.level || 'INFO', payload.message);
                    break;
                case 'error':
                    addLog('ERROR', `ERROR: ${payload.message}`);
                    break;
                case 'strategy_result':
                    addResult(payload);
                    break;
                case 'batch_complete':
                    addLog('SUCCESS', '--- BATCH COMPLETE ---');
                    markComplete();
                    break;
                case 'system':
                    if (payload.event === 'open') {
                        setIsConnected(true);
                        addLog('SYSTEM', 'Connection established. Waiting for logs...');
                    } else if (payload.event === 'close') {
                        setIsConnected(false);
                        addLog('SYSTEM', 'Connection closed.');
                    } else if (payload.event === 'error') {
                        addLog('ERROR', payload.message || 'A WebSocket error occurred.');
                    }
                    break;
                default:
                    console.warn("Unhandled WebSocket message type:", type);
            }
        });

        // Unsubscribe when the component unmounts to prevent memory leaks
        return () => {
            subscription.unsubscribe();
        };
    // The dependency array ensures this effect always has the latest versions of the context methods
    }, [addLog, setIsConnected, addResult, markComplete]);

    return null; // This component renders nothing
};

loader.init().then((monacoInstance) => {
  const darkTheme = getAppTheme('dark');
  monacoInstance.editor.defineTheme('app-dark-theme', {
    base: 'vs-dark',
    inherit: true,
    rules: [],
    colors: { 'editor.background': darkTheme.palette.background.paper },
  });
});

const API_URL = 'http://127.0.0.1:8000'; // FastAPI backend URL

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

  const contextValue = { settings, isLoading, error, saveApiKeys };

  return (
    <AppContext.Provider value={contextValue}>
      <TerminalContextProvider>
        <AnalysisContextProvider>
          <WebSocketManager />

          <ThemeProvider theme={theme}>
            <Router>
              <Routes>
                <Route element={<MainLayout mode={mode} toggleTheme={toggleTheme} />}>
                  <Route path="/" element={<Navigate to="/lab" replace />} />
                  <Route path="/lab" element={<AnimatedPage><StrategyLab /></AnimatedPage>} />
                  <Route path="/analysis" element={<AnimatedPage><AnalysisHub /></AnimatedPage>} />
                  <Route path="/automation" element={<div style={{display:'flex', justifyContent:'center', alignItems:'center', width:'100%'}}><AnimatedPage>Automation Page</AnimatedPage></div>} />
                </Route>
              </Routes>
            </Router>
          </ThemeProvider>
        </AnalysisContextProvider>
      </TerminalContextProvider>
    </AppContext.Provider>
  );
}

export default App;
