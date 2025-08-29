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


// --- 1. THE WebSocketManager COMPONENT ---
// This component does not render any UI. Its sole purpose is to listen
// to the WebSocket service and route data to the correct context.
const WebSocketManager: React.FC = () => {
    const { addLog, setIsConnected } = useTerminal();
    // --- Get setBatchConfig from useAnalysis ---
    const { addResult, markComplete, setBatchConfig } = useAnalysis();

    useEffect(() => {
        // Subscribe to the central WebSocket service. Use a key for ultimate stability.
        const subscription = websocketService.subscribe('main-app-listener', (data: any) => {
            if (!data || !data.type) return; 
            const { type, payload } = data;

            switch (type) {
                
                case 'batch_info':
                    setBatchConfig(payload.config);
                    addLog('SYSTEM', `Starting test: ${payload.config.test_type || 'Standard'}`);
                    break;

                case 'log':
                    addLog(payload.level || 'INFO', payload.message);
                    break;
                case 'error': {
                    // The backend might send a simple string or an object with a message property
                    const errorMessage = typeof payload === 'string' ? payload : payload.message;
                    addLog('ERROR', `ERROR: ${errorMessage}`);
                    break;
                }
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

        return () => {
            subscription.unsubscribe();
        };
    // --- CHANGE 3: Add setBatchConfig to the dependency array ---
    }, [addLog, setIsConnected, addResult, markComplete, setBatchConfig]);

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
      <AnalysisContextProvider>
        <TerminalContextProvider>
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
        </TerminalContextProvider>
      </AnalysisContextProvider>
    </AppContext.Provider>
  );
}

export default App;
