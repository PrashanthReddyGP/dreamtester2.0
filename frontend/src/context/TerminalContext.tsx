// src/context/TerminalContext.tsx
import React, { createContext, useState, useContext, useRef, useCallback } from 'react';

// Define the shape of a single log entry
export interface LogEntry {
  timestamp: string;
  level: 'INFO' | 'ERROR' | 'SYSTEM' | 'SUCCESS';
  message: string;
}

// Define the shape of the context state
interface TerminalContextType {
  logs: LogEntry[];
  isConnected: boolean;
  isTerminalOpen: boolean;
  connectToBatch: (batchId: string) => void;
  clearLogs: () => void;
  toggleTerminal: (forceState?: boolean) => void;
}

// Create the context with a default value
const TerminalContext = createContext<TerminalContextType | undefined>(undefined);

// Define the WebSocket URL
const WS_URL = 'ws://127.0.0.1:8000/ws/results/';

export const TerminalContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const webSocketRef = useRef<WebSocket | null>(null);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);

  const addLog = (level: LogEntry['level'], message: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
    setLogs(prevLogs => [...prevLogs, { timestamp, level, message }]);
  };

  const clearLogs = useCallback(() => {
    setLogs([]);
    addLog('SYSTEM', 'Terminal cleared.');
  }, []);

  const toggleTerminal = useCallback((forceState?: boolean) => {
    if (typeof forceState === 'boolean') {
      setIsTerminalOpen(forceState);
    } else {
      setIsTerminalOpen(prevState => !prevState);
    }
  }, []);

  const connectToBatch = useCallback((batchId: string) => {
    // If there's an existing connection, close it first
    if (webSocketRef.current) {
      webSocketRef.current.close();
    }

    // Clear previous logs and start a new session
    setLogs([]);
    addLog('SYSTEM', `Initializing connection for batch: ${batchId}...`);
    
    const ws = new WebSocket(`${WS_URL}${batchId}`);
    webSocketRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      addLog('SYSTEM', 'Connection established. Waiting for logs...');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const { type, payload } = data;

        // Match the message types sent from the Python backend
        if (type === 'log') {
            addLog(payload.level || 'INFO', payload.message);
        } else if (type === 'error') {
            addLog('ERROR', `ERROR: ${payload.message}`);
        } else if (type === 'success') {
            addLog('SUCCESS', payload.message);
        } else if (type === 'batch_complete') {
            addLog('SUCCESS', `--- BATCH COMPLETE ---`);
            ws.close();
        }

      } catch (error) {
        addLog('ERROR', 'Failed to parse incoming WebSocket message.');
        console.error('WebSocket message parse error:', error);
      }
    };

    ws.onerror = (event) => {
      addLog('ERROR', 'A WebSocket error occurred. See console for details.');
      console.error('WebSocket Error:', event);
    };

    ws.onclose = () => {
      setIsConnected(false);
      if (webSocketRef.current === ws) { // Ensure it's not an old socket closing
          addLog('SYSTEM', 'Connection closed.');
          webSocketRef.current = null;
      }
    };

  }, []);

  return (
    <TerminalContext.Provider value={{ logs, isConnected, connectToBatch, clearLogs, isTerminalOpen, toggleTerminal }}>
      {children}
    </TerminalContext.Provider>
  );
};

// Custom hook to easily use the context
export const useTerminal = () => {
  const context = useContext(TerminalContext);
  if (context === undefined) {
    throw new Error('useTerminal must be used within a TerminalContextProvider');
  }
  return context;
};