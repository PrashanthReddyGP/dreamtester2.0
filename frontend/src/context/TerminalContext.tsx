// src/context/TerminalContext.tsx
import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { websocketService } from '../services/websocketService';

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

// // Define the WebSocket URL
// const WS_URL = 'ws://127.0.0.1:8000/ws/results/';

export const TerminalContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);
  // const webSocketRef = useRef<WebSocket | null>(null);

  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
    setLogs(prevLogs => [...prevLogs, { timestamp, level, message }]);
  }, []);


  // --- 2. SUBSCRIBE TO THE WEBSOCKET SERVICE ---
  useEffect(() => {
    const subscription = websocketService.subscribe((data) => {
        const { type, payload } = data;

        // This context only cares about log-related and system messages
        if (type === 'log') {
            addLog(payload.level || 'INFO', payload.message);
        } else if (type === 'error') {
            addLog('ERROR', `ERROR: ${payload.message}`);
        } else if (type === 'batch_complete') {
            addLog('SUCCESS', `--- BATCH COMPLETE ---`);
        } else if (type === 'system') {
            // Handle connection status updates from the service
            if (payload.event === 'open') {
                setIsConnected(true);
                addLog('SYSTEM', 'Connection established. Waiting for logs...');
            } else if (payload.event === 'close') {
                setIsConnected(false);
                addLog('SYSTEM', 'Connection closed.');
            } else if (payload.event === 'error') {
                addLog('ERROR', payload.message || 'A WebSocket error occurred.');
            }
        }
    });

    // Clean up the subscription on unmount
    return () => {
        subscription.unsubscribe();
    };
  }, [addLog]); // addLog is a dependency of the effect


  const clearLogs = useCallback(() => {
    setLogs([]);
    addLog('SYSTEM', 'Terminal cleared.');
  }, [addLog]);

  const toggleTerminal = useCallback((forceState?: boolean) => {
    if (typeof forceState === 'boolean') {
      setIsTerminalOpen(forceState);
    } else {
      setIsTerminalOpen(prevState => !prevState);
    }
  }, []);
  
  // --- 3. SIMPLIFY connectToBatch ---
  // This function now just tells the central service to connect.
  // It also clears logs for the new session.
  const connectToBatch = useCallback((batchId: string) => {
    setLogs([]); // Clear logs for the new batch run
    addLog('SYSTEM', `Initializing connection for batch: ${batchId}...`);
    websocketService.connect(batchId);
  }, [addLog]);

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