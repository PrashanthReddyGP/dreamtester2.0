// src/context/TerminalContext.tsx
import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { websocketService } from '../services/websocketService';

// Define the shape of a single log entry
export interface LogEntry {
  timestamp: string;
  level: 'INFO' | 'ERROR' | 'SYSTEM' | 'SUCCESS';
  message: string;
}

interface TerminalContextType {
  logs: LogEntry[];
  isConnected: boolean;
  isTerminalOpen: boolean;
  setIsConnected: (status: boolean) => void; 
  addLog: (level: LogEntry['level'], message: string) => void;
  connectToBatch: (batchId: string) => void;
  clearLogs: () => void;
  toggleTerminal: (forceState?: boolean) => void;
}

const TerminalContext = createContext<TerminalContextType | undefined>(undefined);

export const TerminalContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);

  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
    setLogs(prevLogs => [...prevLogs, { timestamp, level, message }]);
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
    addLog('SYSTEM', 'Terminal cleared.');
  }, [addLog]);

  const toggleTerminal = useCallback((forceState?: boolean) => {
    setIsTerminalOpen(prevState => (typeof forceState === 'boolean' ? forceState : !prevState));
  }, []);
  
  const connectToBatch = useCallback((batchId: string) => {
    setLogs([]); // Clear logs for the new run
    addLog('SYSTEM', `Initializing connection for batch: ${batchId}...`);
    websocketService.connect(batchId);
  }, [addLog]);

  const value = {
    logs,
    isConnected,
    isTerminalOpen,
    setIsConnected, // Expose the new setter
    addLog,
    connectToBatch,
    clearLogs,
    toggleTerminal,
  };

  return <TerminalContext.Provider value={value}>{children}</TerminalContext.Provider>;
};

export const useTerminal = (): TerminalContextType => {
  const context = useContext(TerminalContext);
  if (context === undefined) {
    throw new Error('useTerminal must be used within a TerminalContextProvider');
  }
  return context;
};