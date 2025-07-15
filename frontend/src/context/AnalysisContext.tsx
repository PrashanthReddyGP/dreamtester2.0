// src/context/AnalysisContext.tsx
import React, { createContext, useState, useContext, useCallback, useRef, useEffect } from 'react';
import type {ReactNode} from 'react';
import type { StrategyResult } from '../services/api';

interface AnalysisContextType {
  results: StrategyResult[];
  isComplete: boolean;
  addResult: (result: StrategyResult) => void;
  clearResults: () => void;
  markComplete: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [results, setResults] = useState<StrategyResult[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const resultsBuffer = useRef<StrategyResult[]>([]);
  const updateTimer = useRef<NodeJS.Timeout | null>(null);

  const addResult = useCallback((newResult: StrategyResult) => {
    // Instead of calling setResults directly, add the new result to our buffer.
    resultsBuffer.current.push(newResult);

    // If a timer is already set, clear it. We'll set a new one.
    // This is "debouncing" - we only update after a period of inactivity.
    if (updateTimer.current) {
      clearTimeout(updateTimer.current);
    }

    // Set a timer to update the actual React state in a batch.
    // 100ms is a good value - it feels real-time but prevents overwhelming the browser.
    updateTimer.current = setTimeout(() => {
      setResults(prevResults => {
        // Create a new array with the previous results and everything in the buffer.
        const newBatch = [...prevResults, ...resultsBuffer.current];
        // Clear the buffer for the next batch.
        resultsBuffer.current = [];
        return newBatch;
      });
      // Clear the timer ID
      updateTimer.current = null;
    }, 100);

  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setIsComplete(false);
    // Also clear any pending updates in the buffer
    if (updateTimer.current) {
      clearTimeout(updateTimer.current);
    }
    resultsBuffer.current = [];
  }, []);

  const markComplete = useCallback(() => {
    // When the batch is complete, make sure any remaining items in the buffer are flushed.
    if (updateTimer.current) {
      clearTimeout(updateTimer.current);
    }
    setResults(prevResults => [...prevResults, ...resultsBuffer.current]);
    resultsBuffer.current = [];
    
    setIsComplete(true);
  }, []);

  useEffect(() => {
    return () => {
      if (updateTimer.current) {
        clearTimeout(updateTimer.current);
      }
    };
  }, []);


  const value = {
    results,
    isComplete,
    addResult,
    clearResults,
    markComplete,
  };

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
};

export const useAnalysis = (): AnalysisContextType => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisContextProvider');
  }
  return context;
};