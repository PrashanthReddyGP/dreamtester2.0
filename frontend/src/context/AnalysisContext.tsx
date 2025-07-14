// src/context/AnalysisContext.tsx
import React, { createContext, useState, useContext, useCallback } from 'react';
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

  // This single method handles both batch and optimization results
  const addResult = useCallback((newResult: StrategyResult) => {
    setResults(prev => {
      // This logic correctly handles portfolio updates by replacing the result
      const existingIndex = prev.findIndex(r => r.strategy_name === newResult.strategy_name);
      if (existingIndex > -1) {
        const updatedResults = [...prev];
        updatedResults[existingIndex] = newResult;
        return updatedResults;
      }
      // For all other cases (including all optimization runs), it just adds the new result
      return [...prev, newResult];
    });
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setIsComplete(false);
  }, []);

  const markComplete = useCallback(() => {
    setIsComplete(true);
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