// src/context/AnalysisContext.tsx
import React, { createContext, useState, useContext, useCallback, useRef, useEffect } from 'react';
import type {ReactNode} from 'react';
import type { StrategyResult, MLResult } from '../services/api';

// This is the union type we will use throughout the file
type AnalysisResult = StrategyResult | MLResult;

interface AnalysisContextType {
  results: AnalysisResult[]; // Use the union type
  isComplete: boolean;
  batchConfig: BatchConfig | null;
  addResult: (result: AnalysisResult) => void; // Use the union type
  clearResults: () => void;
  markComplete: () => void;
  setBatchConfig: (config: BatchConfig) => void; 
}

export interface BatchConfig {
  test_type: string;
  [key: string]: any; 
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {

  const [results, setResults] = useState<(StrategyResult | MLResult)[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const resultsBuffer = useRef<StrategyResult[]>([]);
  const animationFrameId = useRef<number | null>(null);
  const [batchConfig, setBatchConfigState] = useState<BatchConfig | null>(null);

  // This is the core function to flush the buffer to the React state
  const flushBuffer = useCallback(() => {
    if (resultsBuffer.current.length > 0) {
      setResults(prevResults => [...prevResults, ...resultsBuffer.current]);
      resultsBuffer.current = []; // Clear the buffer after flushing
    }
    // Mark the update as complete by nullifying the ID
    animationFrameId.current = null;
  }, []);

  // addResult now uses throttling with requestAnimationFrame
  const addResult = useCallback((newResult: StrategyResult) => {
    resultsBuffer.current.push(newResult);

    // If an update is not already scheduled for the next frame, schedule one.
    if (animationFrameId.current === null) {
      animationFrameId.current = requestAnimationFrame(flushBuffer);
    }
  }, [flushBuffer]);

  const clearResults = useCallback(() => {
    if (animationFrameId.current !== null) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }
    
    resultsBuffer.current = [];
    setResults([]);
    setIsComplete(false);
    setBatchConfigState(null);
  }, []);

  const markComplete = useCallback(() => {
    // If an update is scheduled, cancel it because we are about to do a final, immediate flush.
    if (animationFrameId.current !== null) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }
    
    // Immediately flush any remaining items in the buffer
    flushBuffer();
    
    // Then mark the process as complete
    setIsComplete(true);
  }, [flushBuffer]);

  const setBatchConfig = useCallback((config: BatchConfig) => {
    setBatchConfigState(config);
  }, []);

  useEffect(() => {
    return () => {
      if (animationFrameId.current !== null) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, []);

  const value = {
    results,
    isComplete,
    batchConfig,
    addResult,
    clearResults,
    markComplete,
    setBatchConfig,
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