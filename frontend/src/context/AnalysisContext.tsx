// src/context/AnalysisContext.tsx
import React, { createContext, useState, useContext, useCallback, useRef, useEffect, useMemo } from 'react';
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
    setSingleResult: (result: AnalysisResult, config?: BatchConfig) => void;
}

export interface BatchConfig {
  test_type: string;
  [key: string]: any; 
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {

    const [results, setResults] = useState<AnalysisResult[]>([]); // Use the union type
    const [isComplete, setIsComplete] = useState(false);

    const resultsBuffer = useRef<AnalysisResult[]>([]); 
    
    const animationFrameId = useRef<number | null>(null);
    const [batchConfig, setBatchConfigState] = useState<BatchConfig | null>(null);

    // This is the core function to flush the buffer to the React state
    const flushBuffer = useCallback(() => {
      if (resultsBuffer.current.length > 0) {
        setResults(prevResults => [...prevResults, ...resultsBuffer.current]);
        resultsBuffer.current = [];
      }
      animationFrameId.current = null;
    }, []);

    // addResult now uses throttling with requestAnimationFrame
    const addResult = useCallback((newResult: AnalysisResult) => {
      resultsBuffer.current.push(newResult);

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

    const setSingleResult = useCallback((result: AnalysisResult, config: BatchConfig = { test_type: 'single_run' }) => {
        // First, perform a full clear to stop any pending streaming operations
        if (animationFrameId.current !== null) {
            cancelAnimationFrame(animationFrameId.current);
            animationFrameId.current = null;
        }
        resultsBuffer.current = [];
        
        // Now, set all state synchronously. React will batch these updates.
        setResults([result]);
        setBatchConfigState(config);
        setIsComplete(true);
    }, []);

    const markComplete = useCallback(() => {
      if (animationFrameId.current !== null) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
      }
      
      flushBuffer();
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

    const value = useMemo(() => ({
      results,
      isComplete,
      batchConfig,
      addResult,
      clearResults,
      markComplete,
      setBatchConfig,
      setSingleResult
    }), [
        results, 
        isComplete, 
        batchConfig, 
        addResult, 
        clearResults, 
        markComplete, 
        setBatchConfig,
        setSingleResult
    ]);

    return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
  };

  export const useAnalysis = (): AnalysisContextType => {
    const context = useContext(AnalysisContext);
    if (context === undefined) {
      throw new Error('useAnalysis must be used within an AnalysisContextProvider');
    }
    return context;
  };