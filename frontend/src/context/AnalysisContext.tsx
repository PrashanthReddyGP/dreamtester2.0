// src/context/AnalysisContext.tsx
import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { websocketService } from '../services/websocketService';
import type { StrategyResult } from '../services/api'; // Assuming StrategyResult type is in api.ts

// Define the shape of the context state
interface AnalysisContextType {
  results: StrategyResult[];
  isComplete: boolean;
  clearResults: () => void;
}

// Create the context
const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [results, setResults] = useState<StrategyResult[]>([]);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    // Subscribe to the websocket service when the component mounts
    const subscription = websocketService.subscribe((data) => {
      const { type, payload } = data;

      // This context ONLY cares about 'strategy_result' and 'batch_complete' messages
      if (type === 'strategy_result') {
        console.log("AnalysisContext: Received a new strategy result.", payload.strategy_name);
        setResults(prevResults => {
            // Check if a result with the same name already exists (e.g., for Portfolio updates)
            const existingIndex = prevResults.findIndex(r => r.strategy_name === payload.strategy_name);
            if (existingIndex > -1) {
                // Replace the existing result
                const newResults = [...prevResults];
                newResults[existingIndex] = payload;
                return newResults;
            } else {
                // Add the new result
                return [...prevResults, payload];
            }
        });
      } else if (type === 'batch_complete') {
        console.log("AnalysisContext: Batch is complete.");
        setIsComplete(true);
      }
    });

    // Unsubscribe when the component unmounts to prevent memory leaks
    return () => {
      subscription.unsubscribe();
    };
  }, []); // The empty dependency array ensures this runs only once

  // Function to be called from StrategyLab before starting a new run
  const clearResults = useCallback(() => {
    setResults([]);
    setIsComplete(false);
  }, []);

  return (
    <AnalysisContext.Provider value={{ results, isComplete, clearResults }}>
      {children}
    </AnalysisContext.Provider>
  );
};

// Custom hook for easy consumption
export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (context === undefined) {
    throw new Error('useAnalysis must be used within an AnalysisContextProvider');
  }
  return context;
};