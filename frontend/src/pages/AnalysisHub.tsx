import React, { useState, useEffect } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { Panel, PanelGroup } from 'react-resizable-panels';

// Import the components we just created
import { StrategyListPanel } from '../components/analysishub/StrategyListPanel';
import { AnalysisContentPanel } from '../components/analysishub/AnalysisContentPanel';
import { getLatestBacktestResult } from '../services/api';
import type { BacktestResultPayload, StrategyResult } from '../services/api';
import { ResizeHandle } from '../components/common/ResizeHandle';
import { useAppContext } from '../context/AppContext'; // Import the context hook

export const AnalysisHub: React.FC = () => {
  const { latestBacktest, isBacktestLoading, backtestError, fetchLatestResults } = useAppContext();
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyResult | null>(null);
  // const [backtestData, setBacktestData] = useState<BacktestResultPayload | null>(null);
  // const [isLoading, setIsLoading] = useState(true);
  // const [error, setError] = useState<string | null>(null);

  // useEffect(() => {
  //     // A self-calling async function to handle polling
  //     const pollForResult = async () => {
  //         try {
  //             const result = await getLatestBacktestResult();
  //             if (result && result.strategies_results.length > 0) {
  //                 setBacktestData(result);
  //                 // Automatically select the first strategy in the list by default.
  //                 // This is often the "Portfolio" result if you added it first.
  //                 setSelectedStrategy(result.strategies_results[0]);
  //                 setIsLoading(false);
  //             } else if (result) {
  //                 // Result was returned but empty, means no strategies were run
  //                 setError("No strategies were successfully backtested.");
  //                 setIsLoading(false);
  //             }
  //             else {
  //                 // No result yet (backend returned null), so poll again.
  //                 setTimeout(pollForResult, 5000);
  //             }

  //         } catch (err) {
  //               setError("Failed to load backtest results. Is the backend running?");
  //             setIsLoading(false);
  //         }
  //     };

  //     pollForResult(); // Start the polling process
  // }, []); // The empty dependency array ensures this runs only once on mount

  React.useEffect(() => {
      if (latestBacktest && latestBacktest.strategies_results.length > 0) {
          // If there's no selected strategy yet, or if the selected one is not in the new data,
          // default to the first one.
          const currentSelectionExists = latestBacktest.strategies_results.some(s => s.strategy_name === selectedStrategy?.strategy_name);
          if (!selectedStrategy || !currentSelectionExists) {
              setSelectedStrategy(latestBacktest.strategies_results[0]);
          }
      }
  }, [latestBacktest, selectedStrategy]);

  const handleSelectStrategy = (strategyName: string) => {
      const foundStrategy = latestBacktest?.strategies_results.find(s => s.strategy_name === strategyName);
      if (foundStrategy) {
          setSelectedStrategy(foundStrategy);
      }
  };

  if (isBacktestLoading) {
      return (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', width: '100vw' }}>
              <CircularProgress />
              <Typography sx={{ ml: 2 }}>Waiting for backtest results...</Typography>
          </Box>
      );
  }
  
  if (backtestError) return <Typography color="error">{backtestError}</Typography>;

  if (!latestBacktest || !selectedStrategy) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', width: '100vw' }}>
          <Typography sx={{ p: 4 }}>No backtest data available. Run a new backtest from the Strategy Lab.</Typography>
        </Box>
      )
  }

  return (
    <Box sx={{ height: '100%', width: '100vw'}}>

      <PanelGroup direction='horizontal' style={{display:'flex', flexDirection:'row', flexWrap:'nowrap', width:'100vw'}}>

        <Panel style={{flexGrow:1}}>
          <StrategyListPanel
            results={latestBacktest.strategies_results.map(s => ({ id: s.strategy_name, name: s.strategy_name }))}
            selectedId={selectedStrategy.strategy_name}
            onSelect={handleSelectStrategy}
            onReload={fetchLatestResults} 
            isLoading={isBacktestLoading}
          />
        </Panel>

        <ResizeHandle/>
        
        <Panel style={{flexGrow:4}}>
          {selectedStrategy && latestBacktest? (
            <AnalysisContentPanel 
                // Pass the selected strategy object down
                result={selectedStrategy} 
                // Also pass down the initial capital for PnL calculations
                initialCapital={latestBacktest.initial_capital}
            />
            ) : (
                <Typography sx={{ p: 4 }}>Select a strategy to view results.</Typography>
          )}
        </Panel>

      </PanelGroup>

    </Box>
  );
};