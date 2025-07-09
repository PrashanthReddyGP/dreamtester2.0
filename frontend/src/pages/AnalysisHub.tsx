import React, { useState, useEffect } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { Panel, PanelGroup } from 'react-resizable-panels';

// Import the components we just created
import { StrategyListPanel } from '../components/analysishub/StrategyListPanel';
import { AnalysisContentPanel } from '../components/analysishub/AnalysisContentPanel';
import type { StrategyResult } from '../services/api';
import { ResizeHandle } from '../components/common/ResizeHandle';
import { useAnalysis } from '../context/AnalysisContext';

export const AnalysisHub: React.FC = () => {
  
  const { results, isComplete } = useAnalysis();
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyResult | null>(null);

  useEffect(() => {
    // If we have results but nothing is selected, default to the first one.
    if (results.length > 0 && !selectedStrategy) {
      setSelectedStrategy(results[0]);
    }
    // If the currently selected strategy is no longer in the results list (e.g., after clearing), deselect it.
    else if (selectedStrategy && !results.some(r => r.strategy_name === selectedStrategy.strategy_name)) {
      setSelectedStrategy(results.length > 0 ? results[0] : null);
    }
  }, [results, selectedStrategy]);

  const handleSelectStrategy = (strategyName: string) => {
      const foundStrategy = results.find(s => s.strategy_name === strategyName);
      if (foundStrategy) {
          setSelectedStrategy(foundStrategy);
      }
  };

  // Condition 1: The process has started but no results have arrived yet.
  if (results.length === 0 && !isComplete) {
      return (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', width: '100vw' }}>
              <CircularProgress sx={{ mb: 2 }} />
              <Typography variant="h6">Waiting for backtest results...</Typography>
              <Typography variant="body1" color="text.secondary">
                  Results will appear here in real-time.
              </Typography>
          </Box>
      );
  }
  
  // Condition 2: The process is finished, but there are no results (all failed).
  if (results.length === 0 && isComplete) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', width: '100vw' }}>
          <Typography variant="h5" color="error">Backtest Failed</Typography>
          <Typography variant="body1" color="text.secondary">
              No strategies completed successfully. Check the terminal for error logs.
          </Typography>
      </Box>
    )
  }

  // Condition 3: We have results to display.
  return (
    <Box sx={{ height: '100%', width: '100vw'}}>

      <PanelGroup direction='horizontal' style={{display:'flex', flexDirection:'row', flexWrap:'nowrap', width:'100vw'}}>

        <Panel style={{flexGrow:1}}>
          <StrategyListPanel
            results={results.map(s => ({ id: s.strategy_name, name: s.strategy_name }))}
            selectedId={selectedStrategy?.strategy_name || ''}
            onSelect={handleSelectStrategy}
          />
        </Panel>

        <ResizeHandle/>
        
        <Panel style={{flexGrow:4}}>
          {selectedStrategy ? (
            <AnalysisContentPanel 
                result={selectedStrategy} 
                initialCapital={1000}
            />
            ) : (
                <Typography sx={{ p: 4 }}>Select a strategy to view results.</Typography>
          )}
        </Panel>

      </PanelGroup>

    </Box>
  );
};