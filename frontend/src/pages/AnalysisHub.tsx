import React, { useState, useEffect, useMemo } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { Panel, PanelGroup } from 'react-resizable-panels';
import { DataGrid } from '@mui/x-data-grid'; // Import the DataGrid
import type {GridColDef, GridRenderCellParams} from '@mui/x-data-grid'

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
    // This logic now works for both batch and optimization runs.
    // It ensures something is always selected if results exist.
    if (results.length > 0) {
      // If nothing is selected, select the first result in the list.
      if (!selectedStrategy) {
        setSelectedStrategy(results[0]);
      } 
      // This handles a tricky edge case: if a previous selection disappears
      // (which shouldn't happen in this new model, but is good for robustness),
      // it re-selects the first item.
      else if (!results.some(r => r.strategy_name === selectedStrategy.strategy_name)) {
        setSelectedStrategy(results[0]);
      }
    } else {
        // If results are cleared, clear the selection
        setSelectedStrategy(null);
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
              <Typography variant="h6">Running Backtest...</Typography>
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
                results={results}
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
}