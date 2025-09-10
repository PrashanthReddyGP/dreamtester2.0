import React, { useState, useEffect } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { Panel, PanelGroup } from 'react-resizable-panels';

// Import the components we just created
import { StrategyListPanel } from '../components/analysishub/StrategyListPanel';
import { AnalysisContentPanel } from '../components/analysishub/AnalysisContentPanel';
import type { StrategyResult, MLResult } from '../services/api';
import { ResizeHandle } from '../components/common/ResizeHandle';
import { useAnalysis } from '../context/AnalysisContext';
import { ComparisonModal } from '../components/analysishub/ComparisonModal';

// A type alias for clarity
type AnalysisResult = StrategyResult | MLResult;

export const AnalysisHub: React.FC = () => {
  
  const { results, isComplete, batchConfig } = useAnalysis();
  const [selectedStrategy, setSelectedStrategy] = useState<AnalysisResult | null>(null);
  const [isCompareModalOpen, setCompareModalOpen] = useState(false);
  const [isDataSegmentationMode, setIsDataSegmentationMode] = useState(false); 

  useEffect(() => {
    if (results.length > 0) {
      if (!selectedStrategy || !results.some(r => r.strategy_name === selectedStrategy.strategy_name)) {
        setSelectedStrategy(results[0]);
      }
    } else {
      setSelectedStrategy(null);
    }
  }, [results, selectedStrategy]);

  const handleSelectStrategy = (strategyName: string) => {
    const foundStrategy = results.find(s => s.strategy_name === strategyName);
    if (foundStrategy) {
      setSelectedStrategy(foundStrategy);
    }
  };

  const handleOpenCompareModal = () => setCompareModalOpen(true);
  const handleCloseCompareModal = () => setCompareModalOpen(false);

  // When you receive the batch data or first result, check for a tell-tale sign.
  // For example, if the batch job details from your DB include the `test_type`.
  useEffect(() => {
    if (batchConfig && batchConfig.test_type === 'data_segmentation') {
      setIsDataSegmentationMode(true);
    } else {
      setIsDataSegmentationMode(false);
    }
  }, [batchConfig]); // Dependency is now batchConfig

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
    <>
      <Box sx={{ height: '100%', width: '100vw'}}>

        <PanelGroup direction='horizontal' style={{display:'flex', flexDirection:'row', flexWrap:'nowrap', width:'100vw'}}>

          <Panel style={{flexGrow:1}}>
            <StrategyListPanel
              results={results.map(s => ({ id: s.strategy_name, name: s.strategy_name }))}
              selectedId={selectedStrategy?.strategy_name || ''}
              onSelect={handleSelectStrategy}
              onCompareClick={handleOpenCompareModal}
              isDataSegmentationMode={isDataSegmentationMode}
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

      <ComparisonModal
        open={isCompareModalOpen}
        onClose={handleCloseCompareModal}
        results={results}
        initialCapital={1000} // Pass the same initial capital for consistency
      />
    </>
  );
}