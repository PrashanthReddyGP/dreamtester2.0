import React, { useState, useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

// Import the components we just created
import { StrategyListPanel } from '../components/analysishub/StrategyListPanel';
import type { BacktestResult } from '../components/analysishub/StrategyListPanel';
import { AnalysisContentPanel } from '../components/analysishub/AnalysisContentPanel';

// Mock data for the list of available backtest results
const mockResults: BacktestResult[] = [
  { id: '1', name: 'RSI_Momentum_v2.py' },
  { id: '2', name: 'SMA_Crossover_Final.py' },
  { id: '3', name: 'Bollinger_Breakout.py' },
];

export const AnalysisHub: React.FC = () => {
  // State to track which strategy is currently selected
  const [selectedId, setSelectedId] = useState<string | null>(mockResults[0]?.id ?? null);

  // Find the full object for the selected strategy
  const selectedResult = useMemo(
    () => mockResults.find(r => r.id === selectedId),
    [selectedId]
  );

  return (
    <Box sx={{ height: '100%', width: '100vw'}}>

      <PanelGroup direction="horizontal">

        <Panel defaultSize={20}>
          <StrategyListPanel
            results={mockResults}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </Panel>

        <Panel>
          {selectedResult ? (
            <AnalysisContentPanel result={selectedResult} />
          ) : (
            <Box sx={{ p: 4 }}>
              <Typography variant="h5" color="text.secondary">
                Select a backtest result to view the analysis.
              </Typography>
            </Box>
          )}
        </Panel>

      </PanelGroup>

    </Box>
  );
};