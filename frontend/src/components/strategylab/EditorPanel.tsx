import React from 'react';
import Editor from '@monaco-editor/react';
import { Box } from '@mui/material';

const sampleCode = `import crypto_backtester_core as core

class SmaCrossover(core.BaseStrategy):
    """
    A simple moving average crossover strategy.
    """
    def init(self):
        # Define parameters
        self.fast_sma_period = 20
        self.slow_sma_period = 50

        # Pre-calculate indicators
        self.fast_sma = self.data['close'].rolling(self.fast_sma_period).mean()
        self.slow_sma = self.data['close'].rolling(self.slow_sma_period).mean()
    
    def next(self, current_bar_index):
        # Entry condition: Fast SMA crosses above Slow SMA
        if self.fast_sma[current_bar_index] > self.slow_sma[current_bar_index]:
            self.buy()
        
        # Exit condition: Fast SMA crosses below Slow SMA
        elif self.fast_sma[current_bar_index] < self.slow_sma[current_bar_index]:
            self.sell()
`;

export const EditorPanel: React.FC = () => {
  return (
      <Box sx={{ height: '100%', bgcolor: 'background.paper', p: 0.25,}}>
        <Editor
          height="100%"
          defaultLanguage="python"
          defaultValue={sampleCode}
          // We can now directly use our globally defined theme name!
          theme="app-dark-theme" 
          options={{
            minimap: { enabled: false },
            fontSize: 14,
            wordWrap: 'on',
            scrollBeyondLastLine: false,
            padding: { top: 16 }
          }}
        />
      </Box>
  );
};