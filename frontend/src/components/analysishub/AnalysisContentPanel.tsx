import React, {useState} from 'react';
import type { FC } from 'react';
import { Box, Paper, Typography, Tabs, Tab, Grow } from '@mui/material';
import type { BacktestResult } from './StrategyListPanel'; // Import the type

// Import your existing tab components
import { EquityTab } from './EquityTab';
import { TradeLogTab } from './TradeLogTab';
import { AdvancedMetricsTab } from './AdvancedMetricsTab';
import { MetricsOverviewTab } from './MetricsOverviewTab';
import type { StrategyResult } from '../../services/api'; 

// interface AnalysisContentPanelProps {
//   result: BacktestResult; // Expects the full result object
// }

export const AnalysisContentPanel: FC<{
    results: StrategyResult[],
    result: StrategyResult,
    initialCapital: number
  }> = ({ results, result, initialCapital }) => {
  const [currentTab, setCurrentTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const isPortfolioView = result.strategy_name === 'Portfolio';

  return (
    <Paper elevation={0} sx={{display: 'flex', flexDirection: 'column', height:'100%', width:'100%', position:'relative'}}>
      <Box sx={{ display:'flex', flexDirection:'row', justifyContent:'space-between', paddingRight:4, borderBottom: 1, borderColor: 'divider', overflow:'fixed'}}>
        <>
            <Tabs value={currentTab} onChange={handleTabChange} aria-label="analysis tabs">
            <Tab label="Equity" />
            <Tab label="Trade Log" />
            <Tab label="Metrics Overview" />
            <Tab label="Advanced Metrics" />
            </Tabs>
        </>
        <Box sx={{alignContent:'center'}}>
         <Typography color='grey'>
           {result.strategy_name}
         </Typography>
        </Box>
      </Box>

      <Box sx={{display:'flex', flexDirection:'column', p:2, height:'100%', overflow:'auto'}}>
        {currentTab === 0 && 
          <EquityTab
            equity={result.equity_curve}
            initialCapital={initialCapital}
            isPortfolio={isPortfolioView} 
          />}
        {currentTab === 1 && 
          <TradeLogTab 
            trades={result.trades}
          />}        
        {currentTab === 2 && 
          <MetricsOverviewTab
            results={results}
          />}
        {currentTab === 3 && 
          <AdvancedMetricsTab 
            metrics={result.metrics}
            monthlyReturns={result.monthly_returns}
          />}
      </Box>
    </Paper>
  );
};