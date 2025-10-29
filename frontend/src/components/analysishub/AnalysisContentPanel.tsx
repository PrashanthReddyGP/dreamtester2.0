import React, {useState} from 'react';
import type { FC } from 'react';
import { Box, Paper, Typography, Tabs, Tab } from '@mui/material';

// Import your existing tab components
import { EquityTab } from './EquityTab';
import { TradeLogTab } from './TradeLogTab';
import { AdvancedMetricsTab } from './AdvancedMetricsTab';
import { MetricsOverviewTab } from './MetricsOverviewTab';
import type { StrategyResult, MLResult } from '../../services/api';
import { isMLResult } from '../../services/api';
import { CorrelationMatrixTab } from './CorrelationMatrixTab';
import { ModelAnalysisTab } from './ModelAnalysisTab';

// A type alias for clarity
type AnalysisResult = StrategyResult | MLResult;

export const AnalysisContentPanel: FC<{
    results: AnalysisResult[], // Accept the union type
    result: AnalysisResult,   // Accept the union type
    initialCapital: number
  }> = ({ results, result, initialCapital }) => {
  const [currentTab, setCurrentTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const isPortfolioView = result.strategy_name === 'Portfolio';

  const showMLTab = isMLResult(result);

  return (
    <Paper elevation={0} sx={{display: 'flex', flexDirection: 'column', height:'100%', width:'100%', position:'relative'}}>
      <Box sx={{ display:'flex', flexDirection:'row', justifyContent:'space-between', paddingRight:4, borderBottom: 1, borderColor: 'divider', overflow:'fixed'}}>
        <>
            <Tabs value={currentTab} onChange={handleTabChange} aria-label="analysis tabs">
            <Tab label="Equity" />
            <Tab label="Trade Log" />
            <Tab label="Metrics Overview" />
            {showMLTab && <Tab label="Model Analysis" />} // Conditionally render the ML tab
            <Tab label="Correlation Matrix" /> 
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
            strategy_name={result.strategy_name}
          />}
        {currentTab === 1 && 
          <TradeLogTab 
            trades={result.trades}
          />}        
        {currentTab === 2 && 
          <MetricsOverviewTab
            results={results}
          />}
        {showMLTab && currentTab === 3 && (
          <ModelAnalysisTab analysis={result.model_analysis} />
        )}
        {currentTab === (showMLTab ? 4 : 3) &&
          <CorrelationMatrixTab 
            results={results}
          />}
        {currentTab === (showMLTab ? 5 : 4) && 
          <AdvancedMetricsTab 
            metrics={result.metrics}
            monthlyReturns={result.monthly_returns}
          />}
      </Box>
    </Paper>
  );
};