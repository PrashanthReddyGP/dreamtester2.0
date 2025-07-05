import React, {useState} from 'react';
import type { FC } from 'react';
import { Box, Paper, Typography, Tabs, Tab, Grow } from '@mui/material';
import type { BacktestResult } from './StrategyListPanel'; // Import the type

// Import your existing tab components
import { OverviewTab } from './OverviewTab';
import { TradeLogTab } from './TradeLogTab';
import { MetricsTab } from './MetricsTab';

interface AnalysisContentPanelProps {
  result: BacktestResult; // Expects the full result object
}

export const AnalysisContentPanel: FC<AnalysisContentPanelProps> = ({ result }) => {
  const [currentTab, setCurrentTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  return (
    <Paper elevation={0} sx={{display: 'flex', flexDirection: 'column', height:'100%'}}>
      <Box sx={{ display:'flex', flexDirection:'row', justifyContent:'space-between', paddingRight:4, borderBottom: 1, borderColor: 'divider'}}>
        <>
            <Tabs value={currentTab} onChange={handleTabChange} aria-label="analysis tabs">
            <Tab label="Overview" />
            <Tab label="Trade Log" />
            <Tab label="Advanced Metrics" />
            </Tabs>
        </>
        <Box sx={{alignContent:'center'}}>
         <Typography color='grey'>
           {result.name}
         </Typography>
        </Box>
      </Box>

      <Box sx={{display:'flex', flexDirection:'column', p:2, height:'100%'}}>
        {currentTab === 0 && <OverviewTab />}
        {currentTab === 1 && <TradeLogTab />}
        {currentTab === 2 && <MetricsTab />}
      </Box>
    </Paper>
  );
};