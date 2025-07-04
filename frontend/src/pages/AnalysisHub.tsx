import React, { useState } from 'react';
import { Box, Typography, Grid, Paper, Tabs, Tab } from '@mui/material';

// We'll create these components next
import { KpiCard } from '../components/analysis/KpiCard';
import { OverviewTab } from '../components/analysis/OverviewTab';
import { TradeLogTab } from '../components/analysis/TradeLogTab';

// Mock data for our KPIs. Later this will come from the backend.
const kpiData = {
  netProfit: { value: 24.8, unit: '%' },
  sharpeRatio: { value: 1.78, unit: '' },
  profitFactor: { value: 2.5, unit: '' },
  winRate: { value: 62.1, unit: '%' },
  totalTrades: { value: 87, unit: '' },
  maxDrawdown: { value: -12.3, unit: '%' },
};

export const AnalysisHub: React.FC = () => {
  const [currentTab, setCurrentTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  return (
    <Box sx={{height:'100%', width:'100vw'}}>
      <Typography variant="h1" paddingLeft={5} paddingTop={2} gutterBottom>
        Analysis Hub
      </Typography>

      {/* KPI Cards Section */}
      <Grid container spacing={2} sx={{ justifyContent:'space-between', mb: 2, ml: 2, mr: 2 }}>
        <Grid item xs={6} md={4} lg={2} sx={{flexGrow:1}}><KpiCard title="Net Profit" data={kpiData.netProfit} /></Grid>
        <Grid item xs={6} md={4} lg={2} sx={{flexGrow:1}}><KpiCard title="Sharpe Ratio" data={kpiData.sharpeRatio} /></Grid>
        <Grid item xs={6} md={4} lg={2} sx={{flexGrow:1}}><KpiCard title="Profit Factor" data={kpiData.profitFactor} /></Grid>
        <Grid item xs={6} md={4} lg={2} sx={{flexGrow:1}}><KpiCard title="Win Rate" data={kpiData.winRate} /></Grid>
        <Grid item xs={6} md={4} lg={2} sx={{flexGrow:1}}><KpiCard title="Total Trades" data={kpiData.totalTrades} /></Grid>
        <Grid item xs={6} md={4} lg={2} sx={{flexGrow:1}}><KpiCard title="Max Drawdown" data={kpiData.maxDrawdown} /></Grid>
      </Grid>

      {/* Tabs for different analysis views */}
      <Paper elevation={0}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange} aria-label="analysis tabs">
            <Tab label="Overview" />
            <Tab label="Trade Log" />
            <Tab label="Advanced Metrics" />
          </Tabs>
        </Box>
        {currentTab === 0 && <OverviewTab />}
        {currentTab === 1 && <TradeLogTab />}
        {currentTab === 2 && <Box p={3}><Typography>Advanced Metrics content will be here.</Typography></Box>}
      </Paper>
    </Box>
  );
};