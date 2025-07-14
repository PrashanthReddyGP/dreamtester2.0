import React, {useMemo} from 'react';
import type { FC } from 'react';
import {
    Box,
    Paper,
    Typography,
    Grid,
    Divider,
    Tooltip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    useTheme
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

import type { MonthlyReturns, StrategyMetrics } from '../../services/api';
import { color } from 'echarts';

// The MetricItem sub-component is used to render each individual metric line.
interface MetricItemProps {
  label: string;
  value: string | number;
  tooltip: string;
  color?: string; // Optional color for the value (e.g., 'success.main' or 'error.main')
}

const MetricItem: FC<MetricItemProps> = ({ label, value, tooltip, color }) => (
  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 1, px: 4 }}>
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <Typography variant="body1" color="text.secondary">
        {label}
      </Typography>
      <Tooltip title={tooltip} placement="top" arrow>
        <InfoOutlinedIcon sx={{ fontSize: '1rem', ml: 0.5, color: 'text.disabled' }} />
      </Tooltip>
    </Box>
    <Box sx={{ minWidth: '120px', textAlign: 'right' }}>
      <Typography variant="body1" fontWeight={600} sx={{ color }}>
        {value}
      </Typography>
    </Box>
  </Box>
);


// 2. The main component now accepts the `metrics` object as a prop.
export const AdvancedMetricsTab: FC<{ metrics: StrategyMetrics, monthlyReturns: MonthlyReturns }> = ({ metrics, monthlyReturns }) => {
  const theme = useTheme();

  // Helper functions for formatting the values consistently.
  const formatCurrency = (value: number) => {
    const sign = value < 0 ? '-' : '';
    // Formats with commas and no decimal places.
    return `${sign}$${Math.abs(value).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  };
  const formatPercent = (value: number) => `${(value || 0).toFixed(2)}%`;
  const formatRatio = (value: number) => (value || 0).toFixed(2);

  // 3. This configuration array drives the entire display.
  // It maps keys from the `metrics` prop to their display properties.
  const allMetricsConfig = [
    { type: 'header', label: 'Overall Performance' },
    { key: 'Net_Profit', label: 'Net Profit', tooltip: 'Total profit or loss after all trades.', format: formatCurrency, color: metrics.Net_Profit >= 0 ? 'success.main' : 'error.main' },
    { key: 'Gross_Profit', label: 'Gross Profit', tooltip: 'Total profit or loss after all trades.', format: formatCurrency, color: metrics.Net_Profit >= 0 ? 'success.main' : 'error.main' },
    { key: 'Profit_Percentage', label: 'Total Return', tooltip: 'Total return as a percentage of the initial capital.', format: formatPercent },
    { key: 'Annual_Return', label: 'Annualized Return', tooltip: 'The geometric average amount of money earned by an investment each year.', format: formatPercent },
    { key: 'Commission', label: 'Commissions', tooltip: 'The overall commissions deducted on this trade span.', format: formatPercent, color: 'error.main' }, 
    { key: 'Avg_Monthly_Return', label: 'Avg Monthly Return', tooltip: 'The geometric average amount of money earned by an investment each month.', format: formatPercent },
    { key: 'Total_Trades', label: 'Total Trades', tooltip: 'The total number of closed trades executed.' },
    { key: 'Open_Trades', label: 'Open Trades', tooltip: 'The geometric average amount of money earned by an investment each year.' },
    { key: 'Max_Drawdown', label: 'Max Drawdown', tooltip: 'The largest peak-to-trough decline in portfolio value.', format: formatPercent, color: 'error.main' },
    { key: 'Max_Runup', label: 'Max Runup', tooltip: 'The largest peak-to-trough decline in portfolio value.', format: formatPercent, color: 'error.main' },
    { key: 'Avg_Drawdown', label: 'Avg Drawdown', tooltip: 'The average of all drawdown periods.', format: formatPercent, color: 'error.main' },
    { key: 'Avg_Runup', label: 'Avg Runup', tooltip: 'The average of all drawdown periods.', format: formatPercent, color: 'error.main' },
    { key: 'Max_Drawdown_Duration_days', label: 'Max Drawdown Duration', tooltip: 'The longest time it took to recover from a peak.', unit: ' days' },
    { key: 'Sharpe_Ratio', label: 'Sharpe Ratio', tooltip: 'Measures risk-adjusted return, considering volatility.', format: formatRatio },
    { key: 'Profit_Factor', label: 'Profit Factor', tooltip: 'Gross profits divided by gross losses. Higher is better.', format: formatRatio },
    { key: 'Calmar_Ratio', label: 'Calmar Ratio', tooltip: 'Measures return relative to the maximum drawdown.', format: formatRatio },
    { key: 'RR', label: 'Risk/Reward Ratio', tooltip: 'The average profit from winning trades divided by the average loss from losing trades.', format: formatRatio },
    { key: 'Equity_Efficiency_Rate', label: 'Equity Efficiency Rate', tooltip: 'A custom metric for strategy quality.', format: formatRatio },
    { key: 'Strategy_Quality', label: 'Strategy Quality', tooltip: 'A qualitative assessment of the strategy.' },
    { key: 'Winrate', label: 'Win Rate', tooltip: 'The percentage of trades that were profitable.', format: formatPercent },
    { key: 'Total_Wins', label: 'Total Wins', tooltip: 'The total number of closed trades executed.' },
    { key: 'Total_Losses', label: 'Total Losses', tooltip: 'The total number of closed trades executed.' },
    { key: 'Avg_Trade_Time', label: 'Avg. Trade Duration', tooltip: 'The average time a position was held.' },
    { key: 'Avg_Win_Time', label: 'Avg. Win Duration', tooltip: 'The average time a position was held.' },
    { key: 'Avg_Loss_Time', label: 'Avg. Loss Duration', tooltip: 'The average time a position was held.' },
    { key: 'Largest_Win', label: 'Largest Win', tooltip: 'The single largest profitable trade.', format: formatCurrency, color: 'success.main' },
    { key: 'Largest_Loss', label: 'Largest Loss', tooltip: 'The single largest losing trade.', format: formatCurrency, color: 'error.main' },
    { key: 'Avg_Win', label: 'Avg. Win', tooltip: 'The average profit of all winning trades.', format: formatCurrency },
    { key: 'Avg_Loss', label: 'Avg. Loss', tooltip: 'The average loss of all losing trades.', format: formatCurrency },
    { key: 'Consecutive_Wins', label: 'Max Consecutive Wins', tooltip: 'The longest streak of winning trades.' },
    { key: 'Consecutive_Losses', label: 'Max Consecutive Losses', tooltip: 'The longest streak of losing trades.' },
    { key: 'Max_Open_Trades', label: 'Max Concurrent Trades', tooltip: 'The maximum number of trades that were open at the same time.' },
  ];

  const pnlColor = (value: number) => value >= 0 ? 'success.main' : 'error.main';

  const sortedMonthlyReturns = useMemo(() => {
      if (!monthlyReturns) {
          return [];
      }
      
      // Create a shallow copy to avoid mutating the original prop array
      return [...monthlyReturns].sort((a, b) => {
          // Convert "Month" string (e.g., "Jan 2023") to a Date object for proper comparison
          const dateA = new Date(a.Month);
          const dateB = new Date(b.Month);
          
          // Subtracting dates gives their difference in milliseconds.
          // Sorting b - a gives descending (reverse chronological) order.
          return dateB.getTime() - dateA.getTime();
      });
  }, [monthlyReturns]); // This sorting will only re-run when the monthlyReturns prop changes

  return (
    <Box position={'relative'}>
      <Grid container spacing={1}>
        {/* Performance Metrics Column */}
        <Grid sx={{width:'49%'}}>
          <Paper elevation={0} sx={{ height: '100%', border:1, borderColor: 'divider' }}>
            {allMetricsConfig.map((metric, index) => {
              if (metric.type === 'header') {
                return <Typography key={index} variant="h2" sx={{ p: 2, pb: 1, backgroundColor: 'action.hover'}} align='center' >{metric.label}</Typography>;
              }
              if (metric.type === 'divider') {
                return <Divider key={index} />;
              }
              // This check ensures we only render valid metrics from the config
              if (metric.key && metrics.hasOwnProperty(metric.key)) {
                
                const rawValue = metrics[metric.key as keyof StrategyMetrics];
                
                let formattedValue;
                
                if (metric.format && typeof rawValue === 'number') {
                    // Only apply formatter if it exists AND the value is a number
                    formattedValue = metric.format(rawValue);
                } else {
                    // Otherwise, use the raw value as is (for strings like "Good")
                    formattedValue = rawValue;
                }
                
                const finalValue = `${formattedValue}${metric.unit || ''}`;

                return (
                  <MetricItem
                    key={index}
                    label={metric.label}
                    value={finalValue}
                    tooltip={metric.tooltip}
                    color={metric.color}
                  />
                );
              }
              return null;
            })}
          </Paper>
        </Grid>

        {/* Monthly Returns Column */}
          <Grid sx={{width:'50%'}}>
            <Paper elevation={0} sx={{ height: '100%', border:1, borderColor: 'divider' }}>
              <Typography variant="h2" gutterBottom sx={{ p: 2, pb: 1, backgroundColor: 'action.hover' }} align='center'>
                  Monthly Returns
                </Typography>
                <TableContainer sx={{ flexGrow: 1 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Month</TableCell>
                        <TableCell align="right">Profit ($)</TableCell>
                        <TableCell align="right">Returns (%)</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {/* --- The mapping is now direct and simple --- */}
                      {sortedMonthlyReturns.map((row) => {
                        const profitValue = row['Profit ($)'];
                        const returnsValue = row['Returns (%)'];

                        return (
                          <TableRow key={row.Month}>
                            <TableCell component="th" scope="row">
                              {row.Month}
                            </TableCell>
                            
                            {/* --- THE FIX IS HERE --- */}
                            <TableCell align="right" sx={{ color: pnlColor(profitValue) }}>
                              {profitValue >= 0 ? '+' : ''}
                              {/* 
                                Use toLocaleString() to add commas.
                                We can also provide options to control decimal places.
                              */}
                              {Math.abs(profitValue).toLocaleString('en-US', {
                                minimumFractionDigits: 0,
                                maximumFractionDigits: 0,
                              })}
                            </TableCell>

                            <TableCell align="right" sx={{ color: pnlColor(returnsValue) }}>
                              {returnsValue.toFixed(2)}%
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>
      </Grid>
    </Box>
  );
};