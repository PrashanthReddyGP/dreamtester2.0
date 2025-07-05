import React from 'react';
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

interface MetricItemProps {
  label: string;
  value: string | number;
  tooltip: string;
  isPercentage?: boolean;
}

const MetricItem: FC<MetricItemProps> = ({ label, value, tooltip, isPercentage = false }) => (
  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 1.5, px: 2 }}>
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <Typography variant="body1" color="text.secondary">
        {label}
      </Typography>
      <Tooltip title={tooltip} placement="top" arrow>
        <InfoOutlinedIcon sx={{ fontSize: '1rem', ml: 0.5, color: 'text.disabled' }} />
      </Tooltip>
    </Box>
    <Box sx={{ minWidth: '120px', textAlign: 'right' }}>
      <Typography variant="body1" fontWeight={600}>
        {value}{isPercentage ? '%' : ''}
      </Typography>
    </Box>
  </Box>
);

// --- Data interfaces (no changes) ---
interface MetricsData { sharpeRatio: number; sortinoRatio: number; calmarRatio: number; profitFactor: number; maxDrawdown: number; maxDrawdownDuration: number; avgDrawdown: number; annualizedVolatility: number; avgWin: number; avgLoss: number; riskRewardRatio: number; maxConsecutiveWins: number; maxConsecutiveLosses: number; avgHoldingPeriod: number; }
interface MonthlyReturn { month: string; profit: number; returns: number; }

// --- Mock data (no changes) ---
const mockMetricsData: MetricsData = { sharpeRatio: 1.78, sortinoRatio: 2.54, calmarRatio: 3.12, profitFactor: 2.15, maxDrawdown: -15.4, maxDrawdownDuration: 42, avgDrawdown: -5.8, annualizedVolatility: 22.3, avgWin: 450.75, avgLoss: -210.10, riskRewardRatio: 2.14, maxConsecutiveWins: 8, maxConsecutiveLosses: 3, avgHoldingPeriod: 36 };
const mockMonthlyReturns: MonthlyReturn[] = [
  { month: 'Jan 2023', profit: 1200, returns: 12.0 }, 
  { month: 'Feb 2023', profit: -350, returns: -3.1 }, 
  { month: 'Mar 2023', profit: 2100, returns: 18.5 }, 
  { month: 'Apr 2023', profit: 800, returns: 6.7 },
  { month: 'May 2023', profit: 1200, returns: 12.0 }, 
  { month: 'Jun 2023', profit: -350, returns: -3.1 }, 
  { month: 'Jul 2023', profit: 2100, returns: 18.5 }, 
  { month: 'Aug 2023', profit: 800, returns: 6.7 },
  { month: 'Sep 2023', profit: 1200, returns: 12.0 }, 
  { month: 'Oct 2023', profit: -350, returns: -3.1 }, 
  { month: 'Nov 2023', profit: 2100, returns: 18.5 }, 
  { month: 'Dec 2023', profit: 800, returns: 6.7 }
];

const allMetrics = [
    { type: 'header', label: 'Performance Metrics' },
    { type: 'metric', label: 'Sharpe Ratio', value: mockMetricsData.sharpeRatio.toFixed(2), tooltip: 'Measures risk-adjusted return.' },
    { type: 'metric', label: 'Sortino Ratio', value: mockMetricsData.sortinoRatio.toFixed(2), tooltip: 'Similar to Sharpe, but only considers downside volatility.' },
    { type: 'metric', label: 'Calmar Ratio', value: mockMetricsData.calmarRatio.toFixed(2), tooltip: 'Measures return relative to the maximum drawdown.' },
    { type: 'metric', label: 'Profit Factor', value: mockMetricsData.profitFactor.toFixed(2), tooltip: 'Gross profits divided by gross losses.' },
    { type: 'metric', label: 'Max Drawdown', value: mockMetricsData.maxDrawdown.toFixed(2), tooltip: 'The largest peak-to-trough decline in portfolio value.', isPercentage: true },
    { type: 'metric', label: 'Avg. Drawdown', value: mockMetricsData.avgDrawdown.toFixed(2), tooltip: 'The average of all drawdown periods.', isPercentage: true },
    { type: 'metric', label: 'Max Drawdown Duration', value: `${mockMetricsData.maxDrawdownDuration} days`, tooltip: 'The longest time it took to recover from a peak.' },
    { type: 'metric', label: 'Annualized Volatility', value: mockMetricsData.annualizedVolatility.toFixed(2), tooltip: 'The standard deviation of returns, annualized.', isPercentage: true },
    { type: 'metric', label: 'Avg. Win / Avg. Loss', value: mockMetricsData.riskRewardRatio.toFixed(2), tooltip: 'The average profit from winning trades divided by the average loss from losing trades.' },
    { type: 'metric', label: 'Max Consecutive Wins', value: mockMetricsData.maxConsecutiveWins, tooltip: 'The longest streak of winning trades.' },
    { type: 'metric', label: 'Max Consecutive Losses', value: mockMetricsData.maxConsecutiveLosses, tooltip: 'The longest streak of losing trades.' },
    { type: 'metric', label: 'Avg. Holding Period', value: `${mockMetricsData.avgHoldingPeriod} hours`, tooltip: 'The average duration of a trade from entry to exit.' },
];

export const MetricsTab: React.FC = () => {
  const pnlColor = (value: number) => value >= 0 ? 'success.main' : 'error.main';
  const theme = useTheme();
  return (
    <Box p={3} position={'relative'} height={'100%'}>
      <Grid container justifyContent={'space-between'} height={'100%'}>

        <Grid item sx={{width:'50%', height:'100%'}}>
          <Paper elevation={0} sx={{ height: '100%', display:'flex',flexDirection:'column', justifyContent:'space-between' }}>
            {allMetrics.map((metric, index) => {
              if (metric.type === 'header') {
                return (
                  <Typography key={index} variant="h2" sx={{ p: 2, pb: 1 }} align='center'>
                    {metric.label}
                  </Typography>
                );
              }
              if (metric.type === 'metric') {
                return (
                  <MetricItem
                    key={index}
                    label={metric.label}
                    value={metric.value}
                    tooltip={metric.tooltip}
                    isPercentage={metric.isPercentage}
                  />
                );
              }
              if (metric.type === 'divider') {
                return <Divider key={index} sx={{ my: 1 }} />;
              }
              return null;
            })}
          </Paper>
        </Grid>

        <Box border={`1px solid ${theme.palette.divider}`}>
        </Box>

        <Grid item xs={12} md={4}  sx={{width:'40%', height:'100%'}}>
          <Paper elevation={0} sx={{ height: '100%', display:'flex', flexDirection:'column' }}>

            <Typography variant="h2" gutterBottom align={'center'} sx={{ p: 2, pb: 1 }}>
              Monthly Returns
            </Typography>
            
            <TableContainer sx={{flexGrow:1}}>
              <Table size="small" sx={{height:'100%'}}>

                <TableHead>
                  <TableRow >
                    <TableCell>Month</TableCell>
                    <TableCell align="right">Profit ($)</TableCell>
                    <TableCell align="right">Returns (%)</TableCell>
                  </TableRow>
                </TableHead>

                <TableBody>
                  {mockMonthlyReturns.map((row) => (
                    <TableRow key={row.month}>
                      
                      <TableCell >{row.month}</TableCell>
                      
                      <TableCell align="right" sx={{ color: pnlColor(row.profit), fontWeight: 500 }}>
                          {row.profit >= 0 ? '+' : ''}{row.profit.toFixed(2)}
                      </TableCell>
                      
                      <TableCell align="right" sx={{ color: pnlColor(row.returns), fontWeight: 500 }}>
                          {row.returns.toFixed(2)}%
                      </TableCell>
                    
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

      </Grid>
    </Box>
  );
};