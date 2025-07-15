import React, { useMemo } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  useTheme,
  IconButton,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ReactECharts from 'echarts-for-react';
import type { StrategyResult } from '../../services/api';

interface ComparisonModalProps {
  open: boolean;
  onClose: () => void;
  results: StrategyResult[];
  initialCapital: number;
}

// NEW: A modern, vibrant, and distinct color palette for your charts.
const CHART_COLORS = [
  '#5470C6', '#91CC75', '#FAC858', '#EE6666',
  '#73C0DE', '#3BA272', '#FC8452', '#9A60B4', '#EA7CCC'
];

export const ComparisonModal: React.FC<ComparisonModalProps> = ({
  open,
  onClose,
  results,
  initialCapital,
}) => {
  const theme = useTheme();

  const chartOption = useMemo(() => {
    const strategiesToCompare = results.filter(
      (r) => r.strategy_name !== 'Portfolio'
    );

    const seriesData = strategiesToCompare.map((strategy, index) => ({
      name: strategy.strategy_name,
      type: 'line',
      showSymbol: false,
      // NEW: A slightly higher smoothing factor for a less jagged look.
      smooth: 0.2, 
      data: strategy.equity_curve.map((point) => [
        point[0] * 1000,
        point[1] - initialCapital,
      ]),
      // NEW: Enhance interactivity on hover. This highlights the hovered series.
      emphasis: {
        focus: 'series', // This is the key property
        lineStyle: {
          width: 3, // Make the hovered line thicker
        },
      },
      // NEW: Add a semi-transparent area fill for a modern aesthetic.
      areaStyle: {
        opacity: 0.15,
        // The color will be inherited from the line color by default
      },
    }));

    const legendData = strategiesToCompare.map((s) => s.strategy_name);

    return {
      color: CHART_COLORS,
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        formatter: (params: any[]) => {
            const date = new Date(params[0].axisValue);
            const dateString = date.toLocaleDateString();
            let tooltipHtml = `${dateString}<br/>`;

            params.sort((a, b) => b.value[1] - a.value[1]); // Sort tooltip values descending

            params.forEach((param: any) => {
                const seriesName = param.seriesName;
                const value = param.value[1];
                const color = param.color;
                const marker = `<span style="display:inline-block;margin-right:5px;border-radius:10px;width:10px;height:10px;background-color:${color};"></span>`;
                tooltipHtml += `${marker} ${seriesName}: <b>$${value.toFixed(2)}</b><br/>`;
            });

            return tooltipHtml;
        }
      },
      legend: {
        data: legendData,
        textStyle: { color: theme.palette.text.primary },
        type: 'scroll',
        orient: 'horizontal',
        top: 'top',     
        left: 'center',    
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'time',
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        name: 'Net Profit ($)',
        scale: true,
        axisLabel: { formatter: '${value}' },
        splitLine: {
            show: true,
            lineStyle: {
                color: theme.palette.grey[700],
                type: 'dotted',
                opacity: 0.2,
            }
        },
        axisLine: {
            show: true,
        }
      },
      dataZoom: [
        { type: 'inside', xAxisIndex: [0] },
        { type: 'slider', xAxisIndex: [0], textStyle: { color: theme.palette.text.secondary } },
      ],
      series: seriesData,
      backgroundColor: 'transparent',
      textStyle: { color: theme.palette.text.secondary },
    };
  }, [results, initialCapital, theme]);

  const canCompare = results.filter(r => r.strategy_name !== 'Portfolio').length > 1;

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="xl" PaperProps={{ sx: { height: '90vh' } }}>
      <DialogTitle sx={{ m: 0, p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        Strategy Equity Comparison
        <IconButton aria-label="close" onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers sx={{ p: 1, overflow: 'hidden' }}>
        <Box sx={{ height: '100%', width: '100%' }}>
          {canCompare ? (
            <ReactECharts
              option={chartOption}
              style={{ width: '100%', height: '100%' }}
              notMerge={true}
              lazyUpdate={true}
            />
          ) : (
             <Box sx={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%'}}>
                <Typography>At least two strategies are needed for a comparison.</Typography>
             </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};