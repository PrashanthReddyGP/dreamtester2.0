import React from 'react';
import { Box, Paper, Typography, useTheme } from '@mui/material';
import ReactECharts from 'echarts-for-react';

// Mock data for the equity curve. In a real app, this would be an array of [timestamp, value].
const generateMockEquityData = () => {
  const data = [];
  let value = 10000;
  const baseTime = new Date('2023-01-01').getTime();
  for (let i = 0; i < 365; i++) {
    const now = new Date(baseTime + i * 24 * 3600 * 1000);
    value += Math.random() * 300 - 130;
    data.push([now, Math.round(value)]);
  }
  return data;
};

const calculateDrawdown = (equityData: number[][]) => {
  let peak = -Infinity;
  return equityData.map(point => {
    if (point[1] > peak) {
      peak = point[1];
    }
    const drawdown = (point[1] - peak) / peak;
    // Return a tuple: [timestamp, drawdown_percentage]
    return [point[0], drawdown * 100];
  });
};


export const OverviewTab: React.FC = () => {
  const theme = useTheme();
  const equityData = generateMockEquityData();
  const drawdownData = calculateDrawdown(equityData);
  
  const chartOption = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: { // Add a legend
        data: ['Equity', 'Drawdown'],
        textStyle: { color: theme.palette.text.primary }
    },
    xAxis: {
      type: 'time',
    },
    yAxis: [ // Use an array for two Y-axes
      {
        type: 'value',
        name: 'Equity',
        scale: true,
        axisLabel: { formatter: '${value}' }
      },
      {
        type: 'value',
        name: 'Drawdown',
        scale: true,
        axisLabel: { formatter: '{value}%' }
      }
    ],
    grid: { left: '3%', right: '4%', bottom: '10%', containLabel: true },
    series: [
      {
        name: 'Equity',
        type: 'line',
        yAxisIndex: 0,
        showSymbol: false,
        smooth: true,
        data: equityData,
        lineStyle: {
          color: theme.palette.primary.main,
          width: 2,
        },
        areaStyle: {
            color: theme.palette.primary.main,
            opacity: 0.1
        }
      },
      {
        name: 'Drawdown',
        type: 'line',
        yAxisIndex: 1,
        showSymbol: false,
        smooth: true,
        data: drawdownData,
        lineStyle: { width: 0 },
        areaStyle: {
            color: theme.palette.error.main,
            opacity: 0.3
        }
      }
    ],

    backgroundColor: 'transparent',
    textStyle: {
        color: theme.palette.text.secondary
    }
  };

  return (
    <Box p={3}>
      <Typography variant="h2" gutterBottom>
        Equity Curve
      </Typography>
      <Box sx={{ height: 500 }}>
        <ReactECharts option={chartOption} style={{ height: '100%', width: '100%' }} />
      </Box>
    </Box>
  );
};