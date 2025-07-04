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
    legend: {
        data: ['Equity', 'Drawdown'],
        textStyle: { color: theme.palette.text.primary }
    },
    xAxis: {
      type: 'time',
      splitLine: {
        show: false
      }
    },
    yAxis: [
      {
        type: 'value',
        name: 'Equity',
        scale: true,
        axisLabel: { formatter: '${value}' },
        splitLine: {show: false}
      },
      {
        type: 'value',
        name: 'Drawdown',
        scale: true,
        axisLabel: { formatter: '{value}%' },
        splitLine: {show: false}
      }
    ],
    
    grid: { left: '1%', right: '1%', bottom: '1%', containLabel: true },
    
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
            opacity: 0.4
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
            opacity: 0.1
        }
      }
    ],

    backgroundColor: 'transparent',
    textStyle: {
        color: theme.palette.text.secondary
    }
  };

  return (
    <Box sx={{height:'100%'}}>
      
      <Typography variant="h2" gutterBottom>
        Equity Curve
      </Typography>
      
      <Box sx={{height:'100%'}}>
        <ReactECharts option={chartOption} style={{width:'100%', height:'100%'}}/>
      </Box>

    </Box>
  );
};