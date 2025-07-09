import React, { useRef, useEffect, useMemo } from 'react';
import { Box, IconButton, Typography, Tooltip, useTheme } from '@mui/material';
import ReactECharts from 'echarts-for-react';
import type { EquityCurvePoint } from '../../services/api';
import ZoomOutMapIcon from '@mui/icons-material/ZoomOutMap';

const calculateDrawdown = (equityCurve: number[][]) => {
  let peak = -Infinity;
  return equityCurve.map(point => {
    const equityValue = point[1];
    if (equityValue > peak) {
      peak = equityValue;
    }
    // Avoid division by zero if peak is 0 or less
    const drawdown = peak > 0 ? ((equityValue - peak) / peak) * 100 : 0;
    return [point[0], drawdown]; // [timestamp, drawdown_percentage]
  });
};

export const OverviewTab: React.FC<{
  equity: EquityCurvePoint[];
  initialCapital: number;
  isPortfolio: boolean; 
}> = ({ equity, initialCapital, isPortfolio }) => {
  const theme = useTheme();

  // The prop `equityCurve` directly replaces `generateMockEquityData()`
  let equityDataForChart = equity.map(point => [
    point[0] * 1000,
    point[1]
  ]); 

  const drawdownData = calculateDrawdown(equityDataForChart);
  
  equityDataForChart = equity.map(point => [
      point[0] * 1000,
      point[1] - initialCapital
    ]); 
  
  const equityData = equityDataForChart.map(p => p[1])
  let minEquityPnl = Infinity; // Start with the largest possible number

  for (let i = 0; i < equityData.length; i++) {
    if (equityData[i] < minEquityPnl) {
      minEquityPnl = equityData[i];
    }
  }

  const echartsRef = useRef<ReactECharts>(null);

  useEffect(() => {
    // This is a trick to ensure the resize happens after the DOM has settled.
    // The resizable panel needs a render cycle to get its final size.
    const resizeTimer = setTimeout(() => {
      const chartInstance = echartsRef.current?.getEchartsInstance();
      if (chartInstance) {
        chartInstance.resize();
      }
    }, 10); // A tiny delay is often more reliable than 0

    // Cleanup the timer if the component unmounts
    return () => clearTimeout(resizeTimer);
  }, []); // The empty array [] means this effect runs only once on mount

  const chartOption = useMemo(() => ({
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      },
      formatter: function (params: any) {
          // `params` is an array of series data for the hovered point
          // e.g., [ {seriesName: 'Equity', ...}, {seriesName: 'Drawdown', ...} ]
          
          // Get the timestamp from the first series
          const date = new Date(params[0].axisValue);
          const dateString = date.toLocaleDateString(); // e.g., "6/24/2020"
          
          // Start building the HTML string for the tooltip
          let tooltipHtml = `${dateString}<br/>`;
          
          // Loop through each series to format its value
          params.forEach((param: any) => {
              const seriesName = param.seriesName;
              const value = param.value[1]; // The Y-value (equity or drawdown)
              const color = param.color; // The color of the series line

              // The little colored circle marker
              const marker = `<span style="display:inline-block;margin-right:5px;border-radius:10px;width:10px;height:10px;background-color:${color};"></span>`;
              
              if (seriesName === 'Equity') {
                  // For Equity, format to 2 decimal places and add a '$'
                  tooltipHtml += `${marker} ${seriesName}: <b>$${value.toFixed(2)}</b><br/>`;
              } else if (seriesName === 'Drawdown') {
                  // For Drawdown, format to 2 decimal places and add a '%'
                  tooltipHtml += `${marker} ${seriesName}: <b>${value.toFixed(2)}%</b><br/>`;
              } else {
                  // For any other series, just show the value
                  tooltipHtml += `${marker} ${seriesName}: ${value}<br/>`;
              }
          });
          
          return tooltipHtml;
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
        name: 'Equity (Net Profit)',
        // scale: true,
        min: Math.floor(minEquityPnl < 0 ? minEquityPnl * 2 : 0),
        alignTicks: false,
        axisLabel: { formatter: '${value}' },
        splitLine: {show: false},
      },
      {
        type: 'value',
        name: 'Drawdown %',
        // scale: true,
        alignTicks: false,
        max: 0,
        axisLabel: { formatter: '{value}%' },
        splitLine: {show: false}
      }
    ],
    
    grid: { left: '1%', right: '1%', bottom: '10%', containLabel: true },
    
    dataZoom: [
        {
            // This enables zooming and panning inside the chart area
            type: 'inside',
            // Link it to the x-axis (index 0)
            xAxisIndex: [0], 
            // Optional: set a default zoom window on load
            start: isPortfolio ? 80 : 0,
            end: 100
        },
        {
            // This adds a visible slider at the bottom
            type: 'slider',
            xAxisIndex: [0],
            // Style the slider to match your dark theme
            fillerColor: 'rgba(255, 255, 255, 0.2)',
            borderColor: 'rgba(255, 255, 255, 0.2)',
            handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
            handleSize: '70%',
            handleStyle: {
                color: '#fff',
                shadowBlur: 3,
                shadowColor: 'rgba(0, 0, 0, 0.6)',
                shadowOffsetX: 2,
                shadowOffsetY: 2
            },
            textStyle: {
                color: theme.palette.text.secondary
            }
        }
    ],

    series: [
      {
        name: 'Equity',
        type: 'line',
        yAxisIndex: 0,
        showSymbol: false,
        smooth: true,
        data: equityDataForChart,
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
  }), [theme, equityDataForChart, drawdownData]);

  const handleResetZoom = () => {
    // Get the ECharts instance from the ref
    const chartInstance = echartsRef.current?.getEchartsInstance();
    if (chartInstance) {
      // Dispatch a 'dataZoom' action to reset the zoom
      chartInstance.dispatchAction({
        type: 'dataZoom',
        // Specify the start and end percentages
        start: 0,
        end: 100,
      });
    }
  };

  return (
    <Box sx={{ width:'100%', height:'100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>

        <Typography variant="h2" gutterBottom>
          Equity Curve
        </Typography>

        <Tooltip title="Reset Zoom">
          <IconButton onClick={handleResetZoom} size="small">
            <ZoomOutMapIcon />
          </IconButton>
        </Tooltip>
        
      </Box>

      <Box sx={{ flexGrow: 1, width:'100%' }}>
        <ReactECharts ref={echartsRef} option={chartOption} style={{width:'100%', height:'100%'}}/>
      </Box>

    </Box>
  );
};