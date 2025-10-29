// src/components/pipeline/analysis/GenericChartDisplay.tsx

import React from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis,
  Label
} from 'recharts';

interface ChartDisplayProps {
  chartType: 'bar' | 'scatter';
  data: any[];
  config: {
    xAxis: string;
    yAxis: string;
    zAxis?: string; // For bubble charts in scatter plots
    xAxisLabel?: string;
    yAxisLabel?: string;
    groupBy?: string;
  };
}

// A helper to generate distinct colors for grouped data
const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088FE', '#00C49F', '#FFBB28'];

export const GenericChartDisplay: React.FC<ChartDisplayProps> = ({ chartType, data, config }) => {
  const theme = useTheme();
  const tickColor = theme.palette.text.secondary;

  if (!data || data.length === 0) {
    return <Typography>No data available for charting.</Typography>;
  }

  const renderChart = () => {
    switch (chartType) {
      case 'bar':
        return (
          <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
            <XAxis dataKey={config.xAxis} stroke={tickColor} tick={{ fontSize: 12 }}>
              <Label value={config.xAxisLabel} offset={-15} position="insideBottom" fill={tickColor} />
            </XAxis>
            <YAxis stroke={tickColor} tick={{ fontSize: 12 }}>
               <Label value={config.yAxisLabel} angle={-90} position="insideLeft" fill={tickColor} style={{ textAnchor: 'middle' }} />
            </YAxis>
            <Tooltip
              contentStyle={{
                backgroundColor: theme.palette.background.paper,
                borderColor: theme.palette.divider,
              }}
            />
            {/* <Legend /> */}
            <Bar dataKey={config.yAxis} fill={theme.palette.primary.main} />
          </BarChart>
        );

      case 'scatter':
        // Logic for handling grouped vs. non-grouped scatter plots
        if (config.groupBy && config.groupBy in data[0]) {
            const groups = [...new Set(data.map(item => item[config.groupBy]))];
            return (
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid stroke={theme.palette.divider} />
                    <XAxis type="number" dataKey={config.xAxis} name={config.xAxisLabel || config.xAxis} stroke={tickColor}>
                         <Label value={config.xAxisLabel} offset={-15} position="insideBottom" fill={tickColor} />
                    </XAxis>
                    <YAxis type="number" dataKey={config.yAxis} name={config.yAxisLabel || config.yAxis} stroke={tickColor}>
                        <Label value={config.yAxisLabel} angle={-90} position="insideLeft" fill={tickColor} style={{ textAnchor: 'middle' }} />
                    </YAxis>
                    {config.zAxis && <ZAxis type="number" dataKey={config.zAxis} range={[20, 200]} name={config.zAxis} />}
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: theme.palette.background.paper, borderColor: theme.palette.divider }}/>
                    <Legend />
                    {groups.map((group, index) => (
                        <Scatter 
                            key={String(group)} 
                            name={String(group)} 
                            data={data.filter(d => d[config.groupBy] === group)} 
                            fill={COLORS[index % COLORS.length]} 
                        />
                    ))}
                </ScatterChart>
            );
        }
        // Fallback for non-grouped scatter
        return (
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid stroke={theme.palette.divider}/>
                <XAxis type="number" dataKey={config.xAxis} name={config.xAxisLabel || config.xAxis} stroke={tickColor}>
                    <Label value={config.xAxisLabel} offset={-15} position="insideBottom" fill={tickColor} />
                </XAxis>
                <YAxis type="number" dataKey={config.yAxis} name={config.yAxisLabel || config.yAxis} stroke={tickColor}>
                    <Label value={config.yAxisLabel} angle={-90} position="insideLeft" fill={tickColor} style={{ textAnchor: 'middle' }} />
                </YAxis>
                {config.zAxis && <ZAxis type="number" dataKey={config.zAxis} range={[20, 200]} name={config.zAxis} />}
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: theme.palette.background.paper, borderColor: theme.palette.divider }}/>
                <Scatter name="Points" data={data} fill={theme.palette.primary.main} />
            </ScatterChart>
        );

      default:
        return <Typography>Unsupported chart type: {chartType}</Typography>;
    }
  };

  return (
    <Box sx={{ width: '100%', height: '100%', minHeight: 300 }}>
      <ResponsiveContainer>
        {renderChart()}
      </ResponsiveContainer>
    </Box>
  );
};