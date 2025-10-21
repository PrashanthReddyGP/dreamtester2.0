// src/components/pipeline/analysis/GenericChartDisplay.tsx

import React from 'react';
import {
    ScatterChart, Scatter, LineChart, Line, BarChart, Bar,
    XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Typography } from '@mui/material';

// Same color palette as before
const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088FE', '#00C49F', '#FFBB28'];

interface GenericChartProps {
    chartType: 'scatter' | 'line' | 'histogram';
    data: any[];
    config: {
        xAxis: string;
        yAxis?: string;
        groupBy?: string;
    };
}

export const GenericChartDisplay: React.FC<GenericChartProps> = ({ chartType, data, config }) => {
    if (!data || data.length === 0) return <Typography>No data to plot.</Typography>;

    const renderScatterPlot = () => {
        // Find the unique group names if a groupBy key is provided
        const groupKeys = config.groupBy ? [...new Set(data.map(p => p[config.groupBy!]))] : ['default'];

        return (
            <ScatterChart>
                <CartesianGrid />
                <XAxis type="number" dataKey={config.xAxis} name={config.xAxis} tick={{ fill: '#ccc' }} />
                <YAxis type="number" dataKey={config.yAxis} name={config.yAxis} tick={{ fill: '#ccc' }} />
                
                {/* 2. Use ZAxis to control the color/grouping */}
                {/* It maps a categorical value (the groupBy column) to a color fill */}
                {config.groupBy && <ZAxis dataKey={config.groupBy} name={config.groupBy} />}

                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />

                {/* 3. Render a SINGLE <Scatter> component for each group */}
                {/* This is much more efficient than rendering one giant one */}
                {groupKeys.map((key, index) => (
                    <Scatter
                        key={String(key)}
                        name={String(key)}
                        // Filter the data for this specific group
                        data={config.groupBy ? data.filter(p => p[config.groupBy!] === key) : data}
                        // Assign a color from our palette
                        fill={COLORS[index % COLORS.length]}
                    />
                ))}
            </ScatterChart>
        );
    };
    
    const renderLineChart = () => (
        <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={config.xAxis} tick={{ fill: '#ccc' }} />
            <YAxis tick={{ fill: '#ccc' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey={config.yAxis} stroke="#8884d8" activeDot={{ r: 8 }} />
        </LineChart>
    );

    const renderHistogram = () => (
        <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bin" tick={{ fill: '#ccc' }} />
            <YAxis tick={{ fill: '#ccc' }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#82ca9d" />
        </BarChart>
    );
    
    let chart;
    switch (chartType) {
        case 'scatter':
            if (!config.xAxis || !config.yAxis) {
                chart = <Typography color="text.secondary">Please select both an X and Y axis.</Typography>;
            } else {
                chart = renderScatterPlot();
            }
            break;
        case 'line':
            if (!config.xAxis || !config.yAxis) {
                chart = <Typography color="text.secondary">Please select both an X and Y axis.</Typography>;
            } else {
                chart = renderLineChart();
            }
            break;
        case 'histogram':
            if (!config.xAxis) {
                chart = <Typography color="text.secondary">Please select an X axis for the histogram.</Typography>;
            } else {
                chart = renderHistogram();
            }
            break;
        default:
            chart = <Typography>Unknown chart type.</Typography>;
    }


    return (
        <ResponsiveContainer width="100%" height="100%">
            {chart}
        </ResponsiveContainer>
    );
};