import React from 'react';
import { Box, Typography } from '@mui/material';

// The shape of the data for a single item in the legend
export interface LegendData {
    name: string;
    value: string; // Formatted value as a string
    color: string;
    }
    
    interface ChartLegendProps {
    legendData: LegendData[];
    }

    export const ChartLegend: React.FC<ChartLegendProps> = ({ legendData }) => {
    if (legendData.length === 0) {
        return null; // Don't render anything if there's no data
    }

    return (
        <Box
        sx={{
            position: 'absolute',
            top: '12px',
            left: '12px',
            zIndex: 1000,
            pointerEvents: 'none', // Allow mouse events to pass through to the chart
            display: 'flex',
            gap: '12px',
            flexWrap: 'wrap',
        }}
        >
        {legendData.map(item => (
            <Box key={item.name} sx={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Typography sx={{ color: item.color, fontSize: '12px', fontWeight: 'bold' }}>
                {item.name}
            </Typography>
            <Typography sx={{ color: '#D1D4DC', fontSize: '12px' }}>
                {item.value}
            </Typography>
            </Box>
        ))}
        </Box>
    );
};