// src/components/pipeline/nodes/ChartingNode.tsx

import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import { usePipeline } from '../../../context/PipelineContext';
import { NodeHeader } from './NodeHeader'; // Import the new header

interface ChartingNodeData {
    label: string;
    chartType: 'scatter' | 'line' | 'histogram';
    xAxis: string | null;
    yAxis: string | null;
    groupBy: string | null;
}

export const ChartingNode = memo(({ id, data, selected }: NodeProps<ChartingNodeData>) => {
    // This node doesn't have a run button; it updates live when configured in the side panel.
    const handleStyle = { width: 12, height: 12, background: '#00bcd4', border: '1px solid #555' };

    return (
        <Paper 
            elevation={selected ? 6 : 3} 
            sx={{ 
                borderRadius: 2, 
                width: '200px', 
                border: selected ? '2px solid #00bcd4' : '1px solid #555',
                transition: 'border 0.2s ease-in-out'
            }}
        >
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555'}} />

            <Box sx={{ bgcolor: '#00bcd4', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', alignItems: 'center', gap: 1 }}>
                <AutoGraphIcon sx={{ color: 'white' }} />
                <Typography variant="subtitle2" sx={{ color: 'white', fontWeight: 'bold' }}>
                    {data.label || 'Charting'}
                </Typography>
            </Box>
            <Box sx={{ p: 2 }}>
                <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', textAlign: 'center' }}>
                    Select this node to configure the chart in the Properties panel.
                </Typography>
            </Box>
            
            {/* This node doesn't modify data, so it has no output handle */}
        </Paper>
    );
});