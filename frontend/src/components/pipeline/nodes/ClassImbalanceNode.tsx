import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box, Select, MenuItem, FormControl, InputLabel } from '@mui/material';

import { usePipeline } from '../../../context/PipelineContext';
import { NodeHeader } from './NodeHeader';

interface ClassImbalanceNodeData {
    label: string;
    method: 'SMOTE'; // For now, only SMOTE is supported
}

export const ClassImbalanceNode = ({ id, data }: NodeProps<ClassImbalanceNodeData>) => {
    const { updateNodeData } = usePipeline();

    const handleMethodChange = (event: any) => {
        updateNodeData(id, { method: event.target.value });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10,
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: 'red'}} />

            <NodeHeader nodeId={id} title={data.label || 'Class Imbalance'} color="#4caf50" />

            <Box
                sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <Typography variant="body2" sx={{ mb: 1 }}>
                    Applies a resampling technique to the training data to fix class imbalance.
                </Typography>
                <FormControl fullWidth size="small">
                    <InputLabel>Method</InputLabel>
                    <Select
                        label="Method"
                        value={data.method || 'SMOTE'}
                        onChange={handleMethodChange}
                    >
                        <MenuItem value="SMOTE">SMOTE (Oversampling)</MenuItem>
                        {/* You can add other methods like RandomUnderSampler here later */}
                    </Select>
                </FormControl>
            </Box>

            <Handle type="source" position={Position.Right} id="train" style={{ ...handleStyle, backgroundColor: 'red'}} />
        </Paper>
    );
};