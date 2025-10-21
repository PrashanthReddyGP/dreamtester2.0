// src/components/pipeline/nodes/DataScalingNode.tsx

import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import {
    Paper, Typography, Box, Select, MenuItem, FormControl, InputLabel,
    Checkbox, FormControlLabel, TextField, IconButton, CircularProgress, Stack
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';
import type { SelectChangeEvent } from '@mui/material/Select';
import { NodeHeader } from './NodeHeader'; // Import the new header

// Define the data structure for this node's configuration
interface DataScalingNodeData {
    label: string;
    scaler: 'none' | 'StandardScaler' | 'MinMaxScaler';
    removeCorrelated: boolean;
    correlationThreshold: number;
    usePCA: boolean;
    pcaComponents: number;
}

export const DataScalingNode = memo(({ id, data }: NodeProps<DataScalingNodeData>) => {
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    const amIProcessing = isProcessing && processingNodeId === id;

    const handleFieldChange = (fieldName: keyof DataScalingNodeData, value: any) => {
        updateNodeData(id, { [fieldName]: value });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10
    };
    
    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555'}} />

            {/* Header */}

            <NodeHeader nodeId={id} title={data.label} color="#df6413ff">
                <IconButton 
                    size="small" 
                    sx={{ color: 'white' }} 
                    aria-label="run" 
                    onClick={() => executePipelineUpToNode(id)}
                    disabled={amIProcessing}
                >
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                </IconButton>
            </NodeHeader>

            {/* <Box sx={{ bgcolor: '#e1cb28ff', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="subtitle2" sx={{ color: 'black', pl: 1 }}>
                    {data.label || 'Data Scaling & Preprocessing'}
                </Typography>
                <IconButton
                    size="small"
                    sx={{ color: 'black' }}
                    aria-label="run"
                    onClick={() => executePipelineUpToNode(id)}
                    disabled={amIProcessing}
                >
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon />}
                </IconButton>
            </Box> */}

            {/* Content Body */}
            <Box
                sx={{ p: 2 }}
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <Stack spacing={0}>
                    {/* Scaler Selection */}
                    <FormControl fullWidth size="small">
                        <InputLabel>Scaler</InputLabel>
                        <Select
                            value={data.scaler}
                            label="Scaler"
                            onChange={(e: SelectChangeEvent) => handleFieldChange('scaler', e.target.value)}
                        >
                            <MenuItem value="none">None</MenuItem>
                            <MenuItem value="StandardScaler">Standard Scaler</MenuItem>
                            <MenuItem value="MinMaxScaler">Min-Max Scaler</MenuItem>
                        </Select>
                    </FormControl>

                    {/* Remove Correlated Features */}
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={data.removeCorrelated}
                                onChange={(e) => handleFieldChange('removeCorrelated', e.target.checked)}
                            />
                        }
                        label="Remove Correlated Features"
                    />
                    {data.removeCorrelated && (
                        <TextField
                            label="Correlation Threshold"
                            type="number"
                            size="small"
                            value={data.correlationThreshold}
                            onChange={(e) => handleFieldChange('correlationThreshold', parseFloat(e.target.value) || 0.9)}
                            inputProps={{ step: "0.05", min: "0", max: "1" }}
                        />
                    )}

                    {/* PCA */}
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={data.usePCA}
                                onChange={(e) => handleFieldChange('usePCA', e.target.checked)}
                            />
                        }
                        label="Use PCA"
                    />
                    {data.usePCA && (
                        <TextField
                            label="PCA Components"
                            type="number"
                            size="small"
                            value={data.pcaComponents}
                            onChange={(e) => handleFieldChange('pcaComponents', parseInt(e.target.value, 10) || 5)}
                            inputProps={{ step: "1", min: "1" }}
                        />
                    )}
                </Stack>
            </Box>
            
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
});