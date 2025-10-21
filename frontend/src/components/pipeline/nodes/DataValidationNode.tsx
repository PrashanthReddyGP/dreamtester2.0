// src/components/pipeline/nodes/DataValidationNode.tsx

import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import {
    Paper, Typography, Box, Select, MenuItem, FormControl, InputLabel,
    IconButton, CircularProgress, Stack, TextField, Slider
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';
import type { SelectChangeEvent } from '@mui/material/Select';
import { NodeHeader } from './NodeHeader'; // Import the new header

// Define the data structure for this node's configuration
interface DataValidationNodeData {
    label: string;
    validationMethod: 'train_test_split' | 'walk_forward';
    trainSplit: number;
    walkForwardTrainWindow: number;
    walkForwardTestWindow: number;
}

export const DataValidationNode = memo(({ id, data }: NodeProps<DataValidationNodeData>) => {
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    const amIProcessing = isProcessing && processingNodeId === id;

    const handleFieldChange = (fieldName: keyof DataValidationNodeData, value: any) => {
        updateNodeData(id, { [fieldName]: value });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10
    };

    // Style for the output handle text labels
    const labelStyle: React.CSSProperties = {
        position: 'absolute',
        right: 18,
        fontSize: '10px',
        color: '#ccc',
    };
    
    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555'}} />

            {/* Header */}

            <NodeHeader nodeId={id} title={data.label} color="#673ab7">
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

            {/* <Box sx={{ bgcolor: '#673ab7', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="subtitle2" sx={{ color: 'white', pl: 1, fontWeight: 'bold' }}>
                    {data.label || 'Data Validation'}
                </Typography>
                <IconButton
                    size="small"
                    sx={{ color: 'white' }}
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
                <Stack spacing={2.5}>
                    <FormControl fullWidth size="small">
                        <InputLabel>Validation Method</InputLabel>
                        <Select
                            value={data.validationMethod}
                            label="Validation Method"
                            onChange={(e: SelectChangeEvent) => handleFieldChange('validationMethod', e.target.value)}
                        >
                            <MenuItem value="train_test_split">Train/Test Split</MenuItem>
                            <MenuItem value="walk_forward">Walk-Forward</MenuItem>
                        </Select>
                    </FormControl>

                    {data.validationMethod === 'train_test_split' && (
                        <Box sx={{ px: 1 }}>
                            <Typography gutterBottom>Train Split ({data.trainSplit}%)</Typography>
                            <Slider
                                value={data.trainSplit}
                                onChange={(_, newValue) => handleFieldChange('trainSplit', newValue)}
                                aria-labelledby="train-split-slider"
                                valueLabelDisplay="auto"
                                step={5}
                                marks
                                min={10}
                                max={90}
                            />
                        </Box>
                    )}

                    {data.validationMethod === 'walk_forward' && (
                        <Stack spacing={2}>
                            <TextField
                                label="Training Window (days)"
                                type="number"
                                size="small"
                                value={data.walkForwardTrainWindow}
                                onChange={(e) => handleFieldChange('walkForwardTrainWindow', parseInt(e.target.value, 10) || 365)}
                            />
                            <TextField
                                label="Testing Window (days)"
                                type="number"
                                size="small"
                                value={data.walkForwardTestWindow}
                                onChange={(e) => handleFieldChange('walkForwardTestWindow', parseInt(e.target.value, 10) || 30)}
                            />
                        </Stack>
                    )}
                </Stack>
            </Box>
            
            <Handle
                type="source"
                position={Position.Right}
                id="train"
                style={{ ...handleStyle, top: '33%', backgroundColor: 'red' }}
            />
            
            <Handle
                type="source"
                position={Position.Right}
                id="test"
                style={{ ...handleStyle, top: '66%', backgroundColor: 'green' }}
            />
        </Paper>
    );
});