// src/components/pipeline/nodes/DataScalingNode.tsx

import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import {
    Paper, Box, Select, MenuItem, FormControl, InputLabel, IconButton,
    CircularProgress, Stack, Slider, FormGroup, FormControlLabel, Switch, TextField
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';
import type { SelectChangeEvent } from '@mui/material/Select';
import { NodeHeader } from './NodeHeader';

// Define the data structure for this node's configuration
interface DataScalingNodeData {
    label: string;
    scaler: 'none' | 'standard' | 'min_max' | 'robust';
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
        zIndex: 10,
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, background: '#555' }} />

            <NodeHeader nodeId={id} title={data.label} color="#bb2b4aff">
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

            <Box
                sx={{ p: 2.5 }}
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <Stack spacing={1.0}>
                    <FormControl fullWidth size="small">
                        <InputLabel>Scaler</InputLabel>
                        <Select
                            value={data.scaler}
                            label="Scaler"
                            onChange={(e: SelectChangeEvent) => handleFieldChange('scaler', e.target.value)}
                        >
                            <MenuItem value="none">None</MenuItem>
                            <MenuItem value="standard">Standard Scaler</MenuItem>
                            <MenuItem value="min_max">Min-Max Scaler</MenuItem>
                            <MenuItem value="robust">Robust Scaler</MenuItem>
                        </Select>
                    </FormControl>

                    <FormGroup>
                        <FormControlLabel
                            control={<Switch checked={data.removeCorrelated} onChange={(e) => handleFieldChange('removeCorrelated', e.target.checked)} />}
                            label="Remove Correlated Features"
                        />
                    </FormGroup>
                    
                    {data.removeCorrelated && (
                        <Box sx={{ px: 1 }}>
                            <InputLabel shrink>Correlation Threshold</InputLabel>
                            <Slider
                                value={data.correlationThreshold}
                                onChange={(_, newValue) => handleFieldChange('correlationThreshold', newValue)}
                                aria-labelledby="correlation-threshold-slider"
                                valueLabelDisplay="auto"
                                step={0.05}
                                marks
                                min={0.5}
                                max={1.0}
                            />
                        </Box>
                    )}

                    <FormGroup>
                        <FormControlLabel
                            control={<Switch checked={data.usePCA} onChange={(e) => handleFieldChange('usePCA', e.target.checked)} />}
                            label="Use PCA"
                        />
                    </FormGroup>

                    {data.usePCA && (
                        <TextField
                            label="PCA Components"
                            type="number"
                            size="small"
                            value={data.pcaComponents}
                            onChange={(e) => handleFieldChange('pcaComponents', parseInt(e.target.value, 10) || 1)}
                            inputProps={{ min: 1 }}
                        />
                    )}
                </Stack>
            </Box>

            <Handle
                type="source"
                position={Position.Right}
                id="output"
                style={{ ...handleStyle, background: '#555' }}
            />
        </Paper>
    );
});