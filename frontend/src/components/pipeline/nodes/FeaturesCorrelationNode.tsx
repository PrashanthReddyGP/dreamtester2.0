// src/components/pipeline/nodes/FeaturesCorrelationNode.tsx

import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, FormControl, InputLabel, Select, MenuItem, Box, IconButton, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { NodeHeader } from './NodeHeader';
import { usePipeline } from '../../../context/PipelineContext';

// Define the data structure for this specific node
export interface FeaturesCorrelationNodeData {
    label: string;
    method: 'pearson' | 'kendall' | 'spearman';
    displayMode: 'matrix' | 'table';
}

export const FeaturesCorrelationNode = memo(({ id, data, selected }: NodeProps<FeaturesCorrelationNodeData>) => {

    // 2. Pull the necessary state and functions from the context
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    
    // The component determines for itself if it is the one currently being processed
    const amIProcessing = isProcessing && processingNodeId === id;

    const handleMethodChange = (event: any) => {
        updateNodeData(id, { method: event.target.value });
    };

    const handleDisplayModeChange = (event: any) => {
        updateNodeData(id, { displayMode: event.target.value });
    };


    // This node passes data through, so it needs both input and output handles.
    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' }}>

            {/* Input Handle */}
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555'}} />
            
            <NodeHeader nodeId={id} title={data.label} color="#607d8b">
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

            <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }} className="nodrag">
                {/* Method Selection */}
                <FormControl fullWidth size="small">
                    <InputLabel id="correlation-method-label">Method</InputLabel>
                    <Select
                        labelId="correlation-method-label"
                        label="Method"
                        value={data.method || 'pearson'}
                        onChange={handleMethodChange}
                    >
                        <MenuItem value="pearson">Pearson</MenuItem>
                        <MenuItem value="kendall">Kendall</MenuItem>
                        <MenuItem value="spearman">Spearman</MenuItem>
                    </Select>
                </FormControl>

                {/* Display Mode Selection */}
                <FormControl fullWidth size="small">
                    <InputLabel id="display-mode-label">Display Mode</InputLabel>
                    <Select
                        labelId="display-mode-label"
                        label="Display Mode"
                        value={data.displayMode || 'matrix'}
                        onChange={handleDisplayModeChange}
                    >
                        <MenuItem value="matrix">Correlation Matrix</MenuItem>
                        <MenuItem value="table">Table View</MenuItem>
                    </Select>
                </FormControl>
            </Box>
            
            {/* Output Handle */}
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
});