import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, IconButton, Box, Select, MenuItem, FormControl, InputLabel, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';
import { NodeHeader } from './NodeHeader'; // Import the new header

// Define the shape of the data for this specific node
interface MergeNodeData {
    label: string;
    mergeMethod: 'inner' | 'outer' | 'left' | 'right';
}

export const MergeNode = ({ id, data }: NodeProps<MergeNodeData>) => {
    // 2. Pull the necessary state and functions from the context
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    
    // The component determines for itself if it is the one currently being processed
    const amIProcessing = isProcessing && processingNodeId === id;

    const handleMethodChange = (event: any) => {
        updateNodeData(id, { mergeMethod: event.target.value });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '250px', border: '1px solid #555' }}>
            {/* This node requires TWO inputs */}
            <Handle type="target" position={Position.Left} id="a" style={{ ...handleStyle, top: '33%', backgroundColor: '#555'}} />
            <Handle type="target" position={Position.Left} id="b" style={{ ...handleStyle, top: '66%', backgroundColor: '#555'}} />

            <NodeHeader nodeId={id} title={data.label} color="#ff9800" textColor='black'>
                <IconButton 
                    size="small" 
                    sx={{ color: 'black' }} 
                    aria-label="run" 
                    onClick={() => executePipelineUpToNode(id)}
                    disabled={amIProcessing}
                >
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                </IconButton>
            </NodeHeader>
            
            <Box sx={{ p: 2 }} className="nodrag">
                <FormControl fullWidth size="small">
                    <InputLabel id="merge-method-label">Merge Method</InputLabel>
                    <Select
                        labelId="merge-method-label"
                        label="Merge Method"
                        value={data.mergeMethod || 'left'}
                        onChange={handleMethodChange}
                    >
                        <MenuItem value="inner">Inner (Intersection)</MenuItem>
                        <MenuItem value="outer">Outer (Union)</MenuItem>
                        <MenuItem value="left">Left</MenuItem>
                        <MenuItem value="right">Right</MenuItem>
                    </Select>
                </FormControl>
            </Box>

            <Handle
                type="source"
                position={Position.Right}
                style={{ ...handleStyle, backgroundColor: '#555' }}
            />
        </Paper>
    );
};