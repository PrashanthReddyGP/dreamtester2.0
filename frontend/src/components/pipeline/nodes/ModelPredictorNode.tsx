import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box, IconButton, Autocomplete, TextField, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';
import { NodeHeader } from './NodeHeader'; // Import the new header

// Define the shape of the data for this specific node
interface ModelPredictorData {
    label: string;
    trainerNodeId?: string; // The ID of the trainer node to use
}

export const ModelPredictorNode = ({ id, data }: NodeProps<ModelPredictorData>) => {
    const { nodes, updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    const amIProcessing = isProcessing && processingNodeId === id;

    // Find all available Model Trainer nodes in the workflow to create dropdown options
    const trainerNodeOptions = nodes
        .filter(n => n.type === 'modelTrainer')
        .map(n => ({
            id: n.id,
            // Create a user-friendly label for the dropdown
            label: n.data.label ? `${n.data.label} (${n.data.modelName})` : `Trainer (${n.data.modelName})`
        }));
    
    // Find the full object for the currently selected trainer to show in the Autocomplete
    const selectedTrainer = trainerNodeOptions.find(n => n.id === data.trainerNodeId);

    const handleTrainerChange = (event: any, selectedOption: { id: string; label: string } | null) => {
        // Update this node's data with the ID of the chosen trainer node
        updateNodeData(id, { trainerNodeId: selectedOption?.id || null });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10,
    };
    
    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555'}}>
            {/* This handle is for the DATA input */}
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555' }} />


            <NodeHeader nodeId={id} title={data.label} color="#4caf50">
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
            
            {/* <Box sx={{ bgcolor: '#4caf50', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="subtitle2" sx={{ color: 'primary.contrastText', pl: 1 }}>
                    {data.label || 'Model Predictor'}
                </Typography>
                <IconButton size="small" sx={{ color: 'white' }} onClick={() => executePipelineUpToNode(id)} disabled={amIProcessing}>
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon />}
                </IconButton>
            </Box> */}

            <Box sx={{ p: 2 }} className="nodrag">
                <Autocomplete
                    options={trainerNodeOptions}
                    getOptionLabel={(option) => option.label}
                    value={selectedTrainer || null}
                    onChange={handleTrainerChange}
                    disableClearable
                    size="small"
                    renderInput={(params) => <TextField {...params} label="Select Trained Model" />}
                />
            </Box>
            
            {/* This handle is for the DATA output (with predictions added) */}
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
};