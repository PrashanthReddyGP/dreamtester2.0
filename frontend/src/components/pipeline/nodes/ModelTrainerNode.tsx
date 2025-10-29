import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Box, IconButton, Autocomplete, TextField, Select, MenuItem, FormControl, InputLabel, Stack, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

import { usePipeline } from '../../../context/PipelineContext';
import { HYPERPARAMETER_CONFIG } from '../mlModels';
import { NodeHeader } from './NodeHeader'; // Import the new header

// Type definitions for clarity
type ParamDef = { name: string; label: string; type: 'number' | 'text' | 'select'; options?: string[]; defaultValue: any; };

interface MLModelNodeData {
    label: string;
    modelName: string;
    hyperparameters: { [key: string]: any };
    predictionThreshold: number;
}

// Sub-component to render the correct input based on param type
const HyperparameterInput: React.FC<{ paramDef: ParamDef; value: any; onChange: (e: any) => void; }> = ({ paramDef, value, onChange }) => {
    switch (paramDef.type) {
        case 'select':
            return (
                <FormControl fullWidth size="small">
                    <InputLabel>{paramDef.label}</InputLabel>
                    <Select label={paramDef.label} value={value || ''} onChange={onChange}>
                        {paramDef.options?.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </Select>
                </FormControl>
            );
        case 'number':
            return (
                <TextField
                    fullWidth
                    type="number"
                    size="small"
                    variant="outlined"
                    label={paramDef.label}
                    value={value || ''}
                    onChange={onChange}
                />
            );
        case 'text':
        default:
            return (
                <TextField
                    fullWidth
                    size="small"
                    variant="outlined"
                    label={paramDef.label}
                    value={value || ''}
                    onChange={onChange}
                />
            );
    }
};

export const ModelTrainerNode = ({ id, data }: NodeProps<MLModelNodeData>) => {
    // 1. PULL THE REQUIRED FUNCTIONS AND STATE FROM THE CONTEXT
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    const modelOptions = Object.keys(HYPERPARAMETER_CONFIG).filter(
        model => model !== 'BaggingClassifier'
    );
    const currentModelConfig = HYPERPARAMETER_CONFIG[data.modelName] || [];
    
    // 2. DETERMINE THE PROCESSING STATE FOR THIS SPECIFIC NODE
    const amIProcessing = isProcessing && processingNodeId === id;
    
    const handleModelChange = (event: any, newModelName: string | null) => {
        if (!newModelName || !HYPERPARAMETER_CONFIG[newModelName]) return;

        const config = HYPERPARAMETER_CONFIG[newModelName];
        const defaultHyperparameters = config.reduce((acc, param) => {
            acc[param.name] = param.defaultValue;
            return acc;
        }, {} as { [key: string]: any });

        updateNodeData(id, {
            modelName: newModelName,
            hyperparameters: defaultHyperparameters,
        });
    };

    const handleParamChange = (paramName: string, value: any, type: 'number' | 'text' | 'select') => {
        let finalValue = value;
        if (type === 'number') {
            finalValue = Number(value); // Ensure it's stored as a number
        }
        const updatedParams = { ...data.hyperparameters, [paramName]: finalValue };
        updateNodeData(id, { hyperparameters: updatedParams });
    };

    const handleThresholdChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = Math.max(0, Math.min(1, Number(event.target.value))); // Clamp between 0 and 1
        updateNodeData(id, { predictionThreshold: value });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10,
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '350px', border: '1px solid #555' /* Indigo border */ }}>
            <Handle
                type="target"
                position={Position.Left}
                id="train"
                style={{ ...handleStyle, top: '33%', backgroundColor: 'red' }}
            />
            
            <Handle
                type="target"
                position={Position.Left}
                id="test"
                style={{ ...handleStyle, top: '66%', backgroundColor: 'green' }}
            />

            <NodeHeader nodeId={id} title={data.label} color="#c4592aff">
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

            {/* <Box sx={{ bgcolor: '#c4592aff', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="subtitle2" sx={{ color: 'primary.contrastText', pl: 1 }}>
                    {data.label || 'ML Model'}
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

            <Box
                sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <Autocomplete
                    options={modelOptions}
                    value={data.modelName}
                    onChange={handleModelChange}
                    disableClearable
                    size="small"
                    renderInput={(params) => <TextField {...params} label="Select Model" />}
                />
                <TextField
                    fullWidth
                    type="number"
                    size="small"
                    variant="outlined"
                    label="Prediction Threshold"
                    value={data.predictionThreshold || 0.5}
                    onChange={handleThresholdChange}
                    InputProps={{
                        inputProps: { 
                            min: 0, 
                            max: 1, 
                            step: 0.01 
                        }
                    }}
                />
                {currentModelConfig.length > 0 && (
                    <Stack spacing={2}>
                        {currentModelConfig.map(paramDef => (
                            <HyperparameterInput
                                key={paramDef.name}
                                paramDef={paramDef}
                                value={data.hyperparameters?.[paramDef.name]}
                                onChange={(e) => handleParamChange(paramDef.name, e.target.value, paramDef.type)}
                            />
                        ))}
                    </Stack>
                )}
            </Box>
            
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
};