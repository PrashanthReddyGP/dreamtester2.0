import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Box, IconButton, Autocomplete, TextField, Select, MenuItem, FormControl, InputLabel, Stack, CircularProgress, Typography, Divider } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

import { usePipeline } from '../../../context/PipelineContext';
import { HYPERPARAMETER_CONFIG } from '../mlModels';
import { NodeHeader } from './NodeHeader';

// Type definitions for clarity
type ParamDef = { name: string; label: string; type: 'number' | 'text' | 'select'; options?: string[]; defaultValue: any; };

interface BaggingTrainerNodeData {
    label: string;
    // Bagging-specific params
    baggingHyperparameters: { [key: string]: any };
    // Base model configuration
    baseModelName: string;
    baseModelHyperparameters: { [key: string]: any };
    predictionThreshold: number;
}

// Re-usable input component from ModelTrainerNode
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
            return <TextField fullWidth type="number" size="small" variant="outlined" label={paramDef.label} value={value || ''} onChange={onChange} />;
        default:
            return <TextField fullWidth size="small" variant="outlined" label={paramDef.label} value={value || ''} onChange={onChange} />;
    }
};


export const BaggingTrainerNode = ({ id, data }: NodeProps<BaggingTrainerNodeData>) => {
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    
    // Filter out BaggingClassifier itself from the list of base models
    const baseModelOptions = Object.keys(HYPERPARAMETER_CONFIG).filter(m => m !== 'BaggingClassifier');
    
    const baggingConfig = HYPERPARAMETER_CONFIG['BaggingClassifier'] || [];
    const currentBaseModelConfig = HYPERPARAMETER_CONFIG[data.baseModelName] || [];
    
    const amIProcessing = isProcessing && processingNodeId === id;
    
    const handleBaseModelChange = (event: any, newModelName: string | null) => {
        if (!newModelName || !HYPERPARAMETER_CONFIG[newModelName]) return;

        const config = HYPERPARAMETER_CONFIG[newModelName];
        const defaultHyperparameters = config.reduce((acc, param) => {
            acc[param.name] = param.defaultValue;
            return acc;
        }, {} as { [key: string]: any });

        updateNodeData(id, {
            baseModelName: newModelName,
            baseModelHyperparameters: defaultHyperparameters,
        });
    };
    
    // Handler for Bagging's own parameters
    const handleBaggingParamChange = (paramName: string, value: any, type: string) => {
        const finalValue = type === 'number' ? Number(value) : value;
        const updatedParams = { ...data.baggingHyperparameters, [paramName]: finalValue };
        updateNodeData(id, { baggingHyperparameters: updatedParams });
    };

    // Handler for the base model's parameters
    const handleBaseParamChange = (paramName: string, value: any, type: string) => {
        const finalValue = type === 'number' ? Number(value) : value;
        const updatedParams = { ...data.baseModelHyperparameters, [paramName]: finalValue };
        updateNodeData(id, { baseModelHyperparameters: updatedParams });
    };

    const handleThresholdChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = Math.max(0, Math.min(1, Number(event.target.value)));
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
        <Paper elevation={3} sx={{ borderRadius: 2, width: '350px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} id="train" style={{ ...handleStyle, top: '33%', backgroundColor: 'red' }} />
            <Handle type="target" position={Position.Left} id="test" style={{ ...handleStyle, top: '66%', backgroundColor: 'green' }} />

            <NodeHeader nodeId={id} title={data.label} color="#2a8c4a">
                <IconButton size="small" sx={{ color: 'white' }} onClick={() => executePipelineUpToNode(id)} disabled={amIProcessing}>
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                </IconButton>
            </NodeHeader>

            <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }} className="nodrag">
                <Autocomplete
                    options={baseModelOptions}
                    value={data.baseModelName}
                    onChange={handleBaseModelChange}
                    disableClearable
                    size="small"
                    renderInput={(params) => <TextField {...params} label="Select Base Model" />}
                />

                <Divider sx={{ my: 1 }}><Typography variant="caption">Bagging Parameters</Typography></Divider>
                <Stack spacing={2}>
                    {baggingConfig.map(paramDef => (
                        <HyperparameterInput
                            key={paramDef.name}
                            paramDef={paramDef}
                            value={data.baggingHyperparameters?.[paramDef.name]}
                            onChange={(e) => handleBaggingParamChange(paramDef.name, e.target.value, paramDef.type)}
                        />
                    ))}
                </Stack>
                
                <Divider sx={{ my: 1 }}><Typography variant="caption">Base Model: {data.baseModelName}</Typography></Divider>
                {currentBaseModelConfig.length > 0 && (
                    <Stack spacing={2}>
                        {currentBaseModelConfig.map(paramDef => (
                            <HyperparameterInput
                                key={paramDef.name}
                                paramDef={paramDef}
                                value={data.baseModelHyperparameters?.[paramDef.name]}
                                onChange={(e) => handleBaseParamChange(paramDef.name, e.target.value, paramDef.type)}
                            />
                        ))}
                    </Stack>
                )}

                <TextField
                    fullWidth type="number" size="small" variant="outlined"
                    label="Prediction Threshold"
                    value={data.predictionThreshold || 0.5}
                    onChange={handleThresholdChange}
                    InputProps={{ inputProps: { min: 0, max: 1, step: 0.01 } }}
                />

            </Box>
            
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
};