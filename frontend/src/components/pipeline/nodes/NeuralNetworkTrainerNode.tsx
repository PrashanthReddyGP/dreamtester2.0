import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Box, IconButton, TextField, Select, MenuItem, FormControl, InputLabel, Stack, CircularProgress, Typography, Button, Divider } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';

import { usePipeline } from '../../../context/PipelineContext';
import { NodeHeader } from './NodeHeader';

// --- Type Definitions for this specific node ---
interface Layer {
    id: string; // Unique ID for React's key prop
    type: 'Dense' | 'Dropout';
    units?: number;
    activation?: 'relu' | 'sigmoid' | 'tanh';
    rate?: number;
}

interface NNNodeData {
    label: string;
    predictionThreshold: number;
    architecture: {
        layers: Layer[];
    };
    training: {
        optimizer: 'adam' | 'sgd' | 'rmsprop';
        loss: 'binary_crossentropy' | 'mse';
        epochs: number;
        batchSize: number;
        earlyStoppingPatience: number;
    };
}

// --- Layer Editor Sub-component ---
const LayerEditor: React.FC<{ layer: Layer, onUpdate: (updatedLayer: Layer) => void, onRemove: () => void }> = ({ layer, onUpdate, onRemove }) => {
    return (
        <Paper variant="outlined" sx={{ p: 1.5, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" fontWeight="bold">{layer.type} Layer</Typography>
                <IconButton size="small" onClick={onRemove}><DeleteIcon fontSize="small" /></IconButton>
            </Box>
            {layer.type === 'Dense' && (
                <>
                    <TextField
                        label="Units"
                        type="number"
                        size="small"
                        value={layer.units || 64}
                        onChange={(e) => onUpdate({ ...layer, units: parseInt(e.target.value, 10) })}
                    />
                    <FormControl size="small" fullWidth>
                        <InputLabel>Activation</InputLabel>
                        <Select
                            label="Activation"
                            value={layer.activation || 'relu'}
                            onChange={(e) => onUpdate({ ...layer, activation: e.target.value as any })}
                        >
                            <MenuItem value="relu">ReLU</MenuItem>
                            <MenuItem value="tanh">Tanh</MenuItem>
                            <MenuItem value="sigmoid">Sigmoid</MenuItem>
                        </Select>
                    </FormControl>
                </>
            )}
            {layer.type === 'Dropout' && (
                <TextField
                    label="Rate"
                    type="number"
                    size="small"
                    InputProps={{ inputProps: { min: 0, max: 1, step: 0.1 } }}
                    value={layer.rate || 0.5}
                    onChange={(e) => onUpdate({ ...layer, rate: parseFloat(e.target.value) })}
                />
            )}
        </Paper>
    );
};


export const NeuralNetworkTrainerNode = ({ id, data }: NodeProps<NNNodeData>) => {
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    const amIProcessing = isProcessing && processingNodeId === id;

    // --- Handlers for Architecture ---
    const handleLayerUpdate = (index: number, updatedLayer: Layer) => {
        const newLayers = [...data.architecture.layers];
        newLayers[index] = updatedLayer;
        updateNodeData(id, { architecture: { ...data.architecture, layers: newLayers } });
    };

    const handleAddLayer = (type: 'Dense' | 'Dropout') => {
        const newLayerDefaults = {
            id: crypto.randomUUID(),
            type,
            ...(type === 'Dense' && { units: 32, activation: 'relu' }),
            ...(type === 'Dropout' && { rate: 0.2 }),
        };
        const newLayers = [...(data.architecture?.layers || []), newLayerDefaults];
        updateNodeData(id, { architecture: { layers: newLayers } });
    };

    const handleRemoveLayer = (index: number) => {
        const newLayers = data.architecture.layers.filter((_, i) => i !== index);
        updateNodeData(id, { architecture: { layers: newLayers } });
    };

    // --- Handlers for Training Config ---
    const handleTrainingChange = (field: keyof NNNodeData['training'], value: any) => {
        const isNumber = ['epochs', 'batchSize', 'earlyStoppingPatience'].includes(field);
        updateNodeData(id, {
            training: { ...data.training, [field]: isNumber ? Number(value) : value }
        });
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
        <Paper elevation={3} sx={{ borderRadius: 2, width: '400px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} id="train" style={{ ...handleStyle, top: '33%', backgroundColor: 'red' }} />
            <Handle type="target" position={Position.Left} id="test" style={{ ...handleStyle, top: '66%', backgroundColor: 'green' }} />

            <NodeHeader nodeId={id} title={data.label || 'Neural Network'} color="#2439b4ff">
                <IconButton size="small" sx={{ color: 'white' }} onClick={() => executePipelineUpToNode(id)} disabled={amIProcessing}>
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                </IconButton>
            </NodeHeader>

            <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }} className="nodrag" onMouseDown={(e) => e.stopPropagation()}>
                <Typography variant="subtitle2" sx={{color: 'text.secondary', textAlign: 'center'}}>TRAINING CONFIGURATION</Typography>
                <Stack direction="row" spacing={2}>
                    <TextField label="Epochs" type="number" size="small" fullWidth value={data.training?.epochs || 50} onChange={(e) => handleTrainingChange('epochs', e.target.value)} />
                    <TextField label="Batch Size" type="number" size="small" fullWidth value={data.training?.batchSize || 32} onChange={(e) => handleTrainingChange('batchSize', e.target.value)} />
                </Stack>
                <FormControl size="small" fullWidth>
                    <InputLabel>Optimizer</InputLabel>
                    <Select label="Optimizer" value={data.training?.optimizer || 'adam'} onChange={(e) => handleTrainingChange('optimizer', e.target.value)}>
                        <MenuItem value="adam">Adam</MenuItem>
                        <MenuItem value="rmsprop">RMSprop</MenuItem>
                        <MenuItem value="sgd">SGD</MenuItem>
                    </Select>
                </FormControl>
                <TextField
                    label="Prediction Threshold" type="number" size="small" fullWidth
                    value={data.predictionThreshold || 0.5}
                    onChange={handleThresholdChange}
                    InputProps={{ inputProps: { min: 0, max: 1, step: 0.01 } }}
                />

                <Divider sx={{ my: 1 }} />
                <Typography variant="subtitle2" sx={{color: 'text.secondary', textAlign: 'center'}}>MODEL ARCHITECTURE</Typography>
                <Stack spacing={1.5}>
                    {data.architecture?.layers?.map((layer, index) => (
                        <LayerEditor
                            key={layer.id}
                            layer={layer}
                            onUpdate={(updatedLayer) => handleLayerUpdate(index, updatedLayer)}
                            onRemove={() => handleRemoveLayer(index)}
                        />
                    ))}
                </Stack>
                <Stack direction="row" spacing={1} justifyContent="center" sx={{ mt: 1 }}>
                    <Button variant="outlined" size="small" startIcon={<AddIcon />} onClick={() => handleAddLayer('Dense')}>Add Dense</Button>
                    <Button variant="outlined" size="small" startIcon={<AddIcon />} onClick={() => handleAddLayer('Dropout')}>Add Dropout</Button>
                </Stack>
            </Box>
            
            <Handle type="source" position={Position.Right} id="default" style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
};