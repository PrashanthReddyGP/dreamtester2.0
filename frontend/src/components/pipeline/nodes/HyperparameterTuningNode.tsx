import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box, IconButton, Autocomplete, TextField, Select, MenuItem, FormControl, InputLabel, Stack, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

import { usePipeline } from '../../../context/PipelineContext';
import { TUNING_GRID_CONFIG } from '../mlModels';
import { NodeHeader } from './NodeHeader';

// The base metrics the user can choose from.
const BASE_SCORING_METRICS = [
    'accuracy',
    'f1',
    'precision',
    'recall',
    'roc_auc_ovr', // One-vs-Rest for multi-class AUC
    'r2',
    'neg_mean_squared_error',
];

// A set to quickly check which metrics require an averaging strategy.
const AVERAGING_METRICS = new Set(['f1', 'precision', 'recall']);

// The available averaging options.
const AVERAGING_OPTIONS = ['weighted', 'macro', 'micro', 'per-class'];

// --- CHANGE 1: Update node data interface for clarity ---
interface HyperparameterTuningNodeData {
    label: string;
    modelName: string;
    searchStrategy: 'GridSearchCV' | 'RandomizedSearchCV';
    cvFolds: number;
    scoringMetricBase: string;    // e.g., 'f1'
    scoringMetricAvg: string;     // e.g., 'weighted', 'per-class'
    scoringMetricClass: string;   // e.g., '1' (only if avg is 'per-class')
    paramGrid: { [key: string]: string };
}

type ParamDef = { name: string; label: string; type: 'number' | 'text' | 'select'; options?: string[]; defaultValue: string; };

const HyperparameterGridInput: React.FC<{
    paramDef: ParamDef;
    value: string;
    onChange: (e: any) => void;
}> = ({ paramDef, value, onChange }) => (
    <TextField
        fullWidth
        size="small"
        variant="outlined"
        label={paramDef.label}
        value={value || ''}
        onChange={onChange}
    />
);

export const HyperparameterTuningNode = ({ id, data }: NodeProps<HyperparameterTuningNodeData>) => {
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    const modelOptions = Object.keys(TUNING_GRID_CONFIG);
    const currentModelConfig = TUNING_GRID_CONFIG[data.modelName] || [];
    
    const amIProcessing = isProcessing && processingNodeId === id;
    
    // --- CHANGE 2: Update conditional visibility logic ---
    const showAveragingDropdown = AVERAGING_METRICS.has(data.scoringMetricBase);
    const showClassInput = showAveragingDropdown && data.scoringMetricAvg === 'per-class';

    const handleModelChange = (event: any, newModelName: string | null) => {
        if (!newModelName || !TUNING_GRID_CONFIG[newModelName]) return;
        const config = TUNING_GRID_CONFIG[newModelName];
        const defaultParamGrid = config.reduce((acc, param) => {
            acc[param.name] = param.defaultValue;
            return acc;
        }, {} as { [key: string]: string });
        updateNodeData(id, { modelName: newModelName, paramGrid: defaultParamGrid });
    };

    const handleGridChange = (paramName: string, value: string) => {
        updateNodeData(id, { paramGrid: { ...data.paramGrid, [paramName]: value } });
    };

    const handleFieldChange = (fieldName: keyof HyperparameterTuningNodeData, value: any) => {
        updateNodeData(id, { [fieldName]: value });
    };
    
    const handleMetricBaseChange = (newMetricBase: string) => {
        const updates: Partial<HyperparameterTuningNodeData> = { scoringMetricBase: newMetricBase };
        // If the new metric does NOT need averaging, clear the averaging and class fields.
        if (!AVERAGING_METRICS.has(newMetricBase)) {
            updates.scoringMetricAvg = '';
            updates.scoringMetricClass = '';
        } else {
            // Default to 'weighted' if switching to a metric that needs averaging
            updates.scoringMetricAvg = 'weighted';
        }
        updateNodeData(id, updates);
    };

    const handleAvgChange = (newAvg: string) => {
        const updates: Partial<HyperparameterTuningNodeData> = { scoringMetricAvg: newAvg };
        // If the new average is NOT 'per-class', clear the class field.
        if (newAvg !== 'per-class') {
            updates.scoringMetricClass = '';
        }
        updateNodeData(id, updates);
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
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555'}} />

            <NodeHeader nodeId={id} title={data.label} color="#2a7c4f">
                <IconButton size="small" sx={{ color: 'white' }} onClick={() => executePipelineUpToNode(id)} disabled={amIProcessing}>
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                </IconButton>
            </NodeHeader>

            <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2.5 }} className="nodrag" onMouseDown={(e) => e.stopPropagation()}>
                <Autocomplete options={modelOptions} value={data.modelName} onChange={handleModelChange} disableClearable size="small" renderInput={(params) => <TextField {...params} label="Select Model" />} />
                <FormControl fullWidth size="small">
                    <InputLabel>Search Strategy</InputLabel>
                    <Select label="Search Strategy" value={data.searchStrategy || ''} onChange={(e) => handleFieldChange('searchStrategy', e.target.value)}>
                        <MenuItem value="GridSearchCV">Grid Search CV</MenuItem>
                        <MenuItem value="RandomizedSearchCV">Randomized Search CV</MenuItem>
                    </Select>
                </FormControl>
                <TextField fullWidth type="number" size="small" label="Cross-Validation Folds" value={data.cvFolds || 5} onChange={(e) => handleFieldChange('cvFolds', Number(e.target.value))} InputProps={{ inputProps: { min: 2 } }} />

                {/* --- CHANGE 3: Updated multi-part metric selection UI --- */}
                <FormControl fullWidth size="small">
                    <InputLabel>Scoring Metric</InputLabel>
                    <Select label="Scoring Metric" value={data.scoringMetricBase || 'accuracy'} onChange={(e) => handleMetricBaseChange(e.target.value)}>
                        {BASE_SCORING_METRICS.map(metric => <MenuItem key={metric} value={metric}>{metric}</MenuItem>)}
                    </Select>
                </FormControl>

                {showAveragingDropdown && (
                    <FormControl fullWidth size="small">
                        <InputLabel>Averaging</InputLabel>
                        <Select label="Averaging" value={data.scoringMetricAvg || 'weighted'} onChange={(e) => handleAvgChange(e.target.value)}>
                            {AVERAGING_OPTIONS.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                        </Select>
                    </FormControl>
                )}

                {showClassInput && (
                    <TextField
                        fullWidth
                        size="small"
                        label="For Class Label"
                        value={data.scoringMetricClass || ''}
                        onChange={(e) => handleFieldChange('scoringMetricClass', e.target.value)}
                        helperText="Enter target class label (e.g., 1, 0, -1)"
                    />
                )}

                {currentModelConfig.length > 0 && (
                    <Stack spacing={2}>
                        <Box sx={{ display: 'flex', flexDirection:'column', pb: 2 }}>
                            <Typography sx={{ textAlign: 'center' }}>PARAMETER GRID</Typography>
                            <Typography fontSize={10} sx={{ textAlign: 'center', color: '#818181ff' }}>e.g., [10, 50, 100] or ['gini', 'entropy']</Typography>
                        </Box>
                        {currentModelConfig.map(paramDef => <HyperparameterGridInput key={paramDef.name} paramDef={paramDef} value={data.paramGrid?.[paramDef.name]} onChange={(e) => handleGridChange(paramDef.name, e.target.value)} />)}
                    </Stack>
                )}
            </Box>
            
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
};