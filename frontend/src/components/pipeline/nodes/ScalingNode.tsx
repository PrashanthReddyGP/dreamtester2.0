import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box, Select, MenuItem, FormControlLabel, Checkbox, TextField, IconButton, Autocomplete, Stack, CircularProgress, FormControl } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext'; // Assuming this path is correct

// --- Type Definitions (similar to Indicator schema) ---
// It's recommended to move these to a shared types file if used across multiple nodes.
interface ScalerParamDef {
    name: string;
    displayName: string;
    type: 'number' | 'string' | 'boolean';
    defaultValue: number | string | boolean;
    options?: string[]; // For string type
}

interface ScalerDefinition {
    name: string; // User-friendly name, e.g., "Standard Scaler"
    params: ScalerParamDef[];
}

export type ScalerSchema = { [key: string]: ScalerDefinition }; // e.g., { 'standard_scaler': { ... } }

// --- Sub-component for rendering parameter inputs (copied from FeatureNode) ---
// This component is generic and can be reused. Consider moving it to a shared components folder.
const ParameterInput: React.FC<{
    paramDef: ScalerParamDef;
    value: any;
    onChange: (e: React.ChangeEvent<any>, checked?: boolean) => void;
}> = ({ paramDef, value, onChange }) => {
    switch (paramDef.type) {
        case 'boolean':
            return (
                <FormControlLabel
                    control={<Checkbox checked={!!value} onChange={onChange} />}
                    label={paramDef.displayName}
                    sx={{ textTransform: 'capitalize' }}
                />
            );
        case 'string':
            return (
                <FormControl fullWidth size="small">
                    <Select label={paramDef.displayName} value={value} onChange={onChange}>
                        {paramDef.options?.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </Select>
                </FormControl>
            );
        case 'number':
        default:
            return (
                <TextField
                    type="number"
                    size="small"
                    variant="outlined"
                    label={paramDef.displayName}
                    value={value}
                    onChange={onChange}
                    sx={{ width: '265px' }}
                />
            );
    }
};

// --- Node Data Structure ---
interface ScalerNodeData {
    label: string;
    scaler: string; // The key for the selected scaler, e.g., 'standard_scaler'
    params: { [key: string]: any };
}

// --- Node Component Props ---
interface ScalerNodeProps extends NodeProps<ScalerNodeData> {
    scalerSchema: ScalerSchema;
    isLoadingSchema: boolean;
}

export const ScalerNode = memo(({ id, data, scalerSchema, isLoadingSchema }: ScalerNodeProps) => {

    const { updateNodeData } = usePipeline();
    const scalerDef = scalerSchema[data.scaler];

    const handleScalerChange = (newScalerKey: string | null) => {
        if (!newScalerKey || !scalerSchema[newScalerKey]) return;

        const definition = scalerSchema[newScalerKey];
        // When changing the scaler, reset its parameters to their defaults
        const defaultParams = definition.params.reduce((acc, param) => {
            acc[param.name] = param.defaultValue;
            return acc;
        }, {} as { [key: string]: any });

        updateNodeData(id, {
            scaler: newScalerKey,
            params: defaultParams,
        });
    };

    const handleParamChange = (paramName: string, value: any, type: 'number' | 'string' | 'boolean') => {
        let finalValue = value;
        if (type === 'number') {
            finalValue = Number(value) || 0; // Ensure it's a number, default to 0
        }
        
        const updatedParams = { ...data.params, [paramName]: finalValue };
        updateNodeData(id, { params: updatedParams });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #4caf50' /* Green border for distinction */ }}>
            {/* Input Handle (to receive features) */}
            <Handle type="target" position={Position.Left} style={handleStyle} />

            <Box sx={{ bgcolor: '#4caf50', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignItems: 'center'  }}>
                <Typography variant="subtitle2" sx={{ color: 'primary.contrastText', pl: 1, alignSelf: 'center' }}>
                    {data.label}
                </Typography>
                <IconButton size="small" sx={{ color: 'white' }} aria-label="run">
                    <PlayArrowIcon />
                </IconButton>
            </Box>

            <Box
                sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <Autocomplete
                    size="small"
                    options={Object.keys(scalerSchema)}
                    getOptionLabel={(optionKey) => scalerSchema[optionKey]?.name || 'Loading...'}
                    value={data.scaler || null}
                    onChange={(event, value) => handleScalerChange(value)}
                    isOptionEqualToValue={(option, value) => option === value}
                    disabled={isLoadingSchema}
                    slotProps={{
                        paper: {
                            sx: { width: '280px' },
                        },
                    }}
                    renderInput={(params) => (
                        <TextField 
                            {...params} 
                            label="Scaler"
                            InputProps={{
                                ...params.InputProps,
                                endAdornment: (
                                    <>
                                        {isLoadingSchema ? <CircularProgress color="inherit" size={20} /> : null}
                                        {params.InputProps.endAdornment}
                                    </>
                                ),
                            }}
                        />
                    )}
                />

                {/* Dynamically render parameters if the scaler definition is loaded */}
                {scalerDef && (
                    <Stack spacing={1.5} alignItems="flex-start" flexWrap="wrap">
                        {scalerDef.params.map(paramDef => (
                            <ParameterInput
                                key={paramDef.name}
                                paramDef={paramDef}
                                value={data.params[paramDef.name]}
                                onChange={(e, checked) => {
                                    const value = paramDef.type === 'boolean' ? checked : e.target.value;
                                    handleParamChange(paramDef.name, value, paramDef.type);
                                }}
                            />
                        ))}
                    </Stack>
                )}
            </Box>
            
            {/* Output Handle (to pass scaled features along) */}
            <Handle type="source" position={Position.Right} style={handleStyle} />
        </Paper>
    );
});