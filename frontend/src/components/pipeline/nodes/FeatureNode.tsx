import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box, Select, MenuItem, FormControl, InputLabel, TextField, Autocomplete, Stack, CircularProgress, FormControlLabel, Checkbox } from '@mui/material';
import { usePipeline } from '../../../context/PipelineContext';
import type { IndicatorSchema, IndicatorParamDef } from '../types';
import { NodeHeader } from './NodeHeader'; // Import the new header

// --- Sub-component for rendering parameter inputs ---
const ParameterInput: React.FC<{
    paramDef: IndicatorParamDef;
    value: any;
    onChange: (e: React.ChangeEvent<any>, checked?: boolean) => void;
}> = ({ paramDef, value, onChange }) => {
    switch (paramDef.type) {
        case 'boolean':
            return (
                <FormControlLabel
                    control={<Checkbox checked={!!value} onChange={onChange} />}
                    label={paramDef.displayName || paramDef.name}
                    sx={{ textTransform: 'capitalize' }}
                />
            );
        case 'string':
            return (
                <FormControl fullWidth size="small">
                    <InputLabel>{paramDef.displayName || paramDef.name}</InputLabel>
                    <Select label={paramDef.displayName || paramDef.name} value={value || ''} onChange={onChange}>
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
                    label={paramDef.displayName || paramDef.name}
                    value={value}
                    onChange={onChange}
                    sx={{ width: '265px' }}
                />
            );
    }
};

interface FeatureNodeData {
    label: string;
    name: string;
    timeframe: string;
    params: { [key: string]: any };
}

interface FeatureNodeProps extends NodeProps<FeatureNodeData> {
    indicatorSchema: IndicatorSchema;
    isLoadingSchema: boolean;
}

export const FeatureNode = memo(({ id, data, indicatorSchema, isLoadingSchema }: FeatureNodeProps) => {

    const { updateNodeData } = usePipeline();
    const indicatorDef = indicatorSchema[data.name];

    const handleIndicatorChange = (newIndicatorKey: string | null) => {
        if (!newIndicatorKey || !indicatorSchema[newIndicatorKey]) return;

        const definition = indicatorSchema[newIndicatorKey];

        // When changing the indicator, reset its parameters to their defaults
        const defaultParams = definition.params.reduce((acc, param) => {
            acc[param.name] = param.defaultValue;
            return acc;
        }, {} as { [key: string]: any });

        updateNodeData(id, {
            name: newIndicatorKey,
            params: defaultParams,
        });
    };

    const handleParamChange = (paramName: string, value: any, type: 'number' | 'string' | 'boolean') => {
        let finalValue = value;
        if (type === 'number') {
            finalValue = Number(value) || 0; // Ensure it's a number
        }
        
        const updatedParams = { ...data.params, [paramName]: finalValue };
        updateNodeData(id, { params: updatedParams });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main', // Match the edge color
        border: '1px solid #555',
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' }}>
            {/* Input Handle (to receive data) */}
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555'}} />

            <NodeHeader nodeId={id} title={data.label} color="#4641daff">
            </NodeHeader>

            <Box
                sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <Autocomplete
                    size="small"
                    options={Object.keys(indicatorSchema)}
                    getOptionLabel={(optionKey) => indicatorSchema[optionKey]?.name || 'Loading...'}
                    value={data.name || null}
                    onChange={(event, value) => handleIndicatorChange(value)}
                    isOptionEqualToValue={(option, value) => option === value}
                    disabled={isLoadingSchema}
                    slotProps={{
                        paper: {
                            sx: {
                                width: '280px',
                            },
                        },
                    }}
                    renderInput={(params) => (
                        <TextField 
                            {...params} 
                            label="Indicator"
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

                {/* Dynamically render parameters if the indicator definition is loaded */}
                {indicatorDef && (
                    <Stack spacing={1.5} alignItems="flex-start" flexWrap="wrap">
                        {indicatorDef.params.map(paramDef => (
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
            
            {/* Output Handle (to pass data along) */}
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
});