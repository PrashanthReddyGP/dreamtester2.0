import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box, Select, MenuItem, FormControl, InputLabel, TextField, IconButton, Autocomplete, Stack, CircularProgress, FormControlLabel, Checkbox } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';

// --- Type Definitions (copied from DataTab/PipelineContext) ---
interface IndicatorParamDef {
    name: string;
    displayName: string;
    type: 'number' | 'string' | 'boolean';
    defaultValue: number | string | boolean;
    options?: string[];
}
interface IndicatorDefinition {
    name: string;
    params: IndicatorParamDef[];
}
type IndicatorSchema = { [key: string]: IndicatorDefinition };

// --- Sub-component for rendering parameter inputs (copied from DataTab) ---
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
                    label={paramDef.name}
                    sx={{ textTransform: 'capitalize' }}
                />
            );
        case 'string':
            return (
                <FormControl fullWidth size="small">
                    <Select label={paramDef.name} value={value} onChange={onChange}>
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
                    label={paramDef.name}
                    value={value}
                    onChange={onChange}
                    sx={{ width: '265px' }}
                />
            );
    }
};

interface FeatureNodeData {
    label: string;
    indicator: string; // Renamed from 'feature' for clarity
    params: { [key: string]: any };
}

interface FeatureNodeProps extends NodeProps<FeatureNodeData> {
    indicatorSchema: IndicatorSchema;
    isLoadingSchema: boolean;
}

export const FeatureNode = memo(({ id, data, indicatorSchema, isLoadingSchema }: FeatureNodeProps) => {

    const { updateNodeData } = usePipeline();
    const indicatorDef = indicatorSchema[data.indicator];

    const handleIndicatorChange = (newIndicatorKey: string | null) => {
        if (!newIndicatorKey || !indicatorSchema[newIndicatorKey]) return;

        const definition = indicatorSchema[newIndicatorKey];
        // When changing the indicator, reset its parameters to their defaults
        const defaultParams = definition.params.reduce((acc, param) => {
            acc[param.name] = param.defaultValue;
            return acc;
        }, {} as { [key: string]: any });

        updateNodeData(id, {
            indicator: newIndicatorKey,
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
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' /* Light blue border */ }}>
            {/* Input Handle (to receive data) */}
            <Handle type="target" position={Position.Left} style={handleStyle} />

            <Box sx={{ bgcolor: '#4641daff', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignContent: 'center', height: '50px'  }}>
                <Typography variant="subtitle2" sx={{ color: 'primary.contrastText', pl: 1, alignSelf: 'center' }}>
                    {data.label}
                </Typography>
            </Box>

            <Box
                sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <Autocomplete
                    size="small"
                    options={Object.keys(indicatorSchema)}
                    getOptionLabel={(optionKey) => indicatorSchema[optionKey]?.name || 'Loading...'}
                    value={data.indicator || null}
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
            <Handle type="source" position={Position.Right} style={handleStyle} />
        </Paper>
    );
});