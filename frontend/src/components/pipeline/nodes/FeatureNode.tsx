import React, { memo, useState, useEffect, useCallback, useRef } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Box, Select, MenuItem, FormControl, InputLabel, TextField, Autocomplete, Stack, CircularProgress, FormControlLabel, Checkbox } from '@mui/material';
import { usePipeline } from '../../../context/PipelineContext';
import type { IndicatorSchema, IndicatorParamDef } from '../types';
import { NodeHeader } from './NodeHeader'; // Import the new header

// --- A NEW, SMARTER Sub-component for rendering parameter inputs ---
const ParameterInput: React.FC<{
    paramDef: IndicatorParamDef;
    value: any; // The global value from the pipeline context
    // This now receives the raw event, not a pre-processed value
    onChange: (paramName: string, value: any, type: 'number' | 'string' | 'boolean') => void;
}> = ({ paramDef, value, onChange }) => {

    // 1. Local state to manage what's VISIBLE in the text field. It's ALWAYS a string.
    const [inputValue, setInputValue] = useState(String(value ?? ''));
    
    // 2. A ref to hold the debounce timer
    const debounceTimeout = useRef<NodeJS.Timeout | null>(null);

    // 3. Effect to sync local state if the GLOBAL state changes from outside
    //    (e.g., loading a workflow, changing indicator type)
    useEffect(() => {
        setInputValue(String(value ?? ''));
    }, [value]);
    
    // 4. The debounced function to update the global state
    const debouncedGlobalUpdate = useCallback((newVal: string) => {
        let finalValue: string | number | boolean = newVal;
        const { name, type } = paramDef;

        if (type === 'number') {
            const isTemplateVariable = newVal.startsWith('{{') && newVal.endsWith('}}');
            if (isTemplateVariable) {
                finalValue = newVal;
            } else {
                const numValue = parseFloat(newVal);
                // Only store a number if it's valid, otherwise keep the string
                // the user typed so they can see their input.
                finalValue = isNaN(numValue) ? newVal : numValue;
            }
        }
        // Call the parent's update function
        onChange(name, finalValue, type);
    }, [paramDef, onChange]);


    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newStringValue = e.target.value;
        // Update the local state IMMEDIATELY for a responsive UI
        setInputValue(newStringValue);

        // Clear any existing debounce timer
        if (debounceTimeout.current) {
            clearTimeout(debounceTimeout.current);
        }

        // Set a new timer to update the global state after 500ms of inactivity
        debounceTimeout.current = setTimeout(() => {
            debouncedGlobalUpdate(newStringValue);
        }, 500);
    };
    
    const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>, checked: boolean) => {
        const { name, type } = paramDef;
        setInputValue(String(checked)); // Update local state
        onChange(name, checked, type); // Update global state immediately for checkboxes
    };

    const handleSelectChange = (e: React.ChangeEvent<{ value: unknown }>) => {
        const { name, type } = paramDef;
        const val = e.target.value as string;
        setInputValue(val); // Update local state
        onChange(name, val, type); // Update global state immediately for selects
    };


    switch (paramDef.type) {
        case 'boolean':
            return (
                <FormControlLabel
                    control={<Checkbox checked={!!value} onChange={handleCheckboxChange} />}
                    label={paramDef.displayName || paramDef.name}
                    sx={{ textTransform: 'capitalize' }}
                />
            );
        case 'string':
            return (
                <FormControl fullWidth size="small">
                    <InputLabel>{paramDef.displayName || paramDef.name}</InputLabel>
                    <Select label={paramDef.displayName || paramDef.name} value={value || ''} onChange={handleSelectChange as any}>
                        {paramDef.options?.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </Select>
                </FormControl>
            );
        case 'number':
        default:
            return (
                <TextField
                    type="text"
                    size="small"
                    variant="outlined"
                    label={paramDef.displayName || paramDef.name}
                    // The value is now driven by our fast LOCAL state
                    value={inputValue}
                    onChange={handleInputChange}
                    sx={{ width: '265px' }}
                    InputProps={{
                        placeholder: "Enter number or {{variable}}",
                    }}
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

    // This function is now simplified, as the smart logic lives in ParameterInput.
    // It just updates the global state.
    const handleParamChange = (paramName: string, value: any, type: 'number' | 'string' | 'boolean') => {
        const updatedParams = { ...data.params, [paramName]: value };
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
                                // Pass the simplified handler down
                                onChange={handleParamChange}
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