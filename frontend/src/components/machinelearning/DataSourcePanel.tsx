import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Divider, Stack, FormControl, InputLabel, Select, MenuItem, TextField, Autocomplete, IconButton, Card, CardContent, Checkbox, FormControlLabel } from '@mui/material';
import { RemoveCircleOutline as RemoveIcon } from '@mui/icons-material';
import type { MLConfig, IndicatorConfig } from '../../pages/MachineLearning';

// --- Type Definitions (More robust to handle different param types) ---
interface IndicatorParamDef {
  name: string;
  type: 'number' | 'string' | 'boolean';
  defaultValue: number | string | boolean;
  options?: string[]; // Optional array of choices for 'string' type
}
interface IndicatorSchema {
  name: string;
  params: IndicatorParamDef[];
}

// --- Mock Data ---
const MOCK_SYMBOLS = ['ADAUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'];
const MOCK_TIMEFRAMES = ['15m', '1h', '4h', '1d'];

interface DataSourcePanelProps {
    config: MLConfig;
    onConfigChange: (path: string, value: any) => void;
}

const API_URL = 'http://127.0.0.1:8000';

// A new sub-component to dynamically render the correct input field
const ParameterInput: React.FC<{
    paramDef: IndicatorParamDef;
    value: any;
    onChange: (e: React.ChangeEvent<any>) => void;
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
                    <InputLabel sx={{ textTransform: 'capitalize' }}>{paramDef.name}</InputLabel>
                    <Select label={paramDef.name} value={value} onChange={onChange}>
                        {paramDef.options?.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </Select>
                </FormControl>
            );
        case 'number':
        default:
            return (
                <TextField
                    label={paramDef.name}
                    type="number"
                    size="small"
                    variant="outlined"
                    value={value}
                    onChange={onChange}
                    sx={{ textTransform: 'capitalize' }}
                />
            );
    }
};

export const DataSourcePanel: React.FC<DataSourcePanelProps> = ({ config, onConfigChange }) => {
    const [indicatorSchema, setIndicatorSchema] = useState<{ [key: string]: IndicatorSchema }>({});
    const [isLoadingSchema, setIsLoadingSchema] = useState(true);

    useEffect(() => {
        const fetchSchema = async () => {
            setIsLoadingSchema(true);
            try {
                const response = await fetch(`${API_URL}/api/ml/indicators`);
                if (!response.ok) throw new Error('Failed to fetch indicator schema');
                const data = await response.json();
                setIndicatorSchema(data);
            } catch (error) {
                console.error("Error fetching indicator schema:", error);
            } finally {
                setIsLoadingSchema(false);
            }
        };
        fetchSchema();
    }, []);

    const handleAddIndicator = (indicatorKey: string | null) => {
        if (!indicatorKey || !indicatorSchema[indicatorKey]) return;
        const definition = indicatorSchema[indicatorKey];
        const count = config.features.filter(f => f.name === indicatorKey).length;
        const newId = `${indicatorKey}_${count + 1}`;
        const newIndicator: IndicatorConfig = {
            id: newId,
            name: indicatorKey,
            params: definition.params.reduce((acc, param) => {
                acc[param.name] = param.defaultValue;
                return acc;
            }, {} as { [key: string]: any }),
        };
        onConfigChange('features', [...config.features, newIndicator]);
    };

    const handleRemoveIndicator = (idToRemove: string) => {
        const updatedFeatures = config.features.filter(f => f.id !== idToRemove);
        onConfigChange('features', updatedFeatures);
    };

    // Upgraded handler to manage different input types
    const handleParamChange = (indicatorId: string, paramName: string, value: any, type: 'number' | 'string' | 'boolean') => {
        const updatedFeatures = config.features.map(feature => {
            if (feature.id === indicatorId) {
                let finalValue = value;
                if (type === 'number') {
                    finalValue = Number(value) || 0;
                }
                // For strings and booleans, the value is already correct
                return {
                    ...feature,
                    params: { ...feature.params, [paramName]: finalValue },
                };
            }
            return feature;
        });
        onConfigChange('features', updatedFeatures);
    };

    return (
        <Paper elevation={0} sx={{ height: '100%', p: 2, overflowY: 'auto' }}>
            <Typography variant="h6" gutterBottom sx={{ pb: 2 }}>Data & Features</Typography>
            <Stack spacing={3} divider={<Divider />}>
                <Box>
                    <Stack spacing={2}>
                        {/* Data Source Pickers */}
                        <FormControl fullWidth size="small">
                            <InputLabel>Symbol</InputLabel>
                            <Select value={config.dataSource.symbol} label="Symbol" onChange={(e) => onConfigChange('dataSource.symbol', e.target.value)}>
                                {MOCK_SYMBOLS.map(s => <MenuItem key={s} value={s}>{s}</MenuItem>)}
                            </Select>
                        </FormControl>
                        <FormControl fullWidth size="small">
                            <InputLabel>Timeframe</InputLabel>
                            <Select value={config.dataSource.timeframe} label="Timeframe" onChange={(e) => onConfigChange('dataSource.timeframe', e.target.value)}>
                                {MOCK_TIMEFRAMES.map(t => <MenuItem key={t} value={t}>{t}</MenuItem>)}
                            </Select>
                        </FormControl>
                        <Autocomplete
                        size="small"
                        options={Object.keys(indicatorSchema)}
                        getOptionLabel={(optionKey) => indicatorSchema[optionKey]?.name || 'Loading...'}
                        onChange={(event, value) => handleAddIndicator(value)}
                        renderInput={(params) => <TextField {...params} label="Add Indicator..." />}
                        value={null}
                        disabled={isLoadingSchema}
                        />
                    </Stack>
                    <Stack spacing={2} sx={{ mt: 2 }}>
                        {config.features.map(indicator => {
                            // Find the full definition from our fetched schema
                            const indicatorDef = indicatorSchema[indicator.name];
                            if (!indicatorDef) return null; // Safety check in case schema hasn't loaded

                            return (
                                <Card key={indicator.id} variant="outlined">
                                    <CardContent sx={{ p: '16px !important' }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                            <Typography variant="body1" fontWeight="500">
                                                {/* *** THE FIX IS HERE *** */}
                                                {/* Use the fetched 'indicatorSchema' instead of the local static file */}
                                                {indicatorDef.name}
                                            </Typography>
                                            <IconButton size="small" onClick={() => handleRemoveIndicator(indicator.id)}>
                                                <RemoveIcon />
                                            </IconButton>
                                        </Box>
                                        <Stack direction="row" spacing={1} alignItems="center">
                                            {/* *** THE UPGRADE IS HERE *** */}
                                            {/* Dynamically render the correct input for each parameter */}
                                            {Object.keys(indicator.params).map(paramName => {
                                                const paramDef = indicatorDef.params.find(p => p.name === paramName);
                                                if (!paramDef) return null; // Safety check

                                                return (
                                                    <ParameterInput
                                                        key={paramName}
                                                        paramDef={paramDef}
                                                        value={indicator.params[paramName]}
                                                        onChange={(e) => {
                                                            const value = paramDef.type === 'boolean' ? e.target.checked : e.target.value;
                                                            handleParamChange(indicator.id, paramName, value, paramDef.type);
                                                        }}
                                                    />
                                                );
                                            })}
                                        </Stack>
                                    </CardContent>
                                </Card>
                            );
                        })}
                    </Stack>
                </Box>
            </Stack>
        </Paper>
    );
};