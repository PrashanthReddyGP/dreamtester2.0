// src/components/machinelearning/tabs/DataTab.tsx
import React, { useState, useEffect } from 'react';
import { Box, Paper, Autocomplete, TextField, Button, CircularProgress, Alert, MenuItem, Typography, IconButton, Stack,
        FormControl, FormControlLabel, Checkbox, InputLabel, Select
 } from '@mui/material';
import { RemoveCircleOutline as RemoveIcon } from '@mui/icons-material';
import type { MLConfig, IndicatorConfig } from '../types';
import { DataGridDisplay, DataInfoDisplay } from '../shared/DataDisplays';
import { fetchAvailableSymbols } from '../../../services/api';


const API_URL = 'http://127.0.0.1:8000';


interface DataTabProps {
    config: MLConfig;
    onChange: (path: string, value: any) => void;
    onFetch: () => void;
    onCalculate: () => void;
    displayData: any[];
    displayInfo: any;
    isFetching: boolean;
    isCalculating: boolean;
}


// --- Type Definitions ---
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


// --- Sub-component for rendering parameter inputs (Corrected) ---
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
                    <InputLabel sx={{ textTransform: 'capitalize' }}>{paramDef.displayName}</InputLabel>
                    <Select label={paramDef.name} value={value} onChange={onChange}>
                        {paramDef.options?.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </Select>
                </FormControl>
            );
        case 'number':
        default:
            return (
                <Box>
                    <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 2 }}>
                        {paramDef.displayName}
                    </Typography>
                    <TextField
                        type="number"
                        size="small"
                        variant="outlined"
                        label={paramDef.name}
                        value={value}
                        onChange={onChange}
                        sx={{ width: '90px' }}
                    />
                </Box>
            );
    }
};


// Define the available timeframes in a constant
const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];


export const DataTab: React.FC<DataTabProps> = ({ config, onChange, onFetch, onCalculate, displayData, displayInfo, isFetching, isCalculating }) => {
    // State for fetching the list of all available symbols
    const [symbolList, setSymbolList] = useState<string[]>([]);
    const [isFetchingSymbols, setIsFetchingSymbols] = useState(false);
    const [fetchError, setFetchError] = useState<string | null>(null);

    // Fetch the available symbols once when the component mounts
    useEffect(() => {
        const loadSymbols = async () => {
            setIsFetchingSymbols(true);
            setFetchError(null);
            try {
                const fetchedSymbols = await fetchAvailableSymbols('binance');
                setSymbolList(fetchedSymbols);

                // If no symbol is currently set in the config, default to BTCUSDT
                if (!config.dataSource.symbol && fetchedSymbols.includes('BTCUSDT')) {
                    onChange('dataSource.symbol', 'BTCUSDT');
                }
            } catch (err: any) {
                console.error("Failed to fetch symbols:", err);
                setFetchError(err.message || 'An unknown error occurred while fetching symbols.');
            } finally {
                setIsFetchingSymbols(false);
            }
        };

        loadSymbols();
    }, []); // Empty dependency array ensures this runs only once on mount

    const [indicatorSchema, setIndicatorSchema] = useState<IndicatorSchema>({});
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
                setFetchError("Could not load indicator schema.");
            } finally {
                setIsLoadingSchema(false);
            }
        };
        fetchSchema();
    }, []);

    const handleAddIndicator = (indicatorKey: string) => {
        if (!indicatorKey || !indicatorSchema[indicatorKey]) return;

        const definition = indicatorSchema[indicatorKey];
        const newIndicator: IndicatorConfig = {
            id: `${indicatorKey}_${Date.now()}`,
            name: indicatorKey,
            params: definition.params.reduce((acc, param) => {
                acc[param.name] = param.defaultValue;
                return acc;
            }, {} as { [key: string]: any }),
        };

        onChange('features', [...config.features, newIndicator]);
    };

    const handleRemoveIndicator = (idToRemove: string) => {
        const updatedFeatures = config.features.filter(f => f.id !== idToRemove);
        onChange('features', updatedFeatures);
    };

    const handleParamChange = (indicatorId: string, paramName: string, value: any, type: 'number' | 'string' | 'boolean') => {
        const updatedFeatures = config.features.map(feature => {
            if (feature.id === indicatorId) {
                let finalValue = value;
                if (type === 'number') {
                    finalValue = Number(value) || 0;
                }
                return { ...feature, params: { ...feature.params, [paramName]: finalValue } };
            }
            return feature;
        });
        onChange('features', updatedFeatures);
    };

    return (
        <Box sx={{ p: 1, display: 'flex', flexDirection: 'column', gap: 1, height: '100%' }}>

            {/* <Box sx={{ flexGrow: 0 }}>
                <Typography variant="h5" gutterBottom sx={{ textAlign: 'center' }}>Data Tab</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                    Select the financial instrument, timeframe, and date range for your analysis.
                </Typography>
            </Box>
            */}
            
            <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'row', gap: 1, overflow: 'hidden', pb: 3 }}>
                <Box sx={{ flexGrow: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, width: '400px', minWidth: '400px', height: '100%' }}>
                    <Paper variant='outlined' sx={{ flexGrow: 0, height: '300px', width: '100%' }}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, p: 2, height: '100%', width: '100%' }}>
                            {fetchError && <Alert severity="error" sx={{ width: '100%'}}>{fetchError}</Alert>}
                            <Autocomplete
                                id="symbol-autocomplete"
                                options={symbolList}
                                value={config.dataSource.symbol || null} // Controlled component: value must be null if not found
                                onChange={(event, newValue) => {
                                    onChange('dataSource.symbol', newValue || ''); // Send empty string if cleared
                                }}
                                loading={isFetchingSymbols}
                                sx={{ width: '100%' }}
                                renderInput={(params) => (
                                    <TextField
                                        {...params}
                                        label="Symbol"
                                        size="small"
                                        InputProps={{
                                            ...params.InputProps,
                                            endAdornment: (
                                                <>
                                                    {isFetchingSymbols ? <CircularProgress color="inherit" size={20} /> : null}
                                                    {params.InputProps.endAdornment}
                                                </>
                                            ),
                                        }}
                                    />
                                )}
                            />
                            <TextField
                                select
                                label="Timeframe"
                                size="small"
                                value={config.dataSource.timeframe}
                                onChange={(e) => onChange('dataSource.timeframe', e.target.value)}
                                sx={{ width: '100%' }}
                            >
                                {timeframes.map((option) => (
                                    <MenuItem key={option} value={option}>
                                        {option}
                                    </MenuItem>
                                ))}
                            </TextField>
                            <TextField label="Start Date" type="date" size="small" value={config.dataSource.startDate} onChange={(e) => onChange('dataSource.startDate', e.target.value)} InputLabelProps={{ shrink: true }}  sx={{ width: '100%' }}/>
                            <TextField label="End Date" type="date" size="small" value={config.dataSource.endDate} onChange={(e) => onChange('dataSource.endDate', e.target.value)} InputLabelProps={{ shrink: true }}  sx={{ width: '100%' }}/>
                            <Button variant="contained" onClick={onFetch} disabled={isFetching} sx={{ width: '100%' }}>
                                {isFetching ? <CircularProgress size={24} /> : 'Fetch Data'}
                            </Button>
                        </Box>
                    </Paper>
                    <Box sx={{ flexGrow: 1, height: '100%', width: '100%', overflow: 'auto' }}>
                        {/* Technical Indicators Panel */}
                        <Paper variant="outlined" sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column', flexGrow: 1, overflow: 'auto' }}>
                            <Typography variant="subtitle1" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', pb: 1 }}>Technical Indicators</Typography>
                            <Autocomplete
                                size="small"
                                sx={{ mb: 2 }}
                                options={Object.keys(indicatorSchema)}
                                getOptionLabel={(optionKey) => indicatorSchema[optionKey]?.name || 'Loading...'}
                                onChange={(event, value) => handleAddIndicator(value)}
                                renderInput={(params) => <TextField {...params} label="Add Indicator..." />}
                                value={null}
                                disabled={isLoadingSchema}
                                isOptionEqualToValue={(option, value) => option === value}
                            />
                            
                            {fetchError && <Typography color="error" variant="body2" sx={{ mb: 2 }}>{fetchError}</Typography>}

                            <Box sx={{ overflowY: 'auto', flexGrow: 1 }}>
                                <Stack spacing={1}>
                                    {config.features.map(indicator => {
                                        const indicatorDef = indicatorSchema[indicator.name];
                                        if (!indicatorDef) return null;

                                        return (
                                            <Paper key={indicator.id} variant="outlined" sx={{ p: 1.5 }}>
                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                    <Typography variant="body1" fontWeight="500">{indicatorDef.name}</Typography>
                                                    <IconButton size="small" onClick={() => handleRemoveIndicator(indicator.id)}>
                                                        <RemoveIcon />
                                                    </IconButton>
                                                </Box>
                                                <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap">
                                                    {indicatorDef.params.map(paramDef => (
                                                        <ParameterInput
                                                            key={paramDef.name}
                                                            paramDef={paramDef}
                                                            value={indicator.params[paramDef.name]}
                                                            onChange={(e, checked) => {
                                                                const value = paramDef.type === 'boolean' ? checked : e.target.value;
                                                                handleParamChange(indicator.id, paramDef.name, value, paramDef.type);
                                                            }}
                                                        />
                                                    ))}
                                                </Stack>
                                            </Paper>
                                        );
                                    })}
                                </Stack>
                            </Box>
                            <Box sx={{ flexShrink: 0 }}>
                                <Button variant="contained" onClick={onCalculate} disabled={isCalculating} fullWidth>
                                    {isCalculating ? <CircularProgress size={24} /> : 'Calculate Features'}
                                </Button>
                            </Box>
                        </Paper>
                        {/* <DataInfoDisplay info={info} /> */}
                    </Box>
                </Box>

                <Box sx={{ flexGrow: 1, overflowY: 'auto' }}>
                    <DataGridDisplay
                        key={displayInfo?.["Data Points"]} 
                        data={displayData}
                        info={displayInfo} 
                        title="Features Data"
                    />
                </Box>
            </Box>
            
        </Box>
    );
};