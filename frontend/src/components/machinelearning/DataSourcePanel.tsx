import React, { useState, useEffect } from 'react';
import { 
    Box, Typography, Button, CircularProgress, Paper, FormControl, InputLabel, Select, MenuItem, 
    FormGroup, FormControlLabel, Checkbox, Stack, TextField, IconButton, Autocomplete, Card, CardContent 
} from '@mui/material';
import { RemoveCircleOutline as RemoveIcon } from '@mui/icons-material';
import type { MLConfig, IndicatorConfig } from '../machinelearning/types';
import { DataGridDisplay, DataInfoDisplay } from '../machinelearning/shared/DataDisplays';

const API_URL = 'http://127.0.0.1:8000';

// --- ROBUST TYPE DEFINITIONS (from your provided code) ---
interface IndicatorParamDef {
  name: string;          // e.g., "length"
  displayName: string;   // e.g., "Length"
  type: 'number' | 'string' | 'boolean';
  defaultValue: number | string | boolean;
  options?: string[]; // Optional for 'string' type dropdowns
}
interface IndicatorDefinition {
  name: string; // The full name, e.g., "Simple Moving Average"
  params: IndicatorParamDef[];
}
type IndicatorSchema = { [key: string]: IndicatorDefinition };


// --- DYNAMIC INPUT SUB-COMPONENT (from your provided code) ---
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
                    label={paramDef.displayName}
                    sx={{ textTransform: 'capitalize' }}
                />
            );
        case 'string':
            return (
                <FormControl fullWidth size="small">
                    <InputLabel sx={{ textTransform: 'capitalize' }}>{paramDef.displayName}</InputLabel>
                    <Select label={paramDef.displayName} value={value} onChange={onChange}>
                        {paramDef.options?.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </Select>
                </FormControl>
            );
        case 'number':
        default:
            return (
                 <Box>
                    <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 0.5 }}>
                        {paramDef.displayName}
                    </Typography>
                    <TextField
                        type="number"
                        size="small"
                        variant="outlined"
                        value={value}
                        onChange={onChange}
                        sx={{ width: '80px' }}
                    />
                </Box>
            );
    }
};


interface FeatureEngineeringTabProps {
    config: MLConfig;
    onChange: (path: string, value: any) => void;
    onCalculate: () => void;
    featuresData: any[];
    featuresInfo: any;
    isCalculating: boolean;
}

export const FeatureEngineeringTab: React.FC<FeatureEngineeringTabProps> = ({ config, onChange, onCalculate, featuresData, featuresInfo, isCalculating }) => {
    const [indicatorSchema, setIndicatorSchema] = useState<IndicatorSchema>({});
    const [isLoadingSchema, setIsLoadingSchema] = useState(true);
    const [fetchError, setFetchError] = useState<string | null>(null);

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

    const handleAddIndicator = (indicatorKey: string | null) => {
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
        <Box sx={{ p: 1, height: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'row', justifyContent: 'center', gap: 1 }}>
                
                {/* Left Panel: Controls */}
                <Box sx={{ flexGrow: 1, maxWidth: 400, display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Paper variant='outlined' sx={{ p: 2, display: 'flex', flexDirection: 'column', flexGrow: 1, overflow: 'hidden' }}>
                        <Typography variant="subtitle1" gutterBottom>Technical Indicators</Typography>

                        <Autocomplete
                            size="small"
                            sx={{ mb: 2 }}
                            options={Object.keys(indicatorSchema)}
                            getOptionLabel={(optionKey) => indicatorSchema[optionKey]?.name || 'Loading...'}
                            onChange={(event, value) => handleAddIndicator(value)}
                            renderInput={(params) => <TextField {...params} label="Add Indicator..." />}
                            value={null} // Ensures the component is always ready for a new selection
                            disabled={isLoadingSchema}
                        />
                        
                        {fetchError && <Typography color="error" variant="body2" sx={{ mb: 2 }}>{fetchError}</Typography>}

                        <Box sx={{ overflowY: 'auto', flexGrow: 1, pr: 1 }}>
                            <Stack spacing={2}>
                                {config.features.map(indicator => {
                                    const indicatorDef = indicatorSchema[indicator.name];
                                    if (!indicatorDef) return null;

                                    return (
                                        <Card key={indicator.id} variant="outlined">
                                            <CardContent sx={{ p: '16px !important' }}>
                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
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
                                            </CardContent>
                                        </Card>
                                    );
                                })}
                            </Stack>
                        </Box>
                    </Paper>

                    {/* Preprocessing Panel */}
                    <Paper variant='outlined' sx={{ p: 2 }}>
                        <Typography variant="subtitle1" gutterBottom>Preprocessing & Scaling</Typography>
                        <FormControl fullWidth size="small" margin="normal">
                            <InputLabel>Scaler</InputLabel>
                            <Select value={config.preprocessing.scaler} label="Scaler" onChange={(e) => onChange('preprocessing.scaler', e.target.value)}>
                                <MenuItem value="none">None</MenuItem>
                                <MenuItem value="StandardScaler">Standard Scaler</MenuItem>
                                <MenuItem value="MinMaxScaler">MinMax Scaler</MenuItem>
                            </Select>
                        </FormControl>
                        <FormGroup>
                            <FormControlLabel control={<Checkbox checked={config.preprocessing.removeCorrelated} onChange={(e) => onChange('preprocessing.removeCorrelated', e.target.checked)} />} label="Remove Correlated Features" />
                            <FormControlLabel control={<Checkbox checked={config.preprocessing.usePCA} onChange={(e) => onChange('preprocessing.usePCA', e.target.checked)} />} label="Use PCA" />
                        </FormGroup>
                    </Paper>
                    
                    <Box>
                        <Button variant="contained" onClick={onCalculate} disabled={isCalculating} fullWidth>
                            {isCalculating ? <CircularProgress size={24} /> : 'Calculate Features'}
                        </Button>
                    </Box>
                </Box>
                
                {/* Right Panel: Data Display */}
                <Box sx={{ flexGrow: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Box sx={{ minHeight: '200px' }}>
                        <DataInfoDisplay info={featuresInfo} />
                    </Box>
                    <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                        <DataGridDisplay data={featuresData} title="Features Data" />
                    </Box>
                </Box>
            </Box>
        </Box>
    );
};