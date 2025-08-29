import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  TextField,
  Typography,
  Checkbox,
  FormControlLabel,
  CircularProgress,
  Alert,
  IconButton,
  Divider,
  Autocomplete,
  Chip,
  ToggleButtonGroup,
  ToggleButton,
  Select,
  MenuItem
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { fetchAvailableSymbols } from '../../services/api';
import { v4 as uuidv4 } from 'uuid'; // For unique rule IDs
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';

// This should be your actual backend URL
const API_URL = 'http://127.0.0.1:8000'; 

// --- Type Definitions for Indicator-based Optimization ---

// export interface OptimizationConfig {
//   strategy_code: string;
//   parameters_to_optimize: OptimizableParameter[]; // Send the whole object
// }

export interface TestSubmissionConfig {
  mode: 'parameters' | 'assets';
  strategy_code: string;
  parameters_to_optimize?: OptimizableParameter[]; // Optional for asset screening
  symbols_to_screen?: string[]; // Optional for parameter optimization
}

interface OptimizeModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (config: SuperOptimizationConfig) => void; 
  strategyCode: string | null;
  isSubmitting: boolean;
}

interface OptimizableParameter {
    id: string;
    type: 'strategy_param' | 'indicator_param';
    name: string;
    value: number;
    enabled: boolean;
    mode: 'range' | 'list';
    start: number;
    end: number;
    step: number;
    list_values: string;
    indicatorIndex?: number;
    paramIndex?: number;
}

export interface CombinationRule {
  id: string; // For React keys
  param1: string; // The ID of the first parameter
  operator: '<' | '>' | '<=' | '>=' | '===' | '!==';
  param2: string; // The ID of the second parameter
}

export interface SuperOptimizationConfig {
  strategy_code: string;
  parameters_to_optimize: OptimizableParameter[]; // Could be empty
  symbols_to_screen: string[]; // Could be empty
  combination_rules: Omit<CombinationRule, 'id'>[]; // Send the rules to the backend
}

const fetchAllParametersAndSettings = async (code: string): Promise<any> => { // Using `any` for simplicity, can be typed
    const response = await fetch(`${API_URL}/api/strategies/parse-parameters`, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: code,
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "An unknown error" }));
        throw new Error(errorData.detail);
    }
    return response.json();
};

// --- The Main Modal Component ---
export const OptimizeModal: React.FC<OptimizeModalProps> = ({ open, onClose, onSubmit, strategyCode, isSubmitting }) => {
  const [params, setParams] = useState<OptimizableParameter[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rules, setRules] = useState<CombinationRule[]>([]);

  const [mode, setMode] = useState<'parameters' | 'assets'>('parameters');
  const [symbolList, setSymbolList] = useState<string[]>([]);
  const [isFetchingSymbols, setIsFetchingSymbols] = useState(false);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [defaultSymbol, setDefaultSymbol] = useState<string | null>(null);

  const hasFetchedData = React.useRef(false);

  useEffect(() => {
    // If the modal is not open, reset the fetch tracker.
    if (!open) {
      hasFetchedData.current = false;
      return;
    }

    // Only fetch data if the modal is open AND we haven't fetched it yet.
    if (open && strategyCode && !hasFetchedData.current) {
      setIsLoading(true);
      setError(null);

      // Fetch both symbols and parameters at the same time.
      Promise.all([
        fetchAvailableSymbols('binance'),
        fetchAllParametersAndSettings(strategyCode)
      ]).then(([fetchedSymbols, parsedData]) => {
          // --- Handle Symbols ---
          setSymbolList(fetchedSymbols);
          const symbolFromFile = parsedData.settings?.symbol;
          if (symbolFromFile) {
              setDefaultSymbol(symbolFromFile);
              setSelectedSymbols([symbolFromFile]);
          } else {
              setSelectedSymbols([]);
          }

          // --- Handle Parameters ---
          const optimizableParams = parsedData.optimizable_params.map((p: any): OptimizableParameter => ({
              ...p,
              enabled: false,
              mode: 'range', // Default to 'range' mode
              start: p.value,
              end: p.value,
              step: 1,
              list_values: String(p.value), // Default list is the single default value
          }));
          setParams(optimizableParams);

          // Mark that we have successfully fetched the data.
          hasFetchedData.current = true;

      }).catch(err => {
          setError(err.message);
      }).finally(() => {
          setIsLoading(false);
      });
    }
  }, [open, strategyCode]); // This effect now correctly depends only on `open` and `strategyCode`.

  const handleParamChange = (id: string, field: keyof OptimizableParameter, value: any) => {
    setParams(prev => prev.map(p => p.id === id ? { ...p, [field]: value } : p));
  };

  const handleModeChange = (event: React.MouseEvent<HTMLElement>, newMode: 'parameters' | 'assets' | null) => {
    if (newMode !== null) {
        setMode(newMode);
    }
  };

  const handleSubmit = () => {
    if (!strategyCode) {
      setError("Strategy code is missing."); return;
    }
    const enabledParams = params.filter(p => p.enabled);
    const symbolsToTest = selectedSymbols;
    if (symbolsToTest.length === 0) {
        setError("Please select at least one asset to test."); return;
    }
    const config: SuperOptimizationConfig = {
      strategy_code: strategyCode,
      parameters_to_optimize: enabledParams,
      symbols_to_screen: symbolsToTest,
      combination_rules: rules.map(({ id, ...rest }) => rest),
    };
    onSubmit(config);
  };

  const addRule = () => {
    // Add a new, empty rule to the state
    setRules([...rules, { id: uuidv4(), param1: '', operator: '<', param2: '' }]);
  };
  const updateRule = (id: string, field: keyof Omit<CombinationRule, 'id'>, value: string) => {
    setRules(rules.map(rule => (rule.id === id ? { ...rule, [field]: value } : rule)));
  };

  const removeRule = (id: string) => {
    setRules(rules.filter(rule => rule.id !== id));
  };

  // Get a list of enabled parameters for the rule dropdowns
  const enabledParamsForRules = params.filter(p => p.enabled);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        Configure Optimization
        <IconButton aria-label="close" onClick={onClose}><CloseIcon/></IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {isLoading && (<Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}><CircularProgress /><Typography sx={{ml:2}}>Fetching Initial Data...</Typography></Box>)}
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        
        {!isLoading && !error && (
            <Box sx={{ display: 'flex', gap: 3 }}>

                {/* --- LEFT PANEL: ASSET SELECTION --- */}
                <Box sx={{ flex: 1, minWidth: '300px', gap: 3, display: 'flex', flexDirection: 'column' }}>

                    <Box sx={{flexGrow: 1}}>
                      <Typography variant="h6" gutterBottom>Select Assets</Typography>
                      <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                          Choose one or more assets to run the test against.
                      </Typography>
                      <Autocomplete
                          multiple id="asset-screener-autocomplete" options={symbolList} value={selectedSymbols} loading={isFetchingSymbols}
                          onChange={(event, newValue) => { setSelectedSymbols(newValue); }}
                          freeSolo
                          renderTags={(value, getTagProps) => value.map((option, index) => {
                              const { key, ...tagProps } = getTagProps({ index });
                              return <Chip key={key} variant="outlined" label={option} {...tagProps} />;
                          })}
                          renderInput={(params) => (<TextField {...params} variant="outlined" label="Symbols to Test" placeholder="Add symbols..."/>)}
                      />
                    </Box>

                    <Divider orientation="horizontal" flexItem />

                    <Box sx={{flexGrow: 1}}>
                      <Typography variant="h6" gutterBottom>Define Combination Rules</Typography>
                      <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                        Prevent illogical tests. For example, ensure a "Fast" SMA period is always less than a "Slow" SMA period.
                      </Typography>
                      
                      {rules.map((rule, index) => (
                          <CombinationRuleRow 
                              key={rule.id} 
                              rule={rule} 
                              availableParams={enabledParamsForRules}
                              onUpdate={updateRule}
                              onRemove={removeRule}
                          />
                      ))}

                      <Button
                        startIcon={<AddCircleOutlineIcon />}
                        onClick={addRule}
                        disabled={enabledParamsForRules.length < 2}
                        sx={{mt: 1}}
                      >
                        Add Rule
                      </Button>
                    </Box>

                </Box>

                <Divider orientation="vertical" flexItem />

                {/* --- RIGHT PANEL: PARAMETER OPTIMIZATION --- */}
                <Box sx={{ flex: 1 }}>
                    <Typography variant="h6" gutterBottom>Configure Parameters</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                        Enable and configure any parameters you wish to optimize. If none are enabled, the default values from the file will be used.
                    </Typography>
                    <Box sx={{maxHeight: '60vh', overflowY: 'auto', pr: 1}}>
                        {params.filter(p => p.type === 'strategy_param').length > 0 && <Typography variant="overline">Strategy Parameters</Typography>}
                        {params.filter(p => p.type === 'strategy_param').map(param => (
                            <ParameterInputRow key={param.id} param={param} handleParamChange={handleParamChange} isSubmitting={isSubmitting} />
                        ))}
                        
                        {params.filter(p => p.type === 'indicator_param').length > 0 && <Typography variant="overline" sx={{mt:2}}>Indicator Parameters</Typography>}
                        {params.filter(p => p.type === 'indicator_param').map(param => (
                            <ParameterInputRow key={param.id} param={param} handleParamChange={handleParamChange} isSubmitting={isSubmitting} />
                        ))}
                    </Box>
                </Box>

            </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={isSubmitting}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained" disabled={isLoading || isSubmitting}>Run Optimization</Button>
      </DialogActions>
    </Dialog>
  );
};

const CombinationRuleRow: React.FC<{rule: CombinationRule, availableParams: OptimizableParameter[], onUpdate: Function, onRemove: Function}> = ({rule, availableParams, onUpdate, onRemove}) => {
    const operators = ['<', '>', '<=', '>=', '===', '!=='];
    return (
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 1, p: 1.5, borderRadius: 2, bgcolor: 'action.hover' }}>
            <Select size="small" value={rule.param1} onChange={e => onUpdate(rule.id, 'param1', e.target.value)} displayEmpty fullWidth>
                <MenuItem value="" disabled>Select Param...</MenuItem>
                {availableParams.map(p => <MenuItem key={p.id} value={p.id}>{p.name}</MenuItem>)}
            </Select>
            <Select size="small" value={rule.operator} onChange={e => onUpdate(rule.id, 'operator', e.target.value)}>
                {operators.map(op => <MenuItem key={op} value={op}>{op}</MenuItem>)}
            </Select>
            <Select size="small" value={rule.param2} onChange={e => onUpdate(rule.id, 'param2', e.target.value)} displayEmpty fullWidth>
                <MenuItem value="" disabled>Select Param...</MenuItem>
                {availableParams.map(p => <MenuItem key={p.id} value={p.id}>{p.name}</MenuItem>)}
            </Select>
            <IconButton onClick={() => onRemove(rule.id)} color="error"><DeleteOutlineIcon /></IconButton>
        </Box>
    );
}

// --- Helper Component for each parameter's input row (from previous step) ---
const ParameterInputRow: React.FC<{param: OptimizableParameter, handleParamChange: Function, isSubmitting: boolean}> = ({param, handleParamChange, isSubmitting}) => {
    return (
        <Box sx={{ p: 2, mb: 1, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <FormControlLabel
                    control={<Checkbox checked={param.enabled} onChange={e => handleParamChange(param.id, 'enabled', e.target.checked)} />}
                    label={<Typography sx={{ fontWeight: 500 }}>{param.name}</Typography>}
                />
                <ToggleButtonGroup
                    size="small"
                    value={param.mode}
                    exclusive
                    disabled={!param.enabled || isSubmitting}
                    onChange={(e, newMode) => { if (newMode) handleParamChange(param.id, 'mode', newMode) }}
                >
                    <ToggleButton value="range">Range</ToggleButton>
                    <ToggleButton value="list">List</ToggleButton>
                </ToggleButtonGroup>
            </Box>
            
            {param.mode === 'range' ? (
                <Box sx={{ display: 'flex', gap: 2, pl: 6 }}>
                    <TextField fullWidth label="Start" type="number" variant="outlined" size="small" value={param.start} onChange={e => handleParamChange(param.id, 'start', parseFloat(e.target.value) || 0)} disabled={!param.enabled || isSubmitting} InputLabelProps={{ shrink: true }}/>
                    <TextField fullWidth label="End" type="number" variant="outlined" size="small" value={param.end} onChange={e => handleParamChange(param.id, 'end', parseFloat(e.target.value) || 0)} disabled={!param.enabled || isSubmitting} InputLabelProps={{ shrink: true }}/>
                    <TextField fullWidth label="Step" type="number" variant="outlined" size="small" value={param.step} onChange={e => handleParamChange(param.id, 'step', parseFloat(e.target.value) || 1)} disabled={!param.enabled || isSubmitting} InputLabelProps={{ shrink: true }}/>
                </Box>
            ) : (
                <Box sx={{ pl: 6 }}>
                    <TextField
                        fullWidth
                        label="Values (comma-separated)"
                        variant="outlined"
                        size="small"
                        value={param.list_values}
                        onChange={e => handleParamChange(param.id, 'list_values', e.target.value)}
                        placeholder="e.g., 20, 50, 100, 155"
                        disabled={!param.enabled || isSubmitting}
                        InputLabelProps={{ shrink: true }}
                    />
                </Box>
            )}
        </Box>
    );
}