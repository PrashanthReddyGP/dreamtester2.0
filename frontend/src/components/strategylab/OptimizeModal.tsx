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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  Autocomplete,
  Chip,
  ToggleButtonGroup,
  ToggleButton
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { DatePicker, LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import dayjs from 'dayjs';
import { fetchAvailableSymbols } from '../../services/api';

// This should be your actual backend URL
const API_URL = 'http://127.0.0.1:8000'; 

// --- Type Definitions for Indicator-based Optimization ---

// Represents a single parameter within an indicator's tuple, e.g., the '50' in ('SMA', '1m', (50,))
interface IndicatorParam {
  id: string;
  originalIndex: number;
  value: number;
  enabled: boolean;
  start: number;
  end: number;
  step: number;
}

// Represents a full indicator entry from the list, e.g., ('SMA', '1m', (50,))
interface OptimizableIndicator {
  id: string;
  name: string;
  originalIndex: number;
  params: IndicatorParam[];
}

interface StrategySettings {
  symbol: string;
  timeframe: string;
  startDate: string;
  endDate: string;
}

interface ParsedStrategyData {
    settings: StrategySettings;
    indicators: OptimizableIndicator[];
}

// The final configuration object that will be sent to the backend to start the optimization
export interface OptimizationConfig {
  strategy_code: string;
  parameters_to_optimize: OptimizableParameter[]; // Send the whole object
}

export interface TestSubmissionConfig {
  mode: 'parameters' | 'assets';
  strategy_code: string;
  parameters_to_optimize?: OptimizableParameter[]; // Optional for asset screening
  symbols_to_screen?: string[]; // Optional for parameter optimization
}

// interface OptimizeModalProps {
//   open: boolean;
//   onClose: () => void;
//   onSubmit: (config: OptimizationConfig) => void;
//   strategyCode: string | null;
//   isSubmitting: boolean;
// }
interface OptimizeModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (config: TestSubmissionConfig) => void; 
  strategyCode: string | null;
  isSubmitting: boolean;
}

interface OptimizableParameter {
    id: string;
    type: 'strategy_param' | 'indicator_param';
    name: string;
    value: number;
    enabled: boolean;
    start: number;
    end: number;
    step: number;
    indicatorIndex?: number;
    paramIndex?: number;
}

export interface SuperOptimizationConfig {
  strategy_code: string;
  parameters_to_optimize: OptimizableParameter[]; // Could be empty
  symbols_to_screen: string[]; // Could be empty
}

// A list of available symbols. In a real app, you might fetch this from an API.
const availableSymbols = [
  'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
  'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT'
];

const fetchParsedStrategyData = async (code: string): Promise<ParsedStrategyData> => {
  console.log("Parsing strategy file for settings and indicators...");
  const response = await fetch(`${API_URL}/api/strategies/parse-indicators`, {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: code,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "An unknown error occurred" }));
    throw new Error(errorData.detail || "Failed to parse strategy file from the backend.");
  }
  return response.json();
};

// const fetchOptimizableParams = async (code: string): Promise<OptimizableParameter[]> => {
//     const response = await fetch(`${API_URL}/api/strategies/parse-parameters`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'text/plain' },
//         body: code,
//     });
//     if (!response.ok) {
//         const errorData = await response.json().catch(() => ({ detail: "An unknown error" }));
//         throw new Error(errorData.detail);
//     }
//     const data = await response.json();
//     return data.optimizable_params.map((p: any) => ({
//         ...p,
//         enabled: false, start: p.value, end: p.value, step: 1,
//     }));
// };
const fetchAllParameters = async (code: string): Promise<OptimizableParameter[]> => {
    const response = await fetch(`${API_URL}/api/strategies/parse-parameters`, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: code,
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "An unknown error" }));
        throw new Error(errorData.detail);
    }
    const data = await response.json();
    // Map the backend response to the frontend state shape
    return data.optimizable_params.map((p: any) => ({
        ...p,
        enabled: false, start: p.value, end: p.value, step: 1,
    }));
};

// --- The Main Modal Component ---
export const OptimizeModal: React.FC<OptimizeModalProps> = ({ open, onClose, onSubmit, strategyCode, isSubmitting }) => {
  const [params, setParams] = useState<OptimizableParameter[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [mode, setMode] = useState<'parameters' | 'assets'>('parameters');

  const [symbolList, setSymbolList] = useState<string[]>([]);
  const [isFetchingSymbols, setIsFetchingSymbols] = useState(false);
  
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['BTCUSDT', 'ETHUSDT']);

  useEffect(() => {
      // We only need to fetch this once.
      if (open) {
          setIsFetchingSymbols(true);
          fetchAvailableSymbols('binance') // Hardcode binance for now, or make it dynamic
              .then(symbols => {
                  setSymbolList(symbols);
              })
              .finally(() => {
                  setIsFetchingSymbols(false);
              });
      }
  }, [open]); // Run only when the modal is opened


  useEffect(() => {
    if (open && strategyCode && mode === 'parameters') {
      setIsLoading(true);
      setError(null);
      fetchAllParameters(strategyCode)
        .then(data => setParams(data))
        .catch(err => setError(err.message))
        .finally(() => setIsLoading(false));
    }
  }, [open, strategyCode, mode]); // Add `mode` as a dependency

  const handleParamChange = (id: string, field: keyof OptimizableParameter, value: any) => {
    setParams(prev => prev.map(p => p.id === id ? { ...p, [field]: value } : p));
  };

  const handleModeChange = (event: React.MouseEvent<HTMLElement>, newMode: 'parameters' | 'assets' | null) => {
      if (newMode !== null) {
          setMode(newMode);
          setError(null); // Clear errors when switching modes
      }
  };

  const handleSubmit = () => {
    if (!strategyCode) {
      setError("Strategy code is missing."); return;
    }

    let config: TestSubmissionConfig;

    if (mode === 'parameters') {
      const enabledParams = params.filter(p => p.enabled);
      if (enabledParams.length === 0) {
        setError("Please enable at least one parameter to optimize."); return;
      }
      config = {
        mode: 'parameters',
        strategy_code: strategyCode,
        parameters_to_optimize: enabledParams,
      };
    } else { // mode === 'assets'
      if (selectedSymbols.length === 0) {
        setError("Please select at least one asset to screen."); return;
      }  
      config = {
        mode: 'assets',
        strategy_code: strategyCode,
        symbols_to_screen: selectedSymbols,
      };
    }
    // Call the single, unified onSubmit prop
    onSubmit(config);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        Configure Advanced Test
        <IconButton aria-label="close" onClick={onClose}><CloseIcon/></IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center', mb: 2 }}>
            <ToggleButtonGroup value={mode} exclusive onChange={handleModeChange} aria-label="testing mode">
                <ToggleButton value="parameters">Optimize Parameters</ToggleButton>
                <ToggleButton value="assets">Screen Assets</ToggleButton>
            </ToggleButtonGroup>
        </Box>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        {mode === 'parameters' && (

          <>
            {isLoading && ( <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}><CircularProgress /><Typography sx={{ml:2}}>Parsing Parameters...</Typography></Box> )}
            
            {!isLoading && !error && (
                <Box>
                    <Typography variant="h6">Strategy Parameters</Typography>
                    {params.filter(p => p.type === 'strategy_param').map(param => (
                        <Box key={param.id} sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 1, ml: 2 }}>
                            <FormControlLabel control={<Checkbox checked={param.enabled} onChange={e => handleParamChange(param.id, 'enabled', e.target.checked)} />} label={param.name} />
                            <TextField label="Start" type="number" variant="outlined" size="small" defaultValue={param.value} onChange={e => handleParamChange(param.id, 'start', parseFloat(e.target.value))} disabled={!param.enabled || isSubmitting} />
                            <TextField label="End" type="number" variant="outlined" size="small" defaultValue={param.value} onChange={e => handleParamChange(param.id, 'end', parseFloat(e.target.value))} disabled={!param.enabled || isSubmitting} />
                            <TextField label="Step" type="number" variant="outlined" size="small" defaultValue={1} onChange={e => handleParamChange(param.id, 'step', parseFloat(e.target.value))} disabled={!param.enabled || isSubmitting} />
                        </Box>
                    ))}
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="h6">Indicator Parameters</Typography>
                    {params.filter(p => p.type === 'indicator_param').map(param => (
                        <Box key={param.id} sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 1, ml: 2 }}>
                            <FormControlLabel control={<Checkbox checked={param.enabled} onChange={e => handleParamChange(param.id, 'enabled', e.target.checked)} />} label={param.name} />
                            <TextField label="Start" type="number" variant="outlined" size="small" defaultValue={param.value} onChange={e => handleParamChange(param.id, 'start', parseFloat(e.target.value))} disabled={!param.enabled || isSubmitting} />
                            <TextField label="End" type="number" variant="outlined" size="small" defaultValue={param.value} onChange={e => handleParamChange(param.id, 'end', parseFloat(e.target.value))} disabled={!param.enabled || isSubmitting} />
                            <TextField label="Step" type="number" variant="outlined" size="small" defaultValue={1} onChange={e => handleParamChange(param.id, 'step', parseFloat(e.target.value))} disabled={!param.enabled || isSubmitting} />
                        </Box>
                    ))}
                </Box>
            )}  
          </>
        )}

        {mode === 'assets' && (

          <Box>
              <Typography variant="h6" gutterBottom>Asset Universe</Typography>
              <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                  Select the assets to run this strategy against. The parameters defined in the file will be used for all runs.
              </Typography>
              <Autocomplete
                  multiple id="asset-screener-autocomplete" options={symbolList} value={selectedSymbols} loading={isFetchingSymbols}
                  onChange={(event, newValue) => { setSelectedSymbols(newValue); }}
                  freeSolo
                      renderTags={(value: readonly string[], getTagProps) =>
                          value.map((option: string, index: number) => {
                              // 1. Get all the props for the tag
                              const { key, ...tagProps } = getTagProps({ index });
                              // 2. Apply `key` directly and spread the rest
                              return <Chip key={key} variant="outlined" label={option} {...tagProps} />;
                          })
                      }
                  renderInput={(params) => (
                    <TextField {...params} variant="outlined" label="Symbols to Test" placeholder="Add more symbols..."
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
