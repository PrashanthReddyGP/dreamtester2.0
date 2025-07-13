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
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { DatePicker, LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import dayjs from 'dayjs';

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

interface OptimizeModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (config: OptimizationConfig) => void;
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

const fetchOptimizableParams = async (code: string): Promise<OptimizableParameter[]> => {
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

  useEffect(() => {
    if (open && strategyCode) {
      setIsLoading(true);
      setError(null);
      fetchOptimizableParams(strategyCode)
        .then(data => setParams(data))
        .catch(err => setError(err.message))
        .finally(() => setIsLoading(false));
    }
  }, [open, strategyCode]);


  const handleParamChange = (id: string, field: keyof OptimizableParameter, value: any) => {
    setParams(prev => prev.map(p => p.id === id ? { ...p, [field]: value } : p));
  };

  const handleSubmit = () => {
    if (!strategyCode) {
      setError("Strategy code is missing."); return;
    }
    const enabledParams = params.filter(p => p.enabled);
    if (enabledParams.length === 0) {
      setError("Please enable at least one parameter to optimize."); return;
    }
    const config: OptimizationConfig = {
      strategy_code : strategyCode,
      parameters_to_optimize: enabledParams,
    };
    onSubmit(config);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        Configure Optimization
        <IconButton aria-label="close" onClick={onClose}><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {isLoading && ( <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}><CircularProgress /><Typography sx={{ml:2}}>Parsing Strategy...</Typography></Box> )}
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
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
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={isSubmitting}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained" disabled={isLoading || isSubmitting}>Run Optimization</Button>
      </DialogActions>
    </Dialog>
  );
};
