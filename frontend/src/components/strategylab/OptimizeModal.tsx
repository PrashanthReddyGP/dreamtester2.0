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
  strategyCode: string;
  asset: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  parametersToOptimize: {
      indicatorIndex: number;
      paramIndex: number;
      name: string;
      start: number;
      end: number;
      step: number;
  }[];
}

interface OptimizeModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (config: OptimizationConfig) => void;
  strategyCode: string | null;
  isSubmitting: boolean;
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

// --- The Main Modal Component ---
export const OptimizeModal: React.FC<OptimizeModalProps> = ({
  open,
  onClose,
  onSubmit,
  strategyCode,
  isSubmitting,
}) => {
  const [indicators, setIndicators] = useState<OptimizableIndicator[]>([]);
  const [settings, setSettings] = useState<StrategySettings | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open && strategyCode) {
      setIsLoading(true);
      setError(null);
      setIndicators([]);
      setSettings(null);

      fetchParsedStrategyData(strategyCode)
        .then(data => {
            setIndicators(data.indicators);
            setSettings(data.settings);
        })
        .catch(err => setError(err.message))
        .finally(() => setIsLoading(false));
    }
  }, [open, strategyCode]);

  const handleParamChange = (indicatorId: string, paramId: string, field: keyof IndicatorParam, value: any) => {
    setIndicators(prev => prev.map(ind =>
      ind.id === indicatorId
        ? { ...ind, params: ind.params.map(p => p.id === paramId ? { ...p, [field]: value } : p) }
        : ind
    ));
  };

  const handleSubmit = () => {
    if (!strategyCode || !settings) {
      setError("Error: Strategy code or settings are missing.");
      return;
    }

    const parametersToOptimize: OptimizationConfig['parametersToOptimize'] = [];
    
    indicators.forEach(indicator => {
      indicator.params.forEach(param => {
        if (param.enabled) {
          parametersToOptimize.push({
            indicatorIndex: indicator.originalIndex,
            paramIndex: param.originalIndex,
            name: `${indicator.name}_param_${param.originalIndex}`, 
            start: param.start,
            end: param.end,
            step: param.step,
          });
        }
      });
    });

    if (parametersToOptimize.length === 0) {
      setError("Please enable at least one parameter to optimize.");
      return;
    }

    const config: OptimizationConfig = {
      strategyCode,
      asset: settings.symbol,
      timeframe: settings.timeframe,
      startDate: settings.startDate,
      endDate: settings.endDate,
      parametersToOptimize,
    };
    onSubmit(config);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        Configure Strategy Optimization
        <IconButton aria-label="close" onClick={onClose}><CloseIcon /></IconButton>
      </DialogTitle>
      
      <DialogContent dividers>
        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', my: 4, gap: 2 }}>
            <CircularProgress />
            <Typography>Parsing strategy file...</Typography>
          </Box>
        )}
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        
        {!isLoading && !error && settings && (
          <Box>
            {indicators.length === 0 ? (
              <Alert severity="info" sx={{ mt: 2 }}>No indicators found in the `self.indicators` list.</Alert>
            ) : (
              indicators.map(indicator => (
                <Box key={indicator.id} sx={{ mb: 2 }}>
                  <Typography sx={{ fontWeight: 500 }}>{indicator.name}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Default: ({indicator.params.map(p => p.value).join(', ')})
                  </Typography>
                  {indicator.params.map(param => (
                    <Box key={param.id} sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 1, ml: 2, mt: 1 }}>
                      <FormControlLabel
                        control={<Checkbox checked={param.enabled} onChange={e => handleParamChange(indicator.id, param.id, 'enabled', e.target.checked)} />}
                        label={`Param #${param.originalIndex + 1}`}
                      />
                      <TextField label="Start" type="number" variant="outlined" size="small" defaultValue={param.value} onChange={e => handleParamChange(indicator.id, param.id, 'start', parseFloat(e.target.value))} disabled={!param.enabled || isSubmitting} />
                      <TextField label="End" type="number" variant="outlined" size="small" defaultValue={param.value} onChange={e => handleParamChange(indicator.id, param.id, 'end', parseFloat(e.target.value))} disabled={!param.enabled || isSubmitting} />
                      <TextField label="Step" type="number" variant="outlined" size="small" defaultValue={1} onChange={e => handleParamChange(indicator.id, param.id, 'step', parseFloat(e.target.value))} disabled={!param.enabled || isSubmitting} />
                    </Box>
                  ))}
                </Box>
              ))
            )}
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={isSubmitting}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained" disabled={isLoading || isSubmitting || indicators.length === 0}>
          {isSubmitting ? 'Optimizing...' : 'Run Optimization'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};
