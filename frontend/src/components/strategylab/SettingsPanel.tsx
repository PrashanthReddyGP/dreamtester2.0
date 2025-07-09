import React, {useState} from 'react';
import { Box, Typography, Button, MenuItem, FormControl, InputLabel, Select } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SaveAltIcon from '@mui/icons-material/SaveAlt';
import { DatePicker, LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { Save } from 'lucide-react';

export const SettingsPanel: React.FC<{
  onSave: () => void;
  isSaveDisabled: boolean;
  onRunBacktest: () => void;
  isBacktestRunning: boolean;
}> = ({ onSave, isSaveDisabled, onRunBacktest, isBacktestRunning }) => {

  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
      <Box sx={{
          height: '100%',
          bgcolor: 'background.paper',
          p: 2,
          display: 'flex',
          flexDirection: 'column',
      }}>
        <Typography variant="h2" sx={{ mb: 3 }}>Configuration</Typography>
        
        <FormControl fullWidth margin="normal" variant="filled">
          <InputLabel>Asset</InputLabel>
          <Select defaultValue={'BTC/USDT'}><MenuItem value={'BTC/USDT'}>BTC/USDT</MenuItem></Select>
        </FormControl>
        <FormControl fullWidth margin="normal" variant="filled">
          <InputLabel>Timeframe</InputLabel>
          <Select defaultValue={'1h'}><MenuItem value={'1h'}>1 hour</MenuItem></Select>
        </FormControl>

        <DatePicker label="Start Date" sx={{ width: '100%', mt: 1 }} slotProps={{ textField: { variant: 'filled' } }}/>
        <DatePicker label="End Date" sx={{ width: '100%', mt: 2, mb: 2 }} slotProps={{ textField: { variant: 'filled' } }}/>

        <Box sx={{ flexGrow: 1 }} /> 

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          <Button variant="contained" size="large" startIcon={<PlayArrowIcon />} fullWidth onClick={onRunBacktest} disabled={isBacktestRunning || isSaveDisabled}>
            {isBacktestRunning ? 'Running...' : 'Run Backtest'}
          </Button>
          <Button 
            variant="outlined" 
            size="large" 
            startIcon={<Save />} 
            fullWidth
            onClick={onSave} // Call the function from the parent
            disabled={isSaveDisabled} // Disable if no file is selected
          >
            Save Strategy
          </Button>
        </Box>
      </Box>
    </LocalizationProvider>
  );
};