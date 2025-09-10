import React, {useRef, useState} from 'react';
import { Box, Typography, Button, MenuItem, FormControl, InputLabel, Select, Divider, IconButton, FormControlLabel, Checkbox } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import { DatePicker, LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { Save, ArrowDown01, TestTube } from 'lucide-react';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import ClearIcon from '@mui/icons-material/Clear';

export const SettingsPanel: React.FC<{
  onSave: () => Promise<void>;
  isSaveDisabled: boolean;
  onRunBacktest: (useTrainingSet: boolean) => void;
  onOptimizeStrategy: () => void;
  onDurabilityTests: () => void;
  onHedgeOptimize: () => void;
  onRunBacktestWithCsv: () => void;
  isBacktestRunning: boolean;
  onFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onClearCsv: () => void;
  selectedCsvFile: File | null;
}> = ({ onSave, isSaveDisabled, onRunBacktest, onRunBacktestWithCsv, onOptimizeStrategy, onDurabilityTests, onHedgeOptimize, isBacktestRunning, onFileChange, onClearCsv, selectedCsvFile }) => {
  const [useTrainingSet, setUseTrainingSet] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

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

        <FormControl fullWidth margin="normal" variant="outlined">
          <InputLabel>Asset</InputLabel>
          <Select defaultValue={'BTC/USDT'}><MenuItem value={'BTC/USDT'}>BTC/USDT</MenuItem></Select>
        </FormControl>
        <FormControl fullWidth margin="normal" variant="outlined">
          <InputLabel>Timeframe</InputLabel>
          <Select defaultValue={'1h'}><MenuItem value={'1h'}>1 hour</MenuItem></Select>
        </FormControl>

        <DatePicker label="Start Date" sx={{ width: '100%', mt: 1 }} slotProps={{ textField: { variant: 'outlined' } }}/>
        <DatePicker label="End Date" sx={{ width: '100%', mt: 2, mb: 2 }} slotProps={{ textField: { variant: 'outlined' } }}/>

        <Box sx={{ flexGrow: 1 }} /> 

        <Divider sx={{ mt: 2, mb: 2 }}/>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          <Button variant="contained" size="large" startIcon={<TestTube />} fullWidth onClick={onDurabilityTests}>
            Durability Tests
          </Button>
          <Button 
            variant="outlined" 
            size="large" 
            startIcon={<CompareArrowsIcon />} 
            fullWidth 
            onClick={onHedgeOptimize}
            disabled={isBacktestRunning || isSaveDisabled}
          >
            Hedge Optimization
          </Button>
        </Box>

        <Divider sx={{ mt: 2, mb: 2 }}/>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>

            <input
              type="file"
              ref={fileInputRef}
              onChange={onFileChange}
              style={{ display: 'none' }}
              accept=".csv"
            />

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Button
              variant="outlined"
              size='large'
              startIcon={<UploadFileIcon />}
              fullWidth
              onClick={handleUploadClick}
              sx={{
                justifyContent: 'center',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                textAlign: 'center',
              }}
            >
              {selectedCsvFile ? selectedCsvFile.name : 'Select CSV File'}
            </Button>
            {selectedCsvFile && (
              <IconButton onClick={(e) => {
              e.stopPropagation(); // Prevent file dialog from opening
              onClearCsv();
              }} size="small">
              <ClearIcon fontSize="small" />
              </IconButton>
            )}
          </Box>
          
          <Button 
            variant="contained" 
            color="secondary" 
            size="large" 
            startIcon={<PlayArrowIcon />} 
            fullWidth
            onClick={onRunBacktestWithCsv}
            disabled={isBacktestRunning || isSaveDisabled || !selectedCsvFile}
            sx={{ mb: 1 }}
          >
            {isBacktestRunning ? 'Running...' : 'Run with CSV'}
          </Button>

          <Divider/>

          <Box>
            <FormControlLabel
              control={
                <Checkbox 
                  checked={useTrainingSet} 
                  onChange={(event) => setUseTrainingSet(event.target.checked)}
                />
              }
              label="Use Training Set"
              sx={{ display: 'block', textAlign: 'center', mb: 1 }}
            />
            <Button 
              variant="contained" 
              size="large" 
              startIcon={<PlayArrowIcon />} 
              fullWidth 
              onClick={() => onRunBacktest(useTrainingSet)} 
              disabled={isBacktestRunning || isSaveDisabled}
            >
              {isBacktestRunning ? 'Running...' : 'Run Backtest'}
            </Button>
          </Box>
          
          <Button variant="outlined" size="large" startIcon={<ArrowDown01 />} fullWidth onClick={onOptimizeStrategy} disabled={isBacktestRunning || isSaveDisabled}>
            {isBacktestRunning ? 'Fetching Data...' : 'Optimize Strategy'}
          </Button>

          <Button 
            variant="contained" 
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