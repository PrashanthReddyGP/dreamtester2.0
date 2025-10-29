import React, { useState, useEffect} from 'react';
import { Box, Typography, Button, Divider, FormControlLabel, Checkbox, TextField, Alert, Autocomplete, CircularProgress, MenuItem } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { Save, TestTube, FileDown, FileUp } from 'lucide-react';
import { fetchAvailableSymbols } from '../../services/api';
import CandlestickChartIcon from '@mui/icons-material/CandlestickChart';
import { CsvImportDialog } from './CsvImportDialog';
import { uploadCsvData } from '../../services/api';
import type { FileSystemItem } from './ExplorerPanel';

// Define the shape of the data source configuration
export interface DataSourceConfig {
    symbol: string;
    timeframe: string;
    startDate: string;
    endDate: string;
}

export interface PortfolioConfig {
    id: string; // This is the file ID of the selected template
    code: string;
}

// Define the available timeframes in a constant
const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];

export const SettingsPanel: React.FC<{
  dataSourceConfig: DataSourceConfig;
  onDataSourceChange: (field: keyof DataSourceConfig, value: any) => void;
  onSave: () => Promise<void>;
  isSaveDisabled: boolean;
  onRunBacktest: (useTrainingSet: boolean) => void;
  onOptimizeStrategy: () => void;
  onDurabilityTests: () => void;
  onCharting: () => void;
  onHedgeOptimize: () => void;
  onDownloadCSV: () => void;
  isBacktestRunning: boolean;
  onFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onClearCsv: () => void;
  selectedCsvFile: File | null;
  portfolioConfig: PortfolioConfig;
  portfolioTemplates: FileSystemItem[]; // <-- NEW: Receive the list of templates
  onPortfolioChange: (fileId: string) => void; // <-- MODIFIED: Handler only needs the ID
}> = ({ dataSourceConfig, onDataSourceChange, onSave, isSaveDisabled, onRunBacktest, onDownloadCSV, onOptimizeStrategy, onDurabilityTests, onCharting, onHedgeOptimize, isBacktestRunning, onFileChange, onClearCsv, selectedCsvFile, portfolioConfig, portfolioTemplates, onPortfolioChange }) => {
    
    const [useTrainingSet, setUseTrainingSet] = useState(true);

    // State for fetching the list of all available symbols
    const [symbolList, setSymbolList] = useState<string[]>([]);
    const [isFetchingSymbols, setIsFetchingSymbols] = useState(false);
    const [fetchError, setFetchError] = useState<string | null>(null);

    // State to control the new import dialog
    const [isImportDialogOpen, setIsImportDialogOpen] = useState(false);

    // Fetch the available symbols once when the component mounts
    useEffect(() => {
        const loadSymbols = async () => {
            setIsFetchingSymbols(true);
            setFetchError(null);
            try {
                const fetchedSymbols = await fetchAvailableSymbols('binance');
                setSymbolList(fetchedSymbols);

                // If no symbol is currently set in the config, default to BTCUSDT
                if (!dataSourceConfig.symbol && fetchedSymbols.includes('BTCUSDT')) {
                    onDataSourceChange('symbol', 'BTCUSDT');
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

    // This function will be called by the dialog on submission
    const handleImportSubmit = async (symbol: string, timeframe: string, source: string, file: File) => {
      try {
          // We'll create `uploadCsvData` in the next step.
          const response = await uploadCsvData(symbol, timeframe, source, file);
          console.log('Import successful:', response);
          // Optional: Show a success notification (snackbar) to the user.
          // Optional: Re-fetch symbols or update the symbol list if the new one isn't there.
      } catch (error) {
          console.error('Failed to import CSV:', error);
          // Re-throw the error so the dialog can display it
          throw error;
      }
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
        {fetchError && <Alert severity="error" sx={{ width: '100%', mb: 2}}>{fetchError}</Alert>}
        
        <Autocomplete
            id="symbol-autocomplete-settings"
            options={symbolList}
            value={dataSourceConfig.symbol || null}
            onChange={(event, newValue) => {
                onDataSourceChange('symbol', newValue || '');
            }}
            loading={isFetchingSymbols}
            fullWidth
            sx={{ mb: 2 }}
            renderInput={(params) => (
                <TextField
                    {...params}
                    label="Symbol"
                    size="medium"
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
            size="medium"
            value={dataSourceConfig.timeframe}
            onChange={(e) => onDataSourceChange('timeframe', e.target.value)}
            fullWidth
            sx={{ mb: 1 }}
        >
            {timeframes.map((option) => (
                <MenuItem key={option} value={option}>
                    {option}
                </MenuItem>
            ))}
        </TextField>
        <TextField 
          label="Start Date" 
          type="date" 
          size="medium" 
          value={dataSourceConfig.startDate} 
          onChange={(e) => onDataSourceChange('startDate', e.target.value)} 
          InputLabelProps={{ shrink: true }}  
          fullWidth 
          sx={{ mt: 1 }}
        />
        <TextField 
          label="End Date" 
          type="date" 
          size="medium" 
          value={dataSourceConfig.endDate} 
          onChange={(e) => onDataSourceChange('endDate', e.target.value)} 
          InputLabelProps={{ shrink: true }}  
          fullWidth 
          sx={{ mt: 2, mb: 2 }}
        />

        <Box sx={{ flexGrow: 1 }} /> 

        <Divider sx={{ mb: 2 }}/>

        <TextField
            select
            label="Portfolio Template"
            size="medium"
            value={portfolioConfig.id || ''}
            onChange={(e) => onPortfolioChange(e.target.value)}
            fullWidth
            disabled={portfolioTemplates.length === 0} // Disable if no templates exist
            helperText={portfolioTemplates.length === 0 ? "No templates in 'Portfolio' folder" : ""}
        >
            {portfolioTemplates.map((template) => (
                <MenuItem key={template.id} value={template.id}>
                    {template.name}
                </MenuItem>
            ))}
        </TextField>

        <Divider sx={{ mt: 2, mb: 2 }}/>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          <Button variant="contained" size="large" startIcon={<CandlestickChartIcon />} fullWidth onClick={onCharting}>
            Charting View
          </Button>
        </Box>
        
        <Divider sx={{ mt: 2, mb: 2 }}/>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          <Button variant="outlined" size="large" startIcon={<TestTube />} fullWidth onClick={onDurabilityTests}>
            Durability Tests
          </Button>
          <Button 
            variant="contained" 
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
            <Button
              variant="outlined"
              size='medium'
              sx={{ height:'40px' }}
              startIcon={<FileUp />}
              fullWidth
              onClick={() => setIsImportDialogOpen(true)} // Open the dialog on click
            >
              Upload Data from CSV
            </Button>
            
            <Button variant="contained" size="medium" startIcon={<FileDown />} sx={{ height:'40px' }} fullWidth onClick={onDownloadCSV}>
              {isBacktestRunning ? 'Processing Data...' : 'Download Data to CSV'}
            </Button>

            {/* <Button variant="contained" size="medium" startIcon={<FileDown />} sx={{ height:'45px' }} fullWidth onClick={onOptimizeStrategy} disabled={isBacktestRunning || isSaveDisabled}>
              {isBacktestRunning ? 'Processing Data...' : 'Optimize Strategy'}
            </Button> */}
          </Box>

          <CsvImportDialog
            open={isImportDialogOpen}
            onClose={() => setIsImportDialogOpen(false)}
            onSubmit={handleImportSubmit}
          />

        <Divider sx={{ mt: 2, mb: 1 }}/>

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
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, mt: 1.5 }}>

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