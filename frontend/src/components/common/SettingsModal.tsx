import React, { useState, useEffect } from 'react';
import { 
    Dialog, 
    DialogTitle, 
    DialogContent, 
    DialogActions, 
    Button, 
    IconButton, 
    Box, 
    Tabs, 
    Tab, 
    TextField, 
    Typography,
    InputAdornment, 
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    CircularProgress,
    Alert
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close';

import { useAppContext } from '../../context/AppContext';

// --- This part remains the same ---
interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}
// --- End of unchanged part ---


export const SettingsModal: React.FC<SettingsModalProps> = ({ open, onClose }) => {
  // --- ADDED: Get global state and functions from our context ---
  const { settings, saveApiKeys } = useAppContext();

  // Local UI state remains the same
  const [tabIndex, setTabIndex] = useState(0);
  const [selectedExchange, setSelectedExchange] = useState('binance');
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<{type: 'success' | 'error', message: string} | null>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabIndex(newValue);
    setFeedback(null); // Clear feedback when switching tabs
  };

  const handleExchangeChange = (event: SelectChangeEvent) => {
      setSelectedExchange(event.target.value);
  }

  // --- MODIFIED: Effect now reads from context instead of fetching ---
  // This effect synchronizes the form fields with the global 'settings' state.
  useEffect(() => {
    // Find the keys for the currently selected exchange in our global settings object
    const currentSettings = settings[selectedExchange];
    
    if (currentSettings) {
        setApiKey(currentSettings.apiKey || '');
    } else {
        // If no settings are found for this exchange, clear the field
        setApiKey('');
    }
    
    // Always clear the secret and feedback when the selected exchange changes
    setApiSecret('');
    setFeedback(null);

  }, [selectedExchange, settings]); // Re-run when the user changes the exchange or when global settings are loaded/updated

  // --- MODIFIED: Handler now uses the function from the context ---
  const handleSaveKeys = async () => {
    setLoading(true);
    setFeedback(null);
    try {
        // Call the function from AppContext
        const result = await saveApiKeys(selectedExchange, apiKey, apiSecret);

        setFeedback({ type: 'success', message: result.message });
        setApiSecret(''); // Clear secret field after successful save
    } catch (e) {
        console.error(e);
        const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
        setFeedback({ type: 'error', message: `Failed to save keys: ${errorMessage}` });
    }
    setLoading(false);
  };
  
  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      
      <DialogTitle sx={{ m: 0, p: 2 }}>
        Application Settings
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{
            position: 'absolute',
            right: 8,
            top: 8,
            color: (theme) => theme.palette.grey[500],
          }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
            
      <DialogContent dividers sx={{ p: 0 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabIndex} onChange={handleTabChange} variant="fullWidth">
            <Tab label="API Keys" />
            <Tab label="Preferences" />
          </Tabs>
        </Box>

        <TabPanel value={tabIndex} index={0}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Your API keys are stored securely on your local machine and are never sent to any external servers.
          </Typography>
          <FormControl fullWidth margin="normal">
            <InputLabel>Exchange</InputLabel>
            <Select value={selectedExchange} onChange={handleExchangeChange} label="Exchange">
              <MenuItem value="binance">Binance</MenuItem>
              <MenuItem value="bybit">Bybit</MenuItem>
              <MenuItem value="coinbase">Coinbase</MenuItem>
              <MenuItem value="dydx">Dydx</MenuItem>
              <MenuItem value="mexc">Mexc</MenuItem>
            </Select>
          </FormControl>
          <TextField
            margin="normal"
            fullWidth
            label="API Key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            variant="outlined"
          />
          <TextField
            margin="normal"
            fullWidth
            label="API Secret"
            type="password"
            value={apiSecret}
            onChange={(e) => setApiSecret(e.target.value)}
            placeholder="Enter new secret to update"
            variant="outlined"
          />
          
          {feedback && <Alert severity={feedback.type} sx={{mt: 2}}>{feedback.message}</Alert>}

           <Button onClick={handleSaveKeys} variant='contained' sx={{mt: 2}} disabled={loading || !apiKey || !apiSecret}>
            {loading ? <CircularProgress size={24} /> : 'Save Keys'}
           </Button>
        </TabPanel>

        <TabPanel value={tabIndex} index={1}>
          {/* This part remains the same, but you would wire it up to context similarly */}
           <TextField
            margin="normal"
            label="Default Initial Capital"
            defaultValue="10000"
            fullWidth
            InputProps={{ startAdornment: <InputAdornment position="start">$</InputAdornment> }}
          />
        </TabPanel>
      </DialogContent>

      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose}>Close</Button>
        {/* Note: This "Save Settings" button might be for the Preferences tab. 
            We've implemented the save logic on the "Save Keys" button inside the tab.
            You can decide to have one master save button or per-tab saves. */}
      </DialogActions>
    </Dialog>
  );
};