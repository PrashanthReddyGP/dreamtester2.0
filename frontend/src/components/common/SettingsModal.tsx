import React, { useState } from 'react';
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
    MenuItem
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

// Helper component for tab content
function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ open, onClose }) => {
  const [tabIndex, setTabIndex] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabIndex(newValue);
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle sx={{ m: 0, p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6">Application Settings</Typography>
        <IconButton aria-label="close" onClick={onClose}>
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

        {/* API Keys Tab Content */}
        <TabPanel value={tabIndex} index={0}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Your API keys are stored securely on your local machine and are never sent to any external servers.
          </Typography>
          <FormControl fullWidth margin="normal">
            <InputLabel>Exchange</InputLabel>
            <Select defaultValue={'binance'} label="Exchange">
              <MenuItem value="binance">Binance</MenuItem>
              <MenuItem value="bybit">Bybit</MenuItem>
              <MenuItem value="coinbase">Coinbase</MenuItem>
            </Select>
          </FormControl>
          <TextField
            margin="normal"
            fullWidth
            label="API Key"
            variant="outlined"
          />
          <TextField
            margin="normal"
            fullWidth
            label="API Secret"
            type="password"
            variant="outlined"
          />
           <Button variant='outlined' sx={{mt: 1}}>Test & Save Keys</Button>
        </TabPanel>

        {/* Preferences Tab Content */}
        <TabPanel value={tabIndex} index={1}>
           <TextField
            margin="normal"
            label="Default Initial Capital"
            defaultValue="10000"
            fullWidth
            InputProps={{ startAdornment: <InputAdornment position="start">$</InputAdornment> }}
          />
          <TextField
            margin="normal"
            label="Default Commission"
            defaultValue="0.04"
            fullWidth
            InputProps={{ endAdornment: <InputAdornment position="end">%</InputAdornment> }}
          />
        </TabPanel>
      </DialogContent>

      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose}>Close</Button>
        <Button onClick={onClose} variant="contained">Save Settings</Button>
      </DialogActions>
    </Dialog>
  );
};