import React from 'react';
import { Box, Paper, Typography, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

interface TerminalPanelProps {
  onClose: () => void; // A function to close the panel
}

export const TerminalPanel: React.FC<TerminalPanelProps> = ({ onClose }) => {
  return (
    <Paper 
      elevation={0} 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        backgroundColor: 'black',
        color: 'white',
      }}
    >
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          p: 1, 
          backgroundColor: '#0f0f0fff',
          height:'30px',
        }}
      >
        <Typography variant="body2" fontSize={'0.75rem'} color='#818181ff'>TERMINAL</Typography>
        <IconButton onClick={onClose} size="small" sx={{color: '#818181ff'}}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>
      <Box 
        component="pre" 
        sx={{ 
          flexGrow: 1, 
          p: 1,
          pt: 0,
          pl: 2, 
          overflowY: 'auto',
          fontSize: '0.7rem',
          whiteSpace: 'pre-wrap',
          wordWrap: 'break-word',
        }}
      >
        {/* Placeholder logs. Later, this will come from a global context. */}
        <div>[INFO] Backend server started successfully.</div>
        <div>[INFO] Waiting for backtest job...</div>
      </Box>
    </Paper>
  );
};