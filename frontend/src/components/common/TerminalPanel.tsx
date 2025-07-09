import React, { useEffect, useRef } from 'react';
import { Box, Paper, Typography, IconButton, useTheme, Button } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useTerminal } from '../../context/TerminalContext';
import type { LogEntry } from '../../context/TerminalContext';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';

interface TerminalPanelProps {
  onClose: () => void; // A function to close the panel
}

// Helper to determine text color based on log level
const getLogColor = (level: LogEntry['level'], theme: any) => {
    switch (level) {
        case 'ERROR':
            return theme.palette.error.main;
        case 'SUCCESS':
            return theme.palette.success.main;
        case 'SYSTEM':
            return theme.palette.info.light;
        case 'INFO':
        default:
            return '#c5c5c5'; // A neutral light grey for standard logs
    }
};

export const TerminalPanel: React.FC<TerminalPanelProps> = ({ onClose }) => {


  const theme = useTheme();
  const { logs, clearLogs } = useTerminal(); // <-- 2. GET STATE FROM CONTEXT
  const endOfLogsRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to the bottom when new logs are added
  useEffect(() => {
    endOfLogsRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);



  return (
    <Paper 
      elevation={0} 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        backgroundColor: 'black',
        color: 'white',
        fontFamily: 'monospace',
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
        <Box>
            <Button
                size="small"
                startIcon={<DeleteSweepIcon fontSize="small" />}
                onClick={clearLogs} // <-- 3. WIRE UP CLEAR BUTTON
                sx={{ mr: 1, color: '#818181ff', textTransform: 'none', '&:hover': { bgcolor: 'rgba(255,255,255,0.08)'} }}
            >
                Clear
            </Button>
            <IconButton onClick={onClose} size="small" sx={{color: '#818181ff'}}>
              <CloseIcon fontSize="small" />
            </IconButton>
        </Box>
      </Box>
      <Box 
        component="div" // Use div instead of pre for more control
        sx={{ 
          flexGrow: 1, 
          p: 1,
          pl: 2, 
          overflowY: 'auto',
          fontSize: '0.8rem', // Slightly larger for readability
          fontFamily: 'inherit',
        }}
      >
        {/* 4. RENDER LOGS FROM CONTEXT */}
        {logs.map((log, index) => (
          <Box key={index} sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
            <Typography variant="body2" component="span" sx={{ color: '#666', fontFamily: 'inherit' }}>
              {log.timestamp}
            </Typography>
            <Typography
              variant="body2"
              component="span"
              sx={{ color: getLogColor(log.level, theme), whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}
            >
              {log.message}
            </Typography>
          </Box>
        ))}
        {/* This empty div is the target for our auto-scrolling */}
        <div ref={endOfLogsRef} />
      </Box>
    </Paper>
  );
};