import React from 'react';
import type { FC } from 'react';
import { Box, List, ListItemButton, ListItemIcon, ListItemText, Typography, Paper, Button, Tooltip } from '@mui/material';
import AssessmentIcon from '@mui/icons-material/Assessment'; // Icon for backtest results
import { RefreshCcw } from 'lucide-react';

// Define the shape of our strategy/backtest data
export interface BacktestResult {
  id: string;
  name: string;
}

interface StrategyListPanelProps {
  results: BacktestResult[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onReload: () => void;
  isLoading: boolean;
}

export const StrategyListPanel: FC<StrategyListPanelProps> = ({ results, selectedId, onSelect, onReload, isLoading }) => {

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        height: '100%', 
        p: 2, 
        borderRight: 1, 
        borderColor: 'divider',
        display: 'flex', // Use flexbox for layout
        flexDirection: 'column'
      }}
    >
      {/* --- ADD A HEADER FOR THE PANEL --- */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        mb: 2 
      }}>
        <Tooltip title="Refresh Results">
          <Button variant="contained" color="primary" startIcon={<RefreshCcw />} sx={{ mb: 2, width:'100%' }} onClick={onReload}>
            Reload Analytics
          </Button>
        </Tooltip>
      </Box>
      
      {/* --- MAKE THE LIST SCROLLABLE --- */}
      <Box sx={{ flexGrow: 1, overflowY: 'auto' }}>
        <List component="nav" dense>
          {results.map((result) => (
            <ListItemButton
              key={result.id}
              selected={selectedId === result.id}
              onClick={() => onSelect(result.id)}
              sx={{ borderRadius: 2, mb: 0.5 }}
            >
              <ListItemIcon sx={{ minWidth: 32 }}>
                <AssessmentIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText primary={result.name} />
            </ListItemButton>
          ))}
        </List>
      </Box>
    </Paper>
  );
};