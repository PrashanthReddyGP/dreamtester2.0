import React from 'react';
import type { FC } from 'react';
import { Box, List, ListItemButton, ListItemIcon, ListItemText, Typography, Paper } from '@mui/material';
import AssessmentIcon from '@mui/icons-material/Assessment'; // Icon for backtest results

// Define the shape of our strategy/backtest data
export interface BacktestResult {
  id: string;
  name: string;
}

interface StrategyListPanelProps {
  results: BacktestResult[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export const StrategyListPanel: FC<StrategyListPanelProps> = ({ results, selectedId, onSelect }) => {
  return (
    <Paper elevation={0} sx={{ height: '100%', p: 2, borderRight: 1, borderColor: 'divider' }}>
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
    </Paper>
  );
};