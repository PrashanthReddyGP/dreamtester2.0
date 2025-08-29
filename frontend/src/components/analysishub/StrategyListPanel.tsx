import React, { useState, useMemo } from 'react';
import type { FC } from 'react';
import { Box, List, ListItemButton, ListItemIcon, ListItemText, Paper, Button, Tooltip, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import AssessmentIcon from '@mui/icons-material/Assessment'; 
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
  onCompareClick: () => void;
  isDataSegmentationMode: boolean; 
}

export const StrategyListPanel: FC<StrategyListPanelProps> = ({ results, selectedId, onSelect, onCompareClick, isDataSegmentationMode }) => {

  // Default to 'training_set' for Data Segmentation, otherwise null
  const [mode, setMode] = useState<'training_set' | 'validation_set' | 'testing_set' | null>(
    isDataSegmentationMode ? 'training_set' : null
  );

  const handleModeChange = (event: React.MouseEvent<HTMLElement>, newMode: 'training_set' | 'validation_set' | 'testing_set' | null) => {
      if (newMode !== null) {
          setMode(newMode);
      }
  };
  
  // --- FILTERING LOGIC ---
  // useMemo will prevent re-filtering on every render unless `results` or `mode` changes.
  const filteredResults = useMemo(() => {
    // If not in data segmentation mode, show all results
    if (!isDataSegmentationMode || !mode) {
      return results;
    }

    switch (mode) {
      case 'training_set':
        // Show results that do NOT have the [Validation] or [Testing] suffix
        return results.filter(r => !r.name.includes('[Validation]') && !r.name.includes('[Testing]'));
      case 'validation_set':
        // Show only results with the [Validation] suffix
        return results.filter(r => r.name.includes('[Validation]'));
      case 'testing_set':
        // Show only results with the [Testing] suffix
        return results.filter(r => r.name.includes('[Testing]'));
      default:
        return results;
    }
  }, [results, mode, isDataSegmentationMode]);

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        height: '100%', 
        p: 2, 
        borderRight: 1, 
        borderColor: 'divider',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
      }}>
        <Tooltip title="Compare equity curves of all strategies">
          <Button variant="contained" color="primary" startIcon={<RefreshCcw />} sx={{ mb: 2, width:'100%' }} onClick={onCompareClick}>
            Compare Strategies
          </Button>
        </Tooltip>
      </Box>
      
      {/* --- CONDITIONAL RENDERING FOR TOGGLE BUTTONS --- */}
      {isDataSegmentationMode && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
            <ToggleButtonGroup
              color="primary"
              value={mode}
              exclusive
              onChange={handleModeChange}
              aria-label="Data Segmentation Mode"
              size="small"
            >
              <ToggleButton value="training_set">Training Set</ToggleButton>
              <ToggleButton value="validation_set">Validation Set</ToggleButton>
              <ToggleButton value="testing_set">Testing Set</ToggleButton>
            </ToggleButtonGroup>
        </Box>
      )}

      <Box sx={{ flexGrow: 1, overflowY: 'auto', position: 'relative' }}>
        {/* --- USE THE FILTERED LIST --- */}
        {filteredResults.length > 0 ? (
          <List component="nav" dense>
            {filteredResults.map((result) => (
              <ListItemButton
                key={result.id}
                selected={selectedId === result.id}
                onClick={() => onSelect(result.id)}
                sx={{ borderRadius: 2, mb: 0.5 }}
              >
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <AssessmentIcon fontSize="small" />
                </ListItemIcon>
                {/* --- Clean up the name for display --- */}
                <ListItemText 
                  primary={result.name.replace(' [Validation]', '').replace(' [Testing]', '')} 
                />
              </ListItemButton>
            ))}
          </List>
        ) : (
          // --- Show a message if the list is empty ---
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <Typography variant="body2" color="text.secondary">
              No results in this set.
            </Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
};
