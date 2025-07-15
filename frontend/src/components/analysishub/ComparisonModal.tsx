import React, { useMemo, useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  useTheme,
  IconButton,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Checkbox,
  Alert
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ReactECharts from 'echarts-for-react';
import type { StrategyResult } from '../../services/api';

interface ComparisonModalProps {
  open: boolean;
  onClose: () => void;
  results: StrategyResult[];
  initialCapital: number;
}

// NEW: A modern, vibrant, and distinct color palette for your charts.
const CHART_COLORS = [
  '#5470C6', '#91CC75', '#FAC858', '#EE6666',
  '#73C0DE', '#3BA272', '#FC8452', '#9A60B4', '#EA7CCC'
];

const MAX_COMPARISON_COUNT = 10;

export const ComparisonModal: React.FC<ComparisonModalProps> = ({
  open,
  onClose,
  results,
  initialCapital,
}) => {
  const theme = useTheme();

  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  useEffect(() => {
    if (open) {
      setSelectedIds([]);
    }
  }, [open]);

  const chartOption = useMemo(() => {
    const strategiesToCompare = results.filter(
      (r) => selectedIds.includes(r.strategy_name)
    );

    const seriesData = strategiesToCompare.map((strategy, index) => ({
      name: strategy.strategy_name,
      type: 'line',
      showSymbol: false,
      // NEW: A slightly higher smoothing factor for a less jagged look.
      smooth: 0.2, 
      data: strategy.equity_curve.map((point) => [
        point[0] * 1000,
        point[1] - initialCapital,
      ]),
      // NEW: Enhance interactivity on hover. This highlights the hovered series.
      emphasis: {
        focus: 'series', // This is the key property
        lineStyle: {
          width: 3, // Make the hovered line thicker
        },
      },
      // NEW: Add a semi-transparent area fill for a modern aesthetic.
      areaStyle: {
        opacity: 0.15,
        // The color will be inherited from the line color by default
      },
    }));

    const legendData = strategiesToCompare.map((s) => s.strategy_name);

    return {
      color: CHART_COLORS,
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        formatter: (params: any[]) => {
            const date = new Date(params[0].axisValue);
            const dateString = date.toLocaleDateString();
            let tooltipHtml = `${dateString}<br/>`;

            params.sort((a, b) => b.value[1] - a.value[1]); // Sort tooltip values descending

            params.forEach((param: any) => {
                const seriesName = param.seriesName;
                const value = param.value[1];
                const color = param.color;
                const marker = `<span style="display:inline-block;margin-right:5px;border-radius:10px;width:10px;height:10px;background-color:${color};"></span>`;
                tooltipHtml += `${marker} ${seriesName}: <b>$${value.toFixed(2)}</b><br/>`;
            });

            return tooltipHtml;
        }
      },
      legend: {
        data: legendData,
        textStyle: { color: theme.palette.text.primary },
        type: 'scroll',
        orient: 'horizontal',
        top: 'top',     
        left: 'center',    
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'time',
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        name: 'Net Profit ($)',
        scale: true,
        axisLabel: { formatter: '${value}' },
        splitLine: {
            show: true,
            lineStyle: {
                color: theme.palette.grey[700],
                type: 'dotted',
                opacity: 0.2,
            }
        },
        axisLine: {
            show: true,
        }
      },
      dataZoom: [
        { type: 'inside', xAxisIndex: [0] },
        { type: 'slider', xAxisIndex: [0], textStyle: { color: theme.palette.text.secondary } },
      ],
      series: seriesData,
      backgroundColor: 'transparent',
      textStyle: { color: theme.palette.text.secondary },
    };
  }, [selectedIds, results, initialCapital, theme]);

  const handleToggle = (value: string) => () => {
    const currentIndex = selectedIds.indexOf(value);
    const newChecked = [...selectedIds];

    if (currentIndex === -1) {
      // Add to the list if not already there, respecting the max limit
      if (selectedIds.length < MAX_COMPARISON_COUNT) {
        newChecked.push(value);
      }
    } else {
      // Remove from the list if already checked
      newChecked.splice(currentIndex, 1);
    }

    setSelectedIds(newChecked);
  };

  const availableChoices = results.filter(r => r.strategy_name !== 'Portfolio');

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="xl" PaperProps={{ sx: { height: '90vh' } }}>
      <DialogTitle sx={{ m: 0, p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        Select Strategies to Compare
        <IconButton aria-label="close" onClick={onClose}><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent dividers sx={{ p: 0, display: 'flex', overflow: 'hidden' }}>
        
        {/* --- LEFT PANEL: SELECTION LIST --- */}
        <Box sx={{ width: '300px', borderRight: 1, borderColor: 'divider', overflowY: 'auto' }}>
            <Box sx={{p: 2}}>
                <Typography variant="h6">Available Results</Typography>
                <Typography variant="caption" color="text.secondary">
                    Select up to {MAX_COMPARISON_COUNT} to compare.
                </Typography>
            </Box>
            <List dense>
              {availableChoices.map((result) => {
                const labelId = `checkbox-list-label-${result.strategy_name}`;
                const isSelected = selectedIds.indexOf(result.strategy_name) !== -1;
                const isDisabled = !isSelected && selectedIds.length >= MAX_COMPARISON_COUNT;

                return (
                  <ListItem key={result.strategy_name} disablePadding>
                    <ListItemButton role={undefined} onClick={handleToggle(result.strategy_name)} dense disabled={isDisabled}>
                      <ListItemIcon>
                        <Checkbox
                          edge="start"
                          checked={isSelected}
                          tabIndex={-1}
                          disableRipple
                          inputProps={{ 'aria-labelledby': labelId }}
                          disabled={isDisabled}
                        />
                      </ListItemIcon>
                      <ListItemText id={labelId} primary={result.strategy_name} primaryTypographyProps={{sx: {textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap'}}}/>
                    </ListItemButton>
                  </ListItem>
                );
              })}
            </List>
        </Box>

        {/* --- RIGHT PANEL: THE CHART --- */}
        <Box sx={{ flexGrow: 1, height: '100%', width: 'calc(100% - 300px)' }}>
          {selectedIds.length > 0 ? (
            <ReactECharts
              option={chartOption}
              style={{ width: '100%', height: '100%' }}
              notMerge={true}
              lazyUpdate={true}
            />
          ) : (
             <Box sx={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%'}}>
                <Alert severity="info">Select one or more strategies from the left panel to display their equity curves.</Alert>
             </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};