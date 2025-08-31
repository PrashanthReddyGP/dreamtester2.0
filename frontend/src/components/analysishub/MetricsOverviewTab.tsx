// src/components/analysishub/MetricsOverviewTab.tsx
// (This is the modified version of your MetricsGridTab.tsx)

import React, { useMemo } from 'react';
import type { FC } from 'react';
import {
    Box,
    Typography,
    Tooltip,
    useTheme,
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { DataGrid } from '@mui/x-data-grid/DataGrid';
import type { GridColDef, GridRenderCellParams, GridColumnHeaderParams } from '@mui/x-data-grid';

// We now expect the full StrategyResult type
import type { StrategyResult, StrategyMetrics } from '../../services/api';
import { color, format } from 'echarts';

// --- Helper functions remain the same ---
const formatCurrency = (value: number) => {
    const sign = value < 0 ? '-' : '';
    return `${sign}$${Math.abs(value).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
};
const formatPercent = (value: number) => `${(value || 0).toFixed(2)}%`;
const formatRatio = (value: number | string) => {
    // Check if the value is not a valid number
    if (typeof value !== 'number' || !isFinite(value)) {
        // This will correctly handle strings like "inf"
        return value || '0.00';
    }
    return value.toFixed(2);
};

// --- The configuration array is now static, as dynamic properties will be handled in the cell renderer ---
const metricsConfig = [
    { key: 'Net_Profit', label: 'Net Profit', tooltip: 'Total profit or loss after all trades.', format: formatCurrency },
    { key: 'Commission', label: 'Commissions', tooltip: 'The overall commissions deducted on this trade span.', format: formatPercent, color: 'error.main' }, 
    { key: 'Max_Drawdown_Duration_days', label: 'DD Duration', tooltip: 'The longest time it took to recover from a peak (days).', unit: ' days' },
    { key: 'Max_Drawdown', label: 'Max Drawdown', tooltip: 'The largest peak-to-trough decline in portfolio value.', format: formatPercent, color: 'error.main' },
    { key: 'Total_Trades', label: 'Total Trades', tooltip: 'The total number of closed trades executed.' },
    { key: 'Sharpe_Ratio', label: 'Sharpe Ratio', tooltip: 'Measures risk-adjusted return, considering volatility.', format: formatRatio },
    { key: 'Profit_Factor', label: 'Profit Factor', tooltip: 'Gross profits divided by gross losses. Higher is better.', format: formatRatio },
    { key: 'Calmar_Ratio', label: 'Calmar Ratio', tooltip: 'Measures return relative to the maximum drawdown.', format: formatRatio },
    { key: 'Equity_Efficiency_Rate', label: 'EER', tooltip: 'A custom metric for strategy quality.', format: formatRatio },
];

// --- The component now accepts an array of results ---
export const MetricsOverviewTab: FC<{ results: StrategyResult[] }> = ({ results }) => {
    const theme = useTheme();

    const columns: GridColDef[] = useMemo(() => {
        // --- 1. Define the static first column for the Strategy Name ---
        const strategyColumn: GridColDef = {
            field: 'strategyName',
            headerName: 'Strategy',
            minWidth: 180,
            flex: 1.5,
            align: 'center',
            headerAlign: 'center',
            sortable: true,
            renderCell: (params: GridRenderCellParams) => (
                <Typography variant="body1" fontWeight={600}>
                    {params.value as string}
                </Typography>
            ),
        };

        // --- 2. Dynamically create columns for each metric in the config ---
        const metricsColumns: GridColDef[] = metricsConfig
            .map((config): GridColDef => ({
                field: config.key, // The field is the key from the metrics object
                minWidth: 140,
                align: 'right',
                headerAlign: 'right',
                sortable: true,
                flex:1,
                renderHeader: (params: GridColumnHeaderParams) => (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', width: '100%' }}>
                        <Typography variant="body2" fontWeight={600}>
                            {config.label}
                        </Typography>
                        <Tooltip title={config.tooltip} placement="top" arrow>
                            <InfoOutlinedIcon sx={{ fontSize: '1rem', ml: 0.5, color: 'text.disabled', verticalAlign: 'middle' }} />
                        </Tooltip>
                    </Box>
                ),
                renderCell: (params: GridRenderCellParams) => {
                    const rawValue = params.value as number;
                    
                    // Determine color dynamically based on the row's data
                    let cellColor = config.color;
                    if (config.key === 'Net_Profit') {
                        cellColor = rawValue >= 0 ? theme.palette.success.main : theme.palette.error.main;
                    }
                    
                    const formattedValue = config.format ? config.format(rawValue) : rawValue;

                    return (
                        <Typography variant="body1" fontWeight={500} sx={{ color: cellColor }}>
                            {formattedValue}
                        </Typography>
                    );
                }
            }));
            
        return [strategyColumn, ...metricsColumns];

    }, [theme]);

    const rows = useMemo(() => {
        // --- 3. Map the results array to the format the DataGrid expects ---
        return results
          .filter(result => result.strategy_name !== 'Portfolio')
          .map(result => ({
            id: result.strategy_name, // Use strategy name as the unique ID for the row
            strategyName: result.strategy_name,
            ...result.metrics // Spread all metrics into the row object
        }));
    }, [results]);

    if (!results || results.length === 0) {
        return <Box sx={{ p: 4, textAlign: 'center' }}>No metric data available.</Box>;
    }

    return (
        <Box sx={{ height: '100%', width: '100%' }}>
            <DataGrid
                rows={rows}
                columns={columns}
                getRowId={(row) => row.id}
                pageSizeOptions={[15, 25, 50]}
                initialState={{
                    sorting: {
                      sortModel: [{ field: 'Net_Profit', sort: 'desc' }], // Initially sort by Net Profit
                    },
                    pagination: {
                      paginationModel: { page: 0, pageSize: 15 },
                    },
                  }}
                density="compact"
                sx={{
                    backgroundColor: theme.palette.background.paper,
                    border: `1px solid ${theme.palette.divider}`,

                   '& .MuiDataGrid-columnHeader': {
                        backgroundColor: theme.palette.background.paper,
                        borderRight: `1px solid ${theme.palette.divider}}`,
                    },
                    
                    '& .MuiDataGrid-columnHeaders': {
                        borderBottom: `1px solid ${theme.palette.divider}`,
                    },
                    
                    '& .MuiDataGrid-cell': {
                        borderRight: `1px solid ${theme.palette.divider}`,
                        alignContent: 'center'
                    },
                    
                    "& .MuiDataGrid-cell:focus, & .MuiDataGrid-cell:focus-within": {
                        outline: "none !important",
                    },
                }}
            />
        </Box>
    );
};