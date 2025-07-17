import React, {useMemo} from 'react';
import type { FC } from 'react';
import { Box, useTheme } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid/DataGrid';
import type { GridColDef } from '@mui/x-data-grid';

import type {Trades} from '../../services/api'

// --- Helper function to format a Unix timestamp in SECONDS ---
const formatTimestamp = (timestampInSeconds: number): string => {
  // If the timestamp is 0, null, or otherwise invalid, return an empty string
  if (!timestampInSeconds) {
    return '';
  }
  // Multiply by 1000 to convert to milliseconds for the JS Date object
  const date = new Date(timestampInSeconds * 1000);
  
  // Use toLocaleString for a nice, locale-aware format.
  // You can customize the options as needed.
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
};

const formatPrice = (price: any): string => {
  const numericPrice = parseFloat(price);
  if (isNaN(numericPrice)) return 'N/A';
  return numericPrice.toPrecision(4);
};

export const TradeLogTab: FC<{ trades: Trades }> = ({ trades }) => {
  
  const theme = useTheme();

  const columns = useMemo(() => {
    if (!trades || trades.length === 0) {
      return [];
    }

    // Get all the unique column names from the first row of data
    const firstRow = trades[0];
    const columnKeys = Object.keys(firstRow);
    
    // Define some columns we always want to show first and with specific properties
    const fixedColumns: GridColDef[] = [
        { 
            field: 'timestamp', 
            headerName: 'Timestamp', 
            width: 200, 
            valueFormatter: (value) => {
                // The `value` here is the raw data for the cell (e.g., 1594755000)
                return formatTimestamp(value as number);
            }
        },
        { field: 'Symbol', headerName: 'Symbol', width: 120 },
        { field: 'Timeframe', headerName: 'Timeframe', width: 80 },
        {field: 'entry', headerName: 'Entry', width: 100, type: 'number', valueFormatter: (value) => {return formatPrice(value as number)}},
        {field: 'take_profit', headerName: 'Take Profit', width: 100, type:'number', valueFormatter: (value) => {return formatPrice(value as number)} },
        {field: 'stop_loss', headerName: 'Stop Loss', width: 100, type:'number', valueFormatter: (value) => {return formatPrice(value as number)} },
        { 
            field: 'Exit_Time', 
            headerName: 'Exit Time', 
            width: 200, 
            valueFormatter: (value) => {
                // The `value` here is the raw data for the cell (e.g., 1594755000)
                return formatTimestamp(value as number);
            }
        },
        { field: 'Result', headerName: 'Result', width: 80, type:'number' },
      ];
    
    const fixedColumnFields = new Set(fixedColumns.map(c => c.field));

    // Dynamically generate columns for the rest of the keys
    const dynamicColumns: GridColDef[] = columnKeys
      .filter(key => !fixedColumnFields.has(key))
      .map(key => {
          const colDef: GridColDef = {
            field: key,
            headerName: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            width: 150,
            type: typeof firstRow[key] === 'number' ? 'number' : 'string',
          };

          // You could also apply the formatter to dynamic columns if they match
          if (key.toLowerCase().includes('time')) {
              // --- THE FIX (also applied here) ---
              colDef.valueFormatter = (value) => {
                return formatTimestamp(value as number);
              };
          }
          
          return colDef;
      });

    return [...fixedColumns, ...dynamicColumns];

  }, [trades]);
  
  if (!trades || trades.length === 0) {
    return <Box sx={{ p: 4 }}>No trades were executed in this backtest.</Box>;
  }

  const sortedTrades = useMemo(() => {
    if (!trades) {
        return [];
    }
    // Create a shallow copy to avoid mutating the original prop array
    // and sort it in descending order based on the timestamp.
    return [...trades].sort((a, b) => b.timestamp - a.timestamp);
  }, [trades]); // This sorting will only re-run when the trades prop changes


  return (
    <Box sx={{ height: '100%', width: '100%' }}>
      <DataGrid
        rows={sortedTrades}
        columns={columns}
        getRowId={(row) => row.timestamp + Math.random()}
        initialState={{
          pagination: {
            paginationModel: { page: 0, pageSize: 20 },
          },
        }}
        pageSizeOptions={[5, 10, 20]}
        checkboxSelection
        sx={{
          // --- 1. Set the background for the BODY (rows) ---
          // This will be the default background for the whole component.
          backgroundColor: theme.palette.background.paper,
          border: `1px solid  ${theme.palette.divider}`,

          // --- 2. Target the HEADER container to give it a separate background ---
          '& .MuiDataGrid-columnHeaders': {
            '--DataGrid-t-header-background-base': theme.palette.background.paper,
            borderBottom: `1px solid ${theme.palette.divider}`,
          },

          '& .MuiDataGrid-columnSeparator': {
            visibility: 'visible', // Ensure it's visible
            color: theme.palette.divider, // Use the theme's divider color
          },

          '& .MuiDataGrid-columnHeader': {
            borderRight: `1px solid ${theme.palette.divider}}`,
          },

          // --- 3. (Optional but good practice) Ensure text colors have good contrast ---
          '& .MuiDataGrid-columnHeaderTitle': {
            fontWeight: 600,
          },
          
          '& .MuiDataGrid-cell': {
            borderBottom: `1px solid ${theme.palette.divider}`,
            borderRight: `1px solid ${theme.palette.divider}`,
          },
          // '& .MuiDataGrid-footerContainer': {
          //   // '--DataGrid-t-color-border-base': `1px solid ${theme.palette.divider}`,
          //    borderTop: `1px solid ${theme.palette.divider}`,
          // },
        }}
      />
    </Box>
  );
};