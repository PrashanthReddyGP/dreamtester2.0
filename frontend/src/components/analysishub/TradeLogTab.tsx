import React from 'react';
import { Box } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid/DataGrid';
import type { GridColDef } from '@mui/x-data-grid';

const columns: GridColDef[] = [
  { field: 'id', headerName: 'ID', width: 70 },
  { field: 'direction', headerName: 'Direction', width: 130 },
  { field: 'entryTime', headerName: 'Entry Time', width: 200, type: 'dateTime' },
  { field: 'exitTime', headerName: 'Exit Time', width: 200, type: 'dateTime' },
  { field: 'entryPrice', headerName: 'Entry Price', type: 'number', width: 130 },
  { field: 'exitPrice', headerName: 'Exit Price', type: 'number', width: 130 },
  { field: 'pnl', headerName: 'P/L ($)', type: 'number', width: 130 },
  { field: 'pnlPercent', headerName: 'P/L (%)', type: 'number', width: 130 },
];

const rows = [
  { id: 1, direction: 'Long', entryTime: new Date(2023, 1, 5, 10, 30), exitTime: new Date(2023, 1, 5, 14, 0), entryPrice: 23000, exitPrice: 23500, pnl: 500, pnlPercent: 2.17 },
  { id: 2, direction: 'Short', entryTime: new Date(2023, 1, 8, 9, 0), exitTime: new Date(2023, 1, 8, 12, 0), entryPrice: 24000, exitPrice: 23800, pnl: 200, pnlPercent: 0.83 },
  { id: 3, direction: 'Long', entryTime: new Date(2023, 1, 10, 11, 0), exitTime: new Date(2023, 1, 11, 15, 30), entryPrice: 22500, exitPrice: 22300, pnl: -200, pnlPercent: -0.89 },
];

export const TradeLogTab: React.FC = () => {
  return (
    <Box>
      <Box sx={{ height: '100%', width: '100%' }}>
        <DataGrid
          rows={rows}
          columns={columns}
          initialState={{
            pagination: {
              paginationModel: { page: 0, pageSize: 10 },
            },
          }}
          pageSizeOptions={[5, 10, 20]}
          checkboxSelection
          sx={{ 
            border: 'none',
            '& .MuiDataGrid-cell': { borderBottomColor: 'divider' },
            '& .MuiDataGrid-columnHeaders': { borderBottomColor: 'divider' },
          }}
        />
      </Box>
    </Box>
  );
};