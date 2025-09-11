// src/components/machinelearning/shared/DataDisplays.tsx
import React, {useMemo} from 'react';
import { Box, Typography, Paper, Stack, useTheme, Button, Divider } from '@mui/material';
import { DataGrid, GridFooterContainer, GridPagination } from '@mui/x-data-grid';
import type { GridColDef } from '@mui/x-data-grid';

interface DataInfo {
    Symbol?: string;
    Timeframe?: string;
    "Data Points"?: number;
    "Start Date"?: string;
    "End Date"?: string;
    "Memory Usage"?: string;
    message?: string;
    error?: string;
    
    // Data Quality
    "Total Missing Values"?: number;
    "Missing Values by Column"?: Record<string, number>;

    // Data Structure
    "Data Types"?: Record<string, string>;
    "Descriptive Statistics"?: Record<string, Record<string, number>>;
}

const StatItem: React.FC<{ label: string; value: React.ReactNode }> = ({ label, value }) => (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
        <Typography variant="body2" sx={{ color: '#9c9c9cff' }}>{label}:</Typography>
        <Typography variant="body2" component="span" sx={{ fontWeight: 'bold', ml: 1 }}>{value}</Typography>
    </Box>
);

export const DataInfoDisplay = ({ info }: { info: DataInfo | null }) => (
    <Box sx={{ height: '100%' }}>
        <Typography variant="subtitle1" gutterBottom sx={{ borderBottom: '1px solid', borderColor: 'divider', pb: 2, textAlign: 'center', fontWeight: 'bold' }}>Dataset Information</Typography>

        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', pl: 2, pr: 2 }}>
            {info ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-start', height: '100%' }}>
                    <Stack spacing={1} sx={{ pt: 1 }}>
                        {info.error && <Typography color="error">Error: {info.error}</Typography>}
                        
                        {/* General Info */}
                        <StatItem label="Symbol" value={info.Symbol} />
                        <StatItem label="Timeframe" value={info.Timeframe} />
                        <StatItem label="Data Points" value={info["Data Points"]?.toLocaleString()} />
                        <StatItem label="Start Date" value={info["Start Date"]} />
                        <StatItem label="End Date" value={info["End Date"]} />
                        <StatItem label="Memory Usage" value={info["Memory Usage"]} />
                        
                        <Divider sx={{ pt: 1 }} />

                        <StatItem label="Total Missing Values" value={info["Total Missing Values"]?.toLocaleString()} />
                        {info["Missing Values by Column"] && Object.keys(info["Missing Values by Column"]).length > 0 && (
                            <Box>
                                {Object.entries(info["Missing Values by Column"]).map(([col, count]) => (
                                    <StatItem key={col} label={col} value={count.toLocaleString()} />
                                ))}
                            </Box>
                        )}

                        <Divider sx={{ pt: 0 }} />
                        
                        {/* Data Types Section */}
                        {/* <Typography variant="subtitle1" sx={{ fontWeight: 'bold', textAlign: 'center', color: '#c3c3c3ff' }}>Data Types</Typography> */}
                        {info["Data Types"] && (
                            <Box>
                                {Object.entries(info["Data Types"]).map(([col, dtype]) => (
                                <StatItem key={col} label={col} value={dtype} />
                                ))}
                            </Box>
                        )}

                        {info.message && <Typography variant="body2" color="text.secondary">{info.message}</Typography>}
                    </Stack>
                </Box>
            ) : (
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>No data loaded.</Typography>
            )}
        </Box>
    </Box>
);


export const DataInfoDisplayHorizontal = ({ info }: { info: DataInfo | null }) => (
    <Paper variant="outlined" sx={{ p: 1, height: '100%' }}>

        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', overflow: 'hidden' }}>
            {info ? (
                <Box sx={{ height: '100%' }}>
                    <Stack sx={{ display: 'flex', flexDirection: 'row', gap: 2, height: '100%' }}>
                        
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, flexGrow: 1, height: '100%', p: 1 }}>
                            {info.error && <Typography color="error">Error: {info.error}</Typography>}
                            {/* General Info */}
                            <StatItem label="Symbol" value={info.Symbol} />
                            <StatItem label="Timeframe" value={info.Timeframe} />
                            <StatItem label="Data Points" value={info["Data Points"]?.toLocaleString()} />
                            <StatItem label="Start Date" value={info["Start Date"]} />
                            <StatItem label="End Date" value={info["End Date"]} />
                            <StatItem label="Memory Usage" value={info["Memory Usage"]} />
                            
                            <StatItem label="Total Missing Values" value={info["Total Missing Values"]?.toLocaleString()} />
                            {info["Missing Values by Column"] && Object.keys(info["Missing Values by Column"]).length > 0 && (
                                <Box pl={2}>
                                    {Object.entries(info["Missing Values by Column"]).map(([col, count]) => (
                                        <StatItem key={col} label={col} value={count.toLocaleString()} />
                                    ))}
                                </Box>
                            )}
                        </Box>

                        <Divider orientation="vertical" flexItem />
                        
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, flexGrow: 1, height: '100%', p: 1 }}>
                            {info.message && <Typography variant="body2" color="text.secondary">{info.message}</Typography>}
                            {/* Data Types Section */}
                            <Typography variant="subtitle1" sx={{ fontWeight: 'bold', textAlign: 'center', color: '#c3c3c3ff' }}>Data Types</Typography>
                            {info["Data Types"] && (
                                <Box sx={{ overflowY: 'auto' }}>
                                    {Object.entries(info["Data Types"]).map(([col, dtype]) => (
                                    <StatItem key={col} label={col} value={dtype} />
                                    ))}
                                </Box>
                            )}
                        </Box>

                    </Stack>
                </Box>
            ) : (
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>No data loaded.</Typography>
            )}
        </Box>
    </Paper>
);


// Adapted to handle MILLISECONDS from the OHLCV data
const formatTimestamp = (timestampInMillis: number): string => {
    if (!timestampInMillis) return '';
    const date = new Date(timestampInMillis);
    return date.toLocaleString('en-US', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    });
};

const formatPrice = (price: any): string => {
    const numericPrice = parseFloat(price);
    if (isNaN(numericPrice)) return 'N/A';
    // Use toFixed for consistent decimal places, adjust as needed for your assets
    return numericPrice.toFixed(5);
};

const formatIndicatorValue = (value: any): string => {
    const numericValue = parseFloat(value);
    if (isNaN(numericValue)) return ''; // Return empty string for non-numbers
    // Round to 2 decimal places for better readability
    return numericValue.toFixed(5);
};

const formatHeaderName = (field: string): string => {
    // Replaces underscores with spaces and capitalizes each word
    return field
        .replace(/_/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase());
};

function exportToCsv(rows: any[], columns: GridColDef[], filename = 'data_export.csv') {
    if (!rows || rows.length === 0) return;

    const headers = columns.map((col) => col.headerName || col.field);
    const csvRows = [
        headers.join(','),
        ...rows.map((row) =>
            columns.map((col) => {
                let value = row[col.field];
                // Use the formatter if it exists to get the display value
                if (col.valueFormatter) {
                    value = col.valueFormatter(row[col.field]);
                }
                return `"${value ?? ''}"`;
            }).join(',')
        ),
    ];

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// --- Custom Footer Component ---
function CustomFooter({ rows, columns }: { rows: any[]; columns: GridColDef[] }) {
    return (
        <GridFooterContainer sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button
                variant="contained"
                size="small"
                sx={{ mr: 2, ml: 2 }}
                onClick={() => exportToCsv(rows, columns)}
            >
                Export CSV
            </Button>
            <GridPagination />
        </GridFooterContainer>
    );
}

    
export const DataGridDisplay = ({ data, info, title }: { data: any[], info: DataInfo | null, title: string }) => {
    
    const theme = useTheme();

    if (!data || data.length === 0) {
        return (
            <Paper variant="outlined" sx={{ p: 2, height: '100%', overflow: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography color="text.secondary">
                    {title.includes("Raw") ? "Click 'Fetch Data' to see the preview." : "Process data to see the preview."}
                </Typography>
            </Paper>
        );
    }

    const columns: GridColDef[] = useMemo(() => {
        if (!data || data.length === 0) {
            return [];
        }

        const allKeys = Object.keys(data[0]);
        // Get the data types from our reliable info object
        const dataTypes = info?.["Data Types"] || {};

        return allKeys.map((key): GridColDef => {
            const baseColumn: Omit<GridColDef, 'field'> = {
                headerName: formatHeaderName(key),
                minWidth: 150,
                flex: 1,
                sortable: true,
            };
            
            // Check the type from the info object, not the data itself.
            const columnType = dataTypes[key] || '';
            const isNumeric = columnType.includes('float') || columnType.includes('int');

            // Apply special formatting for known column names first
            switch (key) {
                case 'timestamp':
                    return {
                        ...baseColumn,
                        field: key,
                        headerName: 'Timestamp',
                        minWidth: 180,
                        valueFormatter: (value) => formatTimestamp(value as number),
                    };
                case 'open':
                case 'high':
                case 'low':
                case 'close':
                    return {
                        ...baseColumn,
                        field: key,
                        type: 'number',
                        align: 'right',
                        headerAlign: 'right',
                        valueFormatter: (value) => formatPrice(value),
                    };
                case 'volume':
                    return {
                        ...baseColumn,
                        field: key,
                        type: 'number',
                        align: 'right',
                        headerAlign: 'right',
                        valueFormatter: (value) => (value as number)?.toLocaleString(),
                    };
                
                // For all other columns, use our robust type check
                default:
                    if (isNumeric) {
                        return {
                            ...baseColumn,
                            field: key,
                            align: 'right',
                            headerAlign: 'right',
                            // Use renderCell for guaranteed formatting
                            renderCell: (params) => formatIndicatorValue(params.value),
                        };
                    }
                    // Fallback for any non-numeric columns
                    return {
                        ...baseColumn,
                        field: key,
                    };
            }
        });
    }, [data, info]);

    return (
        <Paper variant="elevation" sx={{ p:0, height: '100%', overflow: 'auto' }}>

            {/* <Typography variant="h6" gutterBottom>{title}</Typography> */}

            {data && data.length > 0 ? (
                <Box sx={{ 
                    flexGrow: 1, 
                    height: '100%',
                    '& ::-webkit-scrollbar': {
                        width: '4px',
                        height: '4px',
                    },
                    '& ::-webkit-scrollbar-track': {
                        backgroundColor: 'transparent',
                    },
                    '& ::-webkit-scrollbar-thumb': {
                        backgroundColor: 'transparent', // <-- Hide scrollbar thumb by default
                        borderRadius: '4px',
                        transition: 'background-color 0.2s ease-in-out', // Add a smooth transition
                    },
                    // --- ON HOVER of this Box, make the scrollbar thumb visible ---
                    '&:hover ::-webkit-scrollbar-thumb': {
                        backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.25)' : 'rgba(0, 0, 0, 0.25)',
                    },
                    '& ::-webkit-scrollbar-thumb:hover': {
                        backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.4)' : 'rgba(0, 0, 0, 0.4)', // Make it darker when hovering the thumb itself
                    },

                    }}>
                    <DataGrid
                        rows={data}
                        columns={columns}
                        getRowId={(row) => row.timestamp} // Timestamps are unique and reliable IDs
                        initialState={{
                            pagination: { paginationModel: { pageSize: 5 } },
                            sorting: { sortModel: [{ field: 'timestamp', sort: 'desc' }] },
                        }}
                        pageSizeOptions={[5, 20, 50]}
                        density="compact"
                        slots={{
                            footer: () => <CustomFooter rows={data} columns={columns} />,
                        }}
                        sx={{
                            backgroundColor: theme.palette.background.paper,
                            border: `1px solid ${theme.palette.divider}`,
                            '& .MuiDataGrid-columnHeaders': {
                                '--DataGrid-t-header-background-base': theme.palette.background.paper,
                                borderBottom: `1px solid ${theme.palette.divider}`,
                            },
                            '& .MuiDataGrid-columnSeparator': {
                                visibility: 'visible', // Ensure it's visible
                                color: theme.palette.divider, // Use the theme's divider color
                            },
                            '& .MuiDataGrid-cell': {
                                borderBottom: `1px solid ${theme.palette.divider}`,
                                borderRight: `1px solid ${theme.palette.divider}`,
                                borderLeft: `1px solid ${theme.palette.divider}`,
                            },
                            pl: 2,
                            pr: 2
                        }}
                    />
                </Box>
            ) : (
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                    <Typography color="text.secondary">
                        {title === "Raw Data" ? "Click 'Fetch Data' to see the preview." : "Calculate features to see the preview."}
                    </Typography>
                </Box>
            )}
            
        </Paper>
    );
};