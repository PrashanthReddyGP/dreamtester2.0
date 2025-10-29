// src/components/pipeline/analysis/CorrelationDisplay.tsx

import React from 'react';
import { Box, Typography, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { getColorForCorrelation } from '../../pipeline/colorUtils'

interface CorrelationData {
    columns: string[];
    index: string[];
    data: number[][];
}

interface CorrelationDisplayProps {
    correlationData: CorrelationData;
    displayMode: 'matrix' | 'table';
    method: string;
}

// Sub-component for rendering the heatmap matrix
const CorrelationMatrix: React.FC<{ data: CorrelationData }> = ({ data }) => {
    return (
        <Paper elevation={2} sx={{ mt: 2, overflow: 'auto', p: 1, maxWidth: '100%' }}>
            <Box sx={{ display: 'grid', gridTemplateColumns: `auto repeat(${data.columns.length}, minmax(50px, 1fr))`, gap: '1px' }}>
                {/* Top-left empty cell */}
                <Box />
                {/* Column Headers */}
                {data.columns.map(col => (
                    <Typography key={col} variant="caption" sx={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', textAlign: 'left', fontWeight: 'bold', p: 0.5 }}>{col}</Typography>
                ))}
                
                {/* Rows */}
                {data.index.map((rowName, rowIndex) => (
                    <React.Fragment key={rowName}>
                        {/* Row Header */}
                        <Typography variant="caption" sx={{ textAlign: 'right', fontWeight: 'bold', pr: 1, display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>{rowName}</Typography>
                        {/* Cells */}
                        {data.data[rowIndex].map((value, colIndex) => (
                            <Box
                                key={`${rowIndex}-${colIndex}`}
                                title={`${rowName} / ${data.columns[colIndex]}: ${value.toFixed(4)}`}
                                sx={{
                                    bgcolor: getColorForCorrelation(value),
                                    aspectRatio: '1 / 1',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    color: Math.abs(value) > 0.6 ? 'white' : 'black',
                                    fontSize: '0.65rem',
                                    fontWeight: 'medium',
                                    borderRadius: '2px'
                                }}
                            >
                                {value.toFixed(2)}
                            </Box>
                        ))}
                    </React.Fragment>
                ))}
            </Box>
        </Paper>
    );
};

// Sub-component for rendering the data table
const CorrelationTable: React.FC<{ data: CorrelationData }> = ({ data }) => {
    return (
        <TableContainer component={Paper} elevation={2} sx={{ mt: 2, maxHeight: 'calc(100% - 80px)'}}>
            <Table stickyHeader size="small">
                <TableHead>
                    <TableRow>
                        <TableCell sx={{ fontWeight: 'bold' }}>Feature</TableCell>
                        {data.columns.map(col => <TableCell key={col} sx={{ fontWeight: 'bold' }}>{col}</TableCell>)}
                    </TableRow>
                </TableHead>
                <TableBody>
                    {data.index.map((rowName, rowIndex) => (
                        <TableRow key={rowName}>
                            <TableCell sx={{ fontWeight: 'bold' }}>{rowName}</TableCell>
                            {data.data[rowIndex].map((value, colIndex) => (
                                <TableCell key={`${rowIndex}-${colIndex}`} align="right">{value.toFixed(4)}</TableCell>
                            ))}
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    )
}

export const CorrelationDisplay: React.FC<CorrelationDisplayProps> = ({ correlationData, displayMode, method }) => {
    if (!correlationData) {
        return <Typography sx={{ mt: 3, fontStyle: 'italic', textAlign: 'center' }}>No correlation data available. Run the node to compute.</Typography>;
    }

    return (
        <Box sx={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" sx={{ textAlign: 'center' }}>Feature Correlation</Typography>

            {displayMode === 'matrix' 
                ? <CorrelationMatrix data={correlationData} /> 
                : <CorrelationTable data={correlationData} />
            }
        </Box>
    );
};