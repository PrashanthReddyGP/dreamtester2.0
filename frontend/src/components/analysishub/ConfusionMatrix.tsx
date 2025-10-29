import React from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography, Box } from '@mui/material';

interface Props {
  matrix: number[][];
  labels: string[]; // e.g., ['class_-1', 'class_0', 'class_1']
}

const formatLabel = (key: string) => {
    if (key === 'class_-1') return 'Short';
    if (key === 'class_0') return 'Hold';
    if (key === 'class_1') return 'Long';
    return key;
}

export const ConfusionMatrix: React.FC<Props> = ({ matrix, labels }) => {
  
    if (!matrix || matrix.length === 0 || matrix.length !== labels.length) {
    return <Typography sx={{ p: 2, color: 'error.main' }}>Confusion Matrix data is inconsistent.</Typography>;
  }

  return (
    <TableContainer component={Paper} sx={{ mb: 2 }}>
        <Typography variant="h6" sx={{ p: 2, textAlign: 'center' }}>Confusion Matrix</Typography>
        <Table sx={{ minWidth: 450 }} aria-label="confusion matrix">
            <TableHead>
                <TableRow>
                    <TableCell>Actual \ Predicted</TableCell>
                    {labels.map(label => <TableCell key={label} align="center">{formatLabel(label)}</TableCell>)}
                </TableRow>
            </TableHead>
            <TableBody>
                {matrix.map((row, i) => (
                    <TableRow key={i}>
                        <TableCell component="th" scope="row">{formatLabel(labels[i])}</TableCell>
                        {row.map((cell, j) => (
                            <TableCell key={j} align="center" sx={{ backgroundColor: i === j ? 'rgba(79, 79, 79, 0.2)' : 'inherit' }}>
                                {cell}
                            </TableCell>
                        ))}
                    </TableRow>
                ))}
            </TableBody>
        </Table>
    </TableContainer>
  );
};