import React from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography } from '@mui/material';
import type { ClassificationReport } from '../../../services/api';

interface Props {
  report: ClassificationReport;
}

export const ClassificationReportTable: React.FC<Props> = ({ report }) => {
  // Extract class keys (e.g., 'class_-1', 'class_0') and summary keys
  const classKeys = Object.keys(report).filter(key => key.startsWith('class_'));
  const summaryKeys = Object.keys(report).filter(key => !key.startsWith('class_'));

  const formatLabel = (key: string) => {
      if (key === 'class_-1') return 'Short (-1)';
      if (key === 'class_0') return 'Hold (0)';
      if (key === 'class_1') return 'Long (1)';
      return key.replace(/_/g, ' '); // Format summary keys like "macro avg"
  }
  
  return (
    <TableContainer component={Paper}>
      <Typography variant="h6" sx={{ p: 2, textAlign: 'center' }}>Classification Report</Typography>
      <Table size="medium">
        <TableHead>
          <TableRow>
            <TableCell>Class</TableCell>
            <TableCell align="right">Precision</TableCell>
            <TableCell align="right">Recall</TableCell>
            <TableCell align="right">F1-Score</TableCell>
            <TableCell align="right">Support</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {classKeys.map(key => (
            <TableRow key={key}>
              <TableCell component="th" scope="row">{formatLabel(key)}</TableCell>
              <TableCell align="right">{(report[key] as any).precision.toFixed(2)}</TableCell>
              <TableCell align="right">{(report[key] as any).recall.toFixed(2)}</TableCell>
              <TableCell align="right">{(report[key] as any)['f1-score'].toFixed(2)}</TableCell>
              <TableCell align="right">{(report[key] as any).support}</TableCell>
            </TableRow>
          ))}
          <TableRow sx={{ '& td, & th': { border: 0, fontWeight: 'bold' } }}>
            <TableCell colSpan={5} />
          </TableRow>
          {summaryKeys.map(key => (
            <TableRow key={key}>
              <TableCell component="th" scope="row">{formatLabel(key)}</TableCell>
              <TableCell align="right">{typeof report[key] === 'number' ? (report[key] as number).toFixed(2) : (report[key] as any).precision.toFixed(2)}</TableCell>
              <TableCell align="right">{typeof report[key] !== 'number' && (report[key] as any).recall.toFixed(2)}</TableCell>
              <TableCell align="right">{typeof report[key] !== 'number' && (report[key] as any)['f1-score'].toFixed(2)}</TableCell>
              <TableCell align="right">{typeof report[key] !== 'number' && (report[key] as any).support}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};