import React from 'react';
import { Paper, Typography, Box, useTheme } from '@mui/material';

interface KpiCardProps {
  title: string;
  data: {
    value: number;
    unit: string;
  };
}

export const KpiCard: React.FC<KpiCardProps> = ({ title, data }) => {
  const theme = useTheme();
  // Determine color based on value (profit vs loss)
  const valueColor = data.value >= 0 ? theme.palette.success.main : theme.palette.error.main;

  return (
    <Paper elevation={0} sx={{ p: 2, textAlign: 'center' }}>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1 }}>
        {title}
      </Typography>
      <Typography variant="h4" sx={{ color: valueColor, fontWeight: 600 }}>
        {data.value.toLocaleString()}
        <span style={{ fontSize: '0.7em', marginLeft: '2px' }}>{data.unit}</span>
      </Typography>
    </Paper>
  );
};