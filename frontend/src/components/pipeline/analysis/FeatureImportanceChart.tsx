import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Box, Typography } from '@mui/material';
import type { FeatureImportance } from '../../../services/api';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface Props {
  data: FeatureImportance[];
}

// The desired minimum height for each bar in pixels
const MIN_BAR_HEIGHT_PX = 30;
// Extra vertical space for padding, axis labels, etc.
const CHART_PADDING_PX = 100; 

export const FeatureImportanceChart: React.FC<Props> = ({ data }) => {
  const sortedData = [...data].sort((a, b) => a.importance - b.importance);

  // Calculate the required container height
  const chartContainerHeight = 
    sortedData.length * MIN_BAR_HEIGHT_PX + CHART_PADDING_PX;

  const chartData = {
    labels: sortedData.map(item => item.feature),
    datasets: [
      {
        label: 'Importance',
        data: sortedData.map(item => item.importance),
        backgroundColor: 'rgba(75, 192, 192, 0.7)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    indexAxis: 'y' as const, // This makes the bar chart horizontal
    responsive: true,
    maintainAspectRatio: false, // Helps the chart fill the container's height
    plugins: {
      legend: { display: false },
      title: {
        display: false,
        text: 'Feature Importances',
        color: '#FFFFFF', 
        font: {
          size: 16,
          weight: 'bold' as const,
          family: 'Arial',
        },
      },
      tooltip: {
        titleFont: {
          weight: 'bold' as const,
          family: 'Arial',
        },
        bodyColor: '#FFFFFF',
        titleColor: '#FFFFFF',
      },
    },
    scales: {
        x: {
            title: {
                display: false,
                text: 'Importance Score'
            }
        }
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%' }}>
      <Typography variant="h6" sx={{ textAlign: 'center' }}>Feature Importance</Typography>
        <Box sx={{ height: `${chartContainerHeight}px`, width: '100%' }}>
          <Bar options={options} data={chartData} />
      </Box>
    </Box>
  );
};