import React from 'react';
// CHANGE 1: Update the Box and Typography imports, and add the new Grid import
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid'; // Use the direct, unambiguous import
import Typography from '@mui/material/Typography';

// Import the specific ML analysis types and components
import type { ModelAnalysis } from '../../services/api';
import { FeatureImportanceChart } from './FeatureImportanceChart';
import { ClassificationReportTable } from './ClassificationReportTable';
import { ConfusionMatrix } from './ConfusionMatrix';

interface Props {
  analysis: ModelAnalysis;
}

export const ModelAnalysisTab: React.FC<Props> = ({ analysis }) => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* <Typography variant="h5" gutterBottom sx={{ textAlign: 'center' }}>
        Machine Learning Model Performance
      </Typography> */}
      
      <Box sx={{ display: 'flex', flexDirection: 'row', gap: 2, height: '100%' }}>

        <Box sx={{ width: '50%', minHeight: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', p: 2 }}>
          <FeatureImportanceChart data={analysis.feature_importance} />
        </Box>

        <Box sx={{ width: '50%', display: 'flex', flexDirection: 'column', gap: 2, height: '100%', justifyContent: 'space-between' }}>
          <ClassificationReportTable report={analysis.classification_report} />
          <ConfusionMatrix 
            matrix={analysis.confusion_matrix} 
            labels={Object.keys(analysis.classification_report).filter(k => k.startsWith('class_'))}
          />
        </Box>

      </Box>
    </Box>
  );
};