import React from 'react';
import type { ClassificationReport } from '../../services/api'; // Import the type

interface Props {
  report: ClassificationReport;
  matrix: number[][];
}

export const ModelAnalysisDetails: React.FC<Props> = ({ report, matrix }) => {
  // Logic to render the classification report and confusion matrix in tables
  return (
    <div>
      <h3>Classification Report</h3>
      <pre>{JSON.stringify(report, null, 2)}</pre>
      
      <h3>Confusion Matrix</h3>
      <pre>{JSON.stringify(matrix, null, 2)}</pre>
    </div>
  );
};