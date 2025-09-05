import React from 'react';
import type { FeatureImportance } from '../../services/api'; // Import the type
// You'll need a charting library like Chart.js, Recharts, or ECharts

interface Props {
  data: FeatureImportance[];
}

export const FeatureImportanceChart: React.FC<Props> = ({ data }) => {
  // Logic to render a bar chart using the data
  // For example, data.map(item => item.feature) for labels
  // and data.map(item => item.importance) for values.
  return (
    <div>
      <h3>Feature Importances</h3>
      {/* Your chart component would go here */}
      <ul>
        {data.map(item => (
          <li key={item.feature}>{item.feature}: {item.importance.toFixed(3)}</li>
        ))}
      </ul>
    </div>
  );
};