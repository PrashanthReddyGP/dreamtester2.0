// src/components/pipeline/ModelAnalysisDisplay.tsx

import React, { useState } from 'react';
import { Box, Typography, Paper, Tooltip, IconButton, Snackbar, Alert } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { FeatureImportanceChart } from '../pipeline/analysis/FeatureImportanceChart';
import { ClassificationReportTable } from '../pipeline/analysis/ClassificationReportTable';
import { ConfusionMatrix } from '../pipeline/analysis/ConfusionMatrix';

// Define the shape of the analysis data we expect from the backend
interface ModelAnalysisData {
    feature_importance?: { feature: string; importance: number }[];
    classification_report?: any;
    confusion_matrix?: { labels: string[]; values: number[][] };
}

interface ModelAnalysisDisplayProps {
    analysisData?: ModelAnalysisData;
    panelPosition: 'left' | 'right' | 'top' | 'bottom';
}

export const ModelAnalysisDisplay: React.FC<ModelAnalysisDisplayProps> = ({ analysisData, panelPosition }) => {
    const [snackbarOpen, setSnackbarOpen] = useState(false);

    const handleCopy = () => {
        if (!analysisData) return;
        
        // Format the data as a pretty-printed JSON string
        const dataToCopy = JSON.stringify(analysisData, null, 2);
        
        navigator.clipboard.writeText(dataToCopy)
            .then(() => {
                // On success, show the snackbar
                setSnackbarOpen(true);
            })
            .catch(err => {
                // Log error to the console if copying fails
                console.error('Failed to copy analysis data: ', err);
            });
    };

    const handleSnackbarClose = (event?: React.SyntheticEvent | Event, reason?: string) => {
        if (reason === 'clickaway') {
            return;
        }
        setSnackbarOpen(false);
    };

    if (!analysisData || Object.keys(analysisData).length === 0) {
        return (
            <Paper sx={{ p: 2, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography color="text.secondary">
                    Run the model node to see the analysis.
                </Typography>
            </Paper>
        );
    }
    
    const isClassification = !!analysisData.classification_report;

    const isHorizontalLayout = panelPosition === 'top' || panelPosition === 'bottom';
    const flexDirection = isHorizontalLayout ? 'row' : 'column';

    return (
        <Box sx={{ height: '100%', width: '100%', p: 1, position: 'relative' }}>
            <Tooltip title="Copy all metrics as JSON">
                <IconButton
                    onClick={handleCopy}
                    size="small"
                    sx={{
                        position: 'absolute',
                        top: 12, // Adjust for padding
                        right: 20, // Adjust for padding
                        zIndex: 10, // Ensure it's above other elements
                    }}
                >
                    <ContentCopyIcon fontSize="small" />
                </IconButton>
            </Tooltip>

            <Box sx={{ height: '100%', width: '100%', display: 'flex', flexDirection: flexDirection, gap: 2, overflow: 'auto' }}>
                {isClassification && (
                    <Box sx={{ display: 'flex', flexDirection: flexDirection, gap: 2, flexGrow: 1 }}>
                        <Paper variant="outlined" sx={{ p: 2, flex: 1 }}>
                            <ClassificationReportTable report={analysisData.classification_report} />
                        </Paper>
                        {analysisData.confusion_matrix && (
                            <Paper variant="outlined" sx={{ 
                                p: 2, 
                                flex: 1, 
                                display: 'flex',
                            }}>
                                <Box sx={{ flexGrow: 1, position: 'relative' }}>
                                    <ConfusionMatrix
                                        matrix={analysisData.confusion_matrix.values}
                                        labels={analysisData.confusion_matrix.labels}
                                    />
                                </Box>
                            </Paper>
                        )}
                    </Box>
                )}

                {analysisData.feature_importance && (
                    <Box sx={{ display:'flex', flexDirection:flexDirection, flexGrow: 1 }}>
                        <Paper variant="outlined" sx={{ p: 2, flexGrow: 1, display: 'flex' }}>
                            <FeatureImportanceChart data={analysisData.feature_importance} />
                        </Paper>
                    </Box>
                )}
            </Box>

            <Snackbar
                open={snackbarOpen}
                autoHideDuration={3000}
                onClose={handleSnackbarClose}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            >
                <Alert onClose={handleSnackbarClose} severity="success" sx={{ width: '100%' }}>
                    Analysis data copied to clipboard!
                </Alert>
            </Snackbar>
        </Box>
    );
};