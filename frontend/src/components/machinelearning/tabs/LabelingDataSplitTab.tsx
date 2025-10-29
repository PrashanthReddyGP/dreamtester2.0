// src/components/machinelearning/tabs/LabelingDataSplitTab.tsx
import React from 'react';
import { Box, Typography, Button, CircularProgress, Paper, FormControl, InputLabel, Select, MenuItem, IconButton, Slider, Stack, Divider, Checkbox, FormControlLabel, FormGroup } from '@mui/material';
import { Delete as DeleteIcon } from '@mui/icons-material';
import type { MLConfig } from '../types';
import type { LabelingTemplate } from '../LabelingTemplates'; // Assuming this export from your existing file
import { EditorPanel } from '../shared/EditorPanel';

interface LabelingTabProps {
    config: MLConfig;
    onChange: (path: string, value: any) => void;
    labelingTemplates: { [key: string]: LabelingTemplate };
    onDeleteTemplate: (key: string) => void;
    onSaveTemplate: () => void;
    onValidate: () => void;
    isValidating: boolean;
    validationInfo: any | null;
}

const ValidationInfoDisplay: React.FC<{ info: any | null }> = ({ info }) => {
    if (!info) {
        return (
            <Typography variant="body2" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
                Click "Validate" to see label distribution and data split details.
            </Typography>
        );
    }

    if (info.error) {
        return <Typography color="error" sx={{ p: 2 }}>Error: {info.error}</Typography>;
    }

    const { label_distribution, split_info } = info;

    return (
        <Stack spacing={2} sx={{ p: 2, width: '100%' }}>
            {label_distribution && (
                <Box sx={{ textAlign: 'center', pb: 2 }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', pb: 0 }}>Label Distribution</Typography>
                    <Divider sx={{ mb: 1 }} />
                    {Object.entries(label_distribution).length > 0 ? (
                        Object.entries(label_distribution).map(([label, count]: [string, any]) => (
                            <Typography key={label} variant="body2" color="text.secondary">
                                Label '{label}' : <b>{count}</b> samples
                            </Typography>
                        ))
                    ) : (
                        <Typography variant="body2" color="text.secondary">No labels were generated.</Typography>
                    )}
                </Box>
            )}
            {split_info && (
                <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', pb: 0 }}>Data Split ({split_info.method})</Typography>
                    <Divider sx={{ mb: 1 }} />
                        <Typography variant="body2" color="text.secondary">Total Samples : <b>{split_info.total_samples}</b></Typography>
                        {split_info.train_samples !== undefined && <Typography variant="body2" color="text.secondary">Train Samples : <b>{split_info.train_samples}</b></Typography>}
                        {split_info.test_samples !== undefined && <Typography variant="body2" color="text.secondary">Test Samples : <b>{split_info.test_samples}</b></Typography>}
                        {split_info.train_window_size !== undefined && <Typography variant="body2" color="text.secondary">Train Window : <b>{split_info.train_window_size}</b></Typography>}
                        {split_info.test_window_size !== undefined && <Typography variant="body2" color="text.secondary">Test Window : <b>{split_info.test_window_size}</b></Typography>}
                        {split_info.approximate_folds !== undefined && <Typography variant="body2" color="text.secondary">Approx. Folds : <b>{split_info.approximate_folds}</b></Typography>}
                </Box>
            )}
        </Stack>
    );
};

export const LabelingDataSplitTab: React.FC<LabelingTabProps> = ({ config, onChange, labelingTemplates, onDeleteTemplate, onSaveTemplate, onValidate, isValidating, validationInfo }) => (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'row', gap: 1 }}>
        {/* <Box sx={{ flexGrow: 0 }}>
            <Typography variant="h5" gutterBottom sx={{ textAlign: 'center' }}>Labeling & Splitting</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                Define the target variable (the "question") and the data validation strategy.
            </Typography>
        </Box> */}
        <Box sx={{ flexGrow: 0, height: '98.5%', display: 'flex', flexDirection: 'row', pl: 1, pt: 1, pb: 1 }}>
            <Box sx={{ flexGrow: 0, width: '400px', gap: 1, display: 'flex', flexDirection: 'column', pb: 1 }}>
                <Paper variant="outlined" sx={{p:2, flexGrow: 0, display: 'flex', flexDirection: 'column', gap: 2}}>
                    <Typography variant="subtitle1" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', pb: 0 }}>Labeling Logic (The Question)</Typography>
                        <FormControl fullWidth size="small">
                            <InputLabel>Type</InputLabel>
                            <Select value={config.problemDefinition.type} label="Type" onChange={(e) => onChange('problemDefinition.type', e.target.value)}>
                                <MenuItem value="template">Template</MenuItem>
                                <MenuItem value="custom">Custom</MenuItem>
                            </Select>
                        </FormControl>
                        {config.problemDefinition.type === 'template' && (
                            <FormControl fullWidth size="small">
                                <InputLabel>Question Template</InputLabel>
                                <Select value={config.problemDefinition.templateKey} label="Question Template" onChange={(e) => onChange('problemDefinition.templateKey', e.target.value)}>
                                    {Object.entries(labelingTemplates).map(([key, template]) => (
                                        <MenuItem key={key} value={key}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                                                {template.name}
                                                {template.isDeletable && (
                                                <IconButton size="small" onMouseDown={(e) => e.stopPropagation()} onClick={(e) => { e.stopPropagation(); onDeleteTemplate(key); }}>
                                                    <DeleteIcon fontSize="small" />
                                                </IconButton>
                                                )}
                                            </Box>
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        )}
                </Paper>

                <Paper variant="outlined" sx={{p:2, flexGrow: 0, display: 'flex', flexDirection: 'column', gap: 2}}>
                    <Typography variant="subtitle1" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', pb: 0 }}>Data Validation Method</Typography>
                    <FormControl fullWidth size="small">
                        <InputLabel>Method</InputLabel>
                        <Select value={config.validation.method} label="Method" onChange={(e) => onChange('validation.method', e.target.value)}>
                            <MenuItem value="walk_forward">Walk-Forward</MenuItem>
                            <MenuItem value="train_test_split">Train/Test Split</MenuItem>
                        </Select>
                    </FormControl>
                    {config.validation.method === 'train_test_split' && (
                        <>
                            {/* <Typography gutterBottom>Train/Test Split (%)</Typography> */}
                            <Slider value={config.validation.trainSplit} onChange={(e, val) => onChange('validation.trainSplit', val as number)} valueLabelDisplay="auto" />
                        </>
                    )}
                </Paper>
                <Paper variant="outlined" sx={{ p:2, flexGrow: 0, display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', pb: 0 }}>Data Scaling</Typography>
                    <FormControl fullWidth size="small" sx={{ mt: 1 }}>
                        <InputLabel>Scaler</InputLabel>
                        <Select fullWidth size="small" value={config.preprocessing.scaler} label="Scalar" onChange={(e) => onChange('preprocessing.scaler', e.target.value)}>
                            <MenuItem value="none">None</MenuItem>
                            <MenuItem value="StandardScaler">Standard Scaler</MenuItem>
                            <MenuItem value="MinMaxScaler">MinMax Scaler</MenuItem>
                        </Select>
                    </FormControl>
                    <FormGroup sx={{ mt: 1, mb: 1 }}>
                        <FormControlLabel control={<Checkbox size="small" checked={config.preprocessing.removeCorrelated} onChange={(e) => onChange('preprocessing.removeCorrelated', e.target.checked)} />} label="Remove Correlated Features" />
                        <FormControlLabel control={<Checkbox size="small" checked={config.preprocessing.usePCA} onChange={(e) => onChange('preprocessing.usePCA', e.target.checked)} />} label="Use PCA" />
                    </FormGroup>
                    <Button variant="contained" onClick={onValidate} disabled={isValidating} fullWidth>
                        {isValidating ? <CircularProgress size={24} /> : 'Prepare Data'}
                    </Button>
                </Paper>
                <Paper variant='outlined' sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column'}}>
                    <ValidationInfoDisplay info={validationInfo} />
                </Paper>
            </Box>
        </Box>
        <Box sx={{ flexGrow: 1, width: '50%', height: '100%'}}>
            <EditorPanel 
                code={config.problemDefinition.customCode}
                onChange={(value) => onChange('problemDefinition.customCode', value || '')}
                isCustomMode={config.problemDefinition.type === 'custom'}
                templateInfo={labelingTemplates[config.problemDefinition.templateKey]}
                onSaveAsTemplate={onSaveTemplate}
            />
        </Box>
    </Box>
);