import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Divider, Stack, FormControl, InputLabel, Select, MenuItem, Button, CircularProgress, FormGroup, FormControlLabel, Checkbox, TextField, Slider, IconButton } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import Editor from '@monaco-editor/react';
import { INITIAL_LABELING_TEMPLATES } from '../components/machinelearning/LabelingTemplates';
import type { LabelingTemplate } from '../components/machinelearning/LabelingTemplates';
import { DataSourcePanel } from '../components/machinelearning/DataSourcePanel';
import { NameInputDialog } from '../components/common/NameItemDialog'; 
import { ConfirmationDialog } from '../components/common/ConfirmationDialog';
import { useNavigate } from 'react-router-dom';
import { useTerminal } from '../context/TerminalContext'; // Assuming this is the correct path

// Define the shape of our configuration objects
export interface IndicatorParam {
  [key: string]: number | string;
}

export interface IndicatorConfig {
    id: string; // Unique ID for React keys, e.g., "RSI_1"
    name: string; // The key from INDICATOR_DEFINITIONS, e.g., "RSI"
    params: IndicatorParam;
}

export interface MLConfig {
    problemDefinition: {
        type: 'template' | 'custom';
        templateKey: string;
        customCode: string;
    };

    dataSource: {
        symbol: string;
        timeframe: string;
    };

    features: IndicatorConfig[];
    model: {
        name: string;
    };

    validation: {
        method: 'train_test_split' | 'walk_forward';
        trainSplit: number;
        walkForwardTrainWindow: number;
        walkForwardTestWindow: number;
    };

    preprocessing: {
        scaler: 'none' | 'StandardScaler' | 'MinMaxScaler';
        removeCorrelated: boolean;
        correlationThreshold: number;
        usePCA: boolean;
        pcaComponents: number;
    };
}

// --- Settings Panel Component (UPDATED) ---
interface SettingsPanelProps {
    config: MLConfig;
    onChange: (path: string, value: any) => void;
    onRun: () => void;
    isRunning: boolean;
    labelingTemplates: { [key: string]: LabelingTemplate };
    onDeleteTemplate: (key: string) => void;
}


// --- Settings Panel Component (Now Simplified) ---
interface SettingsPanelProps {
    config: MLConfig;
    onChange: (path: string, value: any) => void;
    onRun: () => void;
    isRunning: boolean;
}


// --- Editor Panel Component (Unchanged) ---
interface EditorPanelProps {
    code: string;
    onChange: (value: string | undefined) => void;
    isCustomMode: boolean;
    templateInfo: { name: string; description: string; };
    onSaveAsTemplate: () => void;
}

// Mock data - keep what's still needed
const MOCK_MODELS = ['LogisticRegression', 'RandomForestClassifier', 'LightGBMClassifier', 'XGBoostClassifier', 'LightGBMRegressor', 'XGBoostRegressor'];

// Updated initialConfig with the new features structure
const initialConfig: MLConfig = {
    problemDefinition: {
        type: 'template',
        templateKey: 'triple_barrier',
        customCode: INITIAL_LABELING_TEMPLATES.triple_barrier.code,
    },
    dataSource: {
        symbol: 'ADAUSDT',
        timeframe: '15m',
    },
    features: [
    ],
    model: {
        name: 'LightGBMClassifier',
    },
    validation: {
        method: 'walk_forward',
        trainSplit: 70,
        walkForwardTrainWindow: 365,
        walkForwardTestWindow: 30,
    },
    preprocessing: {
        scaler: 'StandardScaler',
        removeCorrelated: true,
        correlationThreshold: 0.9,
        usePCA: false,
        pcaComponents: 5,
    },
};

const ResizeHandle = () => (
    <PanelResizeHandle style={{ width: '4px', background: 'transparent' }}>
        <Box sx={{ width: '4px', height: '40px', borderRadius: '2px', backgroundColor: 'action.hover', transition: 'background-color 0.2s ease-in-out', '&:hover': { backgroundColor: 'divider' } }} />
    </PanelResizeHandle>
);

const API_URL = 'http://127.0.0.1:8000'; 

export const MachineLearning: React.FC = () => {
    const [config, setConfig] = useState<MLConfig>(initialConfig);
    const [runStatus, setRunStatus] = useState<'idle' | 'submitting' | 'error'>('idle'); 

    // Initialize state with only the non-deletable, default templates
    const [labelingTemplates, setLabelingTemplates] = useState<{ [key: string]: LabelingTemplate }>(INITIAL_LABELING_TEMPLATES);
    const [isLoadingTemplates, setIsLoadingTemplates] = useState(true);

    const [isSaveTemplateDialogOpen, setIsSaveTemplateDialogOpen] = useState(false);
    const [deleteTemplateState, setDeleteTemplateState] = useState<{ open: boolean; key: string | null }>({ open: false, key: null });

    // --- Context and Navigation Hooks ---
    const navigate = useNavigate();
    const { connectToBatch, toggleTerminal } = useTerminal(); // Get the WebSocket handler from context

    // --- NEW: Fetch custom templates from backend on component mount ---
    useEffect(() => {
        const fetchCustomTemplates = async () => {
            try {
                const response = await fetch(`${API_URL}/api/ml/templates`);
                if (!response.ok) {
                    throw new Error("Failed to fetch custom templates");
                }
                const customTemplates = await response.json();
                
                // Combine the initial hardcoded templates with the custom ones from the DB
                setLabelingTemplates(prev => ({
                    ...prev, // Keep the default, non-deletable ones
                    ...customTemplates // Add/overwrite with user's saved templates
                }));

            } catch (error) {
                console.error("Could not load custom templates:", error);
                // Optionally show an alert to the user
            } finally {
                setIsLoadingTemplates(false);
            }
        };
        
        fetchCustomTemplates();
    }, []); // Empty dependency array ensures this runs only once

    useEffect(() => {
        if (config.problemDefinition.type === 'template') {
            const newCode = labelingTemplates[config.problemDefinition.templateKey]?.code || '// Select a valid template';
            if (newCode !== config.problemDefinition.customCode) {
                setConfig(prev => ({
                    ...prev,
                    problemDefinition: { ...prev.problemDefinition, customCode: newCode },
                }));
            }
        }
    }, [config.problemDefinition.type, config.problemDefinition.templateKey, labelingTemplates]);

    const handleConfigChange = (path: string, value: any) => {
        setConfig(prev => {
            const keys = path.split('.');
            let tempState = { ...prev };
            let current = tempState as any;
            for (let i = 0; i < keys.length - 1; i++) {
                current[keys[i]] = { ...current[keys[i]] };
                current = current[keys[i]];
            }
            current[keys[keys.length - 1]] = value;
            return tempState;
        });
    };

    // --- Template Management Logic ---
    const handleSaveTemplate = async (templateName: string) => {
        const templateKey = `custom_${templateName.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`;
        
        const newTemplate: LabelingTemplate = {
            name: templateName,
            description: 'A custom user-defined template.',
            code: config.problemDefinition.customCode,
            isDeletable: true,
        };

        try {
            // --- API Call to save template ---
            await fetch(`${API_URL}/api/ml/templates`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key: templateKey, ...newTemplate }),
            });

            // Update local state on success
            const updatedTemplates = { ...labelingTemplates, [templateKey]: newTemplate };
            setLabelingTemplates(updatedTemplates);

            setConfig(prev => ({
                ...prev,
                problemDefinition: {
                    ...prev.problemDefinition,
                    type: 'template',
                    templateKey: templateKey,
                }
            }));
        } catch (error) {
            console.error("Failed to save template:", error);
            alert("Error: Could not save the template to the server.");
        }
    };

    const handleDeleteTemplateConfirm = async () => {
        if (!deleteTemplateState.key) return;

        try {
            // --- API Call to delete template ---
            await fetch(`${API_URL}/api/ml/templates/${deleteTemplateState.key}`, {
                method: 'DELETE',
            });
            
            // Update local state on success
            const { [deleteTemplateState.key]: _, ...remainingTemplates } = labelingTemplates;
            setLabelingTemplates(remainingTemplates);

            if (config.problemDefinition.templateKey === deleteTemplateState.key) {
                const firstTemplateKey = Object.keys(remainingTemplates)[0];
                setConfig(prev => ({
                    ...prev,
                    problemDefinition: { ...prev.problemDefinition, templateKey: firstTemplateKey }
                }));
            }
        } catch (error) {
            console.error("Failed to delete template:", error);
            alert("Error: Could not delete the template from the server.");
        } finally {
            setDeleteTemplateState({ open: false, key: null });
        }
    };

    const handleRunPipeline = async () => {
        setRunStatus('submitting');
        toggleTerminal(true); // Open the terminal immediately to show submission status
        console.log("Submitting ML Pipeline with config:", JSON.stringify(config, null, 2));
        
        try {
            // Step 1: Submit the job to the backend
            const response = await fetch(`${API_URL}/api/ml/run`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                // Try to get a detailed error message from the backend's JSON response
                const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred.' }));
                throw new Error(errorData.detail || `Server responded with status: ${response.status}`);
            }

            // Step 2: Get the batch_id from the successful initial response
            const result: { batch_id: string } = await response.json();
            
            if (result.batch_id) {
                // Step 3: Connect to the WebSocket stream using the batch_id
                connectToBatch(result.batch_id);
                
                // Step 4: Navigate to the analysis page to view live results
                navigate('/analysis');
            } else {
                throw new Error("Submission was successful, but no batch ID was returned from the server.");
            }

        } catch (error) {
            // Handle any errors during submission
            console.error("Failed to run ML Pipeline:", error);
            alert(`Error submitting pipeline: ${error instanceof Error ? error.message : 'An unknown error occurred.'}`);
            setRunStatus('error'); // Set an error state
        }
        // NOTE: We don't reset the status here, as successful navigation will unmount the component.
        // If an error occurs, the status remains 'error' until the user tries again.
    };

    // --- Template Management Logic (handleSaveTemplate, handleDeleteTemplateConfirm) remains the same ---
    
    // Reset status if it's in error and user changes something
    useEffect(() => {
        if (runStatus === 'error') {
            setRunStatus('idle');
        }
    }, [config]);

    return (
        <Box sx={{ height: '100%', width: '100vw' }}>
            <PanelGroup direction="horizontal">
                <Panel defaultSize={25} minSize={20}>
                    <SettingsPanel 
                        config={config} 
                        onChange={handleConfigChange} 
                        onRun={handleRunPipeline} 
                        isRunning={runStatus === 'submitting'} 
                        labelingTemplates={labelingTemplates}
                        onDeleteTemplate={(key) => setDeleteTemplateState({ open: true, key })}
                    />
                </Panel>
                <ResizeHandle />
                <Panel defaultSize={50} minSize={30}>
                    <EditorPanel
                        code={config.problemDefinition.customCode}
                        onChange={(value) => handleConfigChange('problemDefinition.customCode', value || '')}
                        isCustomMode={config.problemDefinition.type === 'custom'}
                        templateInfo={labelingTemplates[config.problemDefinition.templateKey]}
                        onSaveAsTemplate={() => setIsSaveTemplateDialogOpen(true)}
                    />
                </Panel>
                <ResizeHandle />
                <Panel defaultSize={25} minSize={20}>
                    <DataSourcePanel 
                        config={config} 
                        onConfigChange={handleConfigChange} 
                    />
                </Panel>
            </PanelGroup>

            <NameInputDialog
                open={isSaveTemplateDialogOpen}
                onClose={() => setIsSaveTemplateDialogOpen(false)}
                onConfirm={handleSaveTemplate}
                dialogTitle="Save as Template"
                dialogText="Please enter a name for your new labeling logic template."
                confirmButtonText="Save"
            />
            
            <ConfirmationDialog
                open={deleteTemplateState.open}
                onClose={() => setDeleteTemplateState({ open: false, key: null })}
                onConfirm={handleDeleteTemplateConfirm}
                title="Delete Template?"
                message="Are you sure you want to permanently delete this template?"
            />
        
        </Box>
    );
};

const SettingsPanel: React.FC<SettingsPanelProps> = ({ config, onChange, onRun, isRunning, labelingTemplates, onDeleteTemplate }) => {
  return (
    <Paper elevation={0} sx={{ height: '100%', pr: 2, pl: 2, display: 'flex', flexDirection: 'column' }}>
        <Stack spacing={2} divider={<Divider />}>
        <Typography variant="h6" gutterBottom sx={{ textAlign: 'center', pt: 3 }}>ML Pipeline Configuration</Typography>
        
        {/* Problem Definition */}
        <Box>
            <Typography variant="subtitle1" gutterBottom sx={{ pl: 1, pb: 1 }}>1. Problem Definition</Typography>
            <Stack spacing={2}>
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
            </Stack>
        </Box>


        <Box>
            <Typography variant="subtitle1" gutterBottom sx={{ pl: 1, pb: 1 }}>2. Model</Typography>
            <FormControl fullWidth size="small">
                <InputLabel>ML Model</InputLabel>
                <Select value={config.model.name} label="ML Model" onChange={(e) => onChange('model.name', e.target.value)}>
                    {MOCK_MODELS.map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
                </Select>
            </FormControl>
        </Box>

        <Box>
            <Typography variant="subtitle1" gutterBottom sx={{ pl: 1, pb: 1 }}>3. Validation</Typography>
            <Stack spacing={2}>
                <FormControl fullWidth size="small">
                    <InputLabel>Method</InputLabel>
                    <Select value={config.validation.method} label="Method" onChange={(e) => onChange('validation.method', e.target.value)}>
                        <MenuItem value="walk_forward">Walk-Forward</MenuItem>
                        <MenuItem value="train_test_split">Train/Test Split</MenuItem>
                    </Select>
                </FormControl>
                {config.validation.method === 'walk_forward' ? (
                    <>
                    <TextField label="Training Window (days)" type="number" size="small" value={config.validation.walkForwardTrainWindow} onChange={e => onChange('validation.walkForwardTrainWindow', Number(e.target.value))} />
                    <TextField label="Testing Window (days)" type="number" size="small" value={config.validation.walkForwardTestWindow} onChange={e => onChange('validation.walkForwardTestWindow', Number(e.target.value))} />
                    </>
                ) : (
                    <>
                    <Typography gutterBottom>Train/Validation/Test Split (%)</Typography>
                    <Slider value={config.validation.trainSplit} onChange={(e, val) => onChange('validation.trainSplit', val as number)} aria-labelledby="train-split-slider" valueLabelDisplay="auto" step={5} marks min={10} max={90} />
                    <Typography variant="body2" align="center">
                        {`Train: ${config.validation.trainSplit}% | Validation: ${Math.round((100 - config.validation.trainSplit) / 2)}% | Test: ${100 - config.validation.trainSplit - Math.round((100 - config.validation.trainSplit) / 2)}%`}
                    </Typography>
                    </>
                )}
            </Stack>
        </Box>

        <Box>
            <Typography variant="subtitle1" gutterBottom sx={{ pl: 1, pb: 1 }}>4. Preprocessing</Typography>
            <Stack spacing={1}>
                <FormControl fullWidth size="small">
                    <InputLabel>Scaler</InputLabel>
                    <Select value={config.preprocessing.scaler} label="Scaler" onChange={(e) => onChange('preprocessing.scaler', e.target.value)}>
                        <MenuItem value="none">None</MenuItem>
                        <MenuItem value="StandardScaler">Standard Scaler</MenuItem>
                        <MenuItem value="MinMaxScaler">MinMax Scaler</MenuItem>
                    </Select>
                </FormControl>
                <FormGroup>
                    <FormControlLabel control={<Checkbox checked={config.preprocessing.removeCorrelated} onChange={(e) => onChange('preprocessing.removeCorrelated', e.target.checked)} />} label="Remove Correlated Features" />
                    <FormControlLabel control={<Checkbox checked={config.preprocessing.usePCA} onChange={(e) => onChange('preprocessing.usePCA', e.target.checked)} />} label="Use PCA for Feature Reduction" />
                </FormGroup>
            </Stack>
        </Box>
        
        <Box sx={{ pt: 2 }}>
            <Button variant="contained" color="primary" fullWidth onClick={onRun} disabled={isRunning}>
                {isRunning ? <CircularProgress size={24} /> : 'Run ML Pipeline'}
            </Button>
        </Box>
        </Stack>
    </Paper>
    );
};

const EditorPanel: React.FC<EditorPanelProps> = ({ code, onChange, isCustomMode, templateInfo, onSaveAsTemplate }) => (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Paper square elevation={0} sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider', textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
                <Box>
                    <Typography variant="h6">Labeling Logic</Typography>
                    <Typography variant="body2" color="text.secondary">
                        {isCustomMode ? "Define your custom labeling function below." : templateInfo?.description || 'Select a template'}
                    </Typography>
                </Box>
                <Button 
                    variant="outlined" 
                    size="medium"
                    onClick={onSaveAsTemplate}
                    sx={{ position: 'absolute', right: 16 }}
                >
                    Save as Template
                </Button>
            </Box>
        </Paper>
        <Box sx={{ height: '40%', flexGrow: 1 }}>
            <Editor
                height="100%"
                language="python"
                theme="app-dark-theme" 
                value={code}
                onChange={onChange}
                options={{
                    minimap: { enabled: false},
                    fontSize: 14,
                    wordWrap: 'on',
                    scrollBeyondLastLine: false,
                    padding: { top: 24 },
                    readOnly: false 
                }}
            />
        </Box>
        {/* <Paper square elevation={0} sx={{ p: 1.5, borderTop: 1, borderColor: 'divider', textAlign: 'center', height: '50%', pb: 8 }}>
            <Typography variant="h6" sx={{ pb: 2 }}>ML Results</Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' , height: '100%', border: '2px dashed', borderColor: 'divider', borderRadius: 1 }}>
                <Typography variant="body2" color="text.secondary">Statistical Results show up in here...</Typography>
            </Box>
        </Paper> */}
    </Box>
);