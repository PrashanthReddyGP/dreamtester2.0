// src/components/machinelearning/shared/EditorPanel.tsx
import React from 'react';
import { Box, Button, Paper, Typography } from '@mui/material';
import Editor from '@monaco-editor/react';

interface EditorPanelProps {
    code: string;
    onChange: (value: string | undefined) => void;
    isCustomMode: boolean;
    templateInfo?: { name: string; description: string; };
    onSaveAsTemplate: () => void;
}

interface FeatureEditorPanelProps {
    code: string;
    onChange: (value: string | undefined) => void;
}


// LABELLING
export const EditorPanel: React.FC<EditorPanelProps> = ({ code, onChange, isCustomMode, templateInfo, onSaveAsTemplate }) => (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Paper square elevation={0} sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider', textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
                <Box>
                    <Typography variant="h6">Labeling Logic Editor</Typography>
                    <Typography variant="body2" color="text.secondary">
                        {isCustomMode ? "Define your custom labeling function below." : templateInfo?.description || 'Select a template'}
                    </Typography>
                </Box>
                <Button variant="outlined" size="medium" onClick={onSaveAsTemplate} sx={{ position: 'absolute', right: 16 }}>
                    Save as Template
                </Button>
            </Box>
        </Paper>
        <Box sx={{ flexGrow: 1 }}>
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
    </Box>
);



// ENGINEERING
export const FeaturesEditorPanel: React.FC<FeatureEditorPanelProps> = ({ code, onChange }) => (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ flexGrow: 1 }}>
            <Editor
                height="100%"
                language="python"
                theme="app-dark-theme" 
                value={code}
                onChange={onChange}
                options={{
                    minimap: { enabled: true},
                    fontSize: 14,
                    wordWrap: 'on',
                    scrollBeyondLastLine: false,
                    padding: { top: 24 },
                    readOnly: false 
                }}
            />
        </Box>
    </Box>
);