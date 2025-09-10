import React from 'react';
import { 
    Box, Typography, Button, CircularProgress, Paper, Select, MenuItem, 
    FormGroup, FormControlLabel, Checkbox,
    InputLabel, FormControl, IconButton, Divider, ListItemIcon, ListItemText
} from '@mui/material';
import { 
    Delete as DeleteIcon,
    AddCircleOutline as AddCircleOutlineIcon
} from '@mui/icons-material';
import type { MLConfig } from '../types';
import { DataGridDisplay, DataInfoDisplay } from '../shared/DataDisplays';
import { FeaturesEditorPanel } from '../shared/EditorPanel';
import type { FETemplate } from '../FeatureEngineeringTemplates'

const API_URL = 'http://127.0.0.1:8000';

interface FeatureEngineeringTabProps {
    config: MLConfig;
    onChange: (path: string, value: any) => void;
    onEngineer: () => void;
    onSaveTemplate: () => void;
    onDeleteTemplate: (key: string) => void;
    displayData: any[];
    displayInfo: any;
    isEngineering: boolean;
    feTemplates: { [key: string]: FETemplate };
}

// Special value to trigger the save action from the dropdown
const SAVE_NEW_TEMPLATE_VALUE = '__SAVE_NEW__';

export const FeatureEngineeringTab: React.FC<FeatureEngineeringTabProps> = ({ config, onChange, onEngineer, onSaveTemplate, onDeleteTemplate, displayData, displayInfo, isEngineering, feTemplates  }) => {

    // A dedicated handler for the template dropdown
    const handleTemplateChange = (event: any) => {
        const selectedValue = event.target.value;
        if (selectedValue === SAVE_NEW_TEMPLATE_VALUE) {
            // If the user clicks our special "Save" item, trigger the dialog
            onSaveTemplate();
        } else {
            // Otherwise, it's a normal template selection
            onChange('preprocessing.featureTemplateKey', selectedValue);
        }
    };

    return (
        <Box sx={{ p: 1, height: '100%', display: 'flex', gap: 1, pb: 4 }}>
            
            {/* Left Panel: Controls */}
            <Box sx={{ flexGrow: 1, minWidth: 400, maxWidth: 400, height: '100%' }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, height: '100%' }}>
                    {/* Preprocessing & Scaling Panel */}
                    <Paper variant="outlined" sx={{ p: 2, flexShrink: 0 }}>
                        <Typography variant="subtitle1" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', pb: 1 }}>Data Processing</Typography>
                    <FormControl fullWidth size="small" sx={{ mt: 1 }}>
                        <InputLabel>Template</InputLabel>
                        <Select
                            value={config.preprocessing.featureTemplateKey}
                            label="Template"
                            onChange={handleTemplateChange} // Use the new handler
                        >
                            {Object.entries(feTemplates).map(([key, template]) => (
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
                            {/* --- ADD THE SPECIAL "SAVE" ACTION ITEM --- */}
                            <Divider sx={{ my: 1 }} />
                            <MenuItem value={SAVE_NEW_TEMPLATE_VALUE}>
                                <ListItemIcon>
                                    <AddCircleOutlineIcon fontSize="small" />
                                </ListItemIcon>
                                <ListItemText primary="Save code as new template..." />
                            </MenuItem>
                        </Select>
                    </FormControl>
                        <Button variant="contained" onClick={onEngineer} disabled={isEngineering} fullWidth sx={{ mt: 2 }}>
                            {isEngineering ? <CircularProgress size={24} /> : 'Process Data'}
                        </Button>
                    </Paper>

                    {/* Technical Indicators Panel */}
                    <Paper variant="outlined" sx={{ p: 2, display: 'flex', flexDirection: 'column', flexGrow: 1, overflowY: 'auto', '&::-webkit-scrollbar': { width: '0px' } }}>
                        <DataInfoDisplay info={displayInfo} />
                    </Paper>

                </Box>
            </Box>
            
            {/* Right Panel: Data Display */}
            <Box sx={{ flexGrow: 2, display: 'flex', flexDirection: 'column', gap: 1, maxWidth: 'calc(100% - 400px)', height: '100%' }}>
                <Paper variant='outlined' sx={{ minHeight: '268px' }}>
                    <FeaturesEditorPanel 
                        code={config.preprocessing.customFeatureCode}
                        onChange={(value) => onChange('preprocessing.customFeatureCode', value || '')}
                    />
                </Paper>
                <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                    <DataGridDisplay
                        key={displayInfo?.["Data Points"] + "_features"} 
                        data={displayData}
                        info={displayInfo} 
                        title="Engineered Features Data"
                    />
                </Box>
            </Box>
        </Box>
    );
};