import React from 'react';
import { Box, List, ListItem, ListItemText, Switch, Typography, Paper } from '@mui/material';

// The shape of a single indicator's configuration
export interface IndicatorConfig {
    name: string;
    isVisible: boolean;
    color: string;
}

interface ChartSettingsPanelProps {
    // An object where keys are indicator names (e.g., "SMA_50")
    configs: Record<string, IndicatorConfig>;
    // A callback to notify the parent component of a change
    onConfigChange: (name: string, newConfig: Partial<IndicatorConfig>) => void;
    }

    export const ChartSettingsPanel: React.FC<ChartSettingsPanelProps> = ({ configs, onConfigChange }) => {
    // Helper to handle color input changes
    const handleColorChange = (name: string, event: React.ChangeEvent<HTMLInputElement>) => {
        onConfigChange(name, { color: event.target.value });
    };

    // Helper to handle visibility switch changes
    const handleVisibilityChange = (name: string, event: React.ChangeEvent<HTMLInputElement>) => {
        onConfigChange(name, { isVisible: event.target.checked });
    };

    return (
        <Paper elevation={2} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                <Typography variant="h6">Chart Settings</Typography>
            </Box>
            <List sx={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
                {Object.values(configs).map((config) => (
                <ListItem key={config.name} dense>
                    {/* The Color Picker Input */}
                    <input
                    type="color"
                    value={config.color}
                    onChange={(e) => handleColorChange(config.name, e)}
                    style={{ width: '24px', height: '24px', border: 'none', background: 'none', marginRight: '16px', cursor: 'pointer' }}
                    />
                    {/* The Indicator Name */}
                    <ListItemText primary={config.name} />
                    {/* The Visibility Switch */}
                    <Switch
                    edge="end"
                    checked={config.isVisible}
                    onChange={(e) => handleVisibilityChange(config.name, e)}
                    />
                </ListItem>
                ))}
            </List>
        </Paper>
    );
};