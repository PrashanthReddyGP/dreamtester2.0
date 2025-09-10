// src/components/machinelearning/MlSidebar.tsx
import React from 'react';
import { List, ListItem, ListItemButton, ListItemIcon, Tooltip, Paper } from '@mui/material';
import { Storage as DataIcon, Tune as FeatureIcon, Label as LabelingIcon, ModelTraining as TrainingIcon } from '@mui/icons-material';

type ActiveTab = 'data' | 'features' | 'labeling' | 'training';

interface MlSidebarProps {
    activeTab: ActiveTab;
    setActiveTab: (tab: ActiveTab) => void;
}

export const MlSidebar: React.FC<MlSidebarProps> = ({ activeTab, setActiveTab }) => {
    const tabs = [
        { key: 'data', icon: <DataIcon />, label: 'Data' },
        { key: 'features', icon: <FeatureIcon />, label: 'Feature Engineering' },
        { key: 'labeling', icon: <LabelingIcon />, label: 'Labeling & Splitting' },
        { key: 'training', icon: <TrainingIcon />, label: 'Training & Backtesting' },
    ];

    return (
        <Paper elevation={2} sx={{ width: 60, height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', py: 2, zIndex: 1 }}>
            <List>
                {tabs.map(tab => (
                    <ListItem key={tab.key} disablePadding>
                        <Tooltip title={tab.label} placement="right">
                            <ListItemButton
                                selected={activeTab === tab.key}
                                onClick={() => setActiveTab(tab.key as ActiveTab)}
                                sx={{ flexDirection: 'column', my: 1 }}
                            >
                                <ListItemIcon sx={{ minWidth: 0 }}>{tab.icon}</ListItemIcon>
                            </ListItemButton>
                        </Tooltip>
                    </ListItem>
                ))}
            </List>
        </Paper>
    );
};