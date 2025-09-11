import React, { useState } from 'react';
import { Box, Typography, IconButton, Tooltip, Tabs, Tab } from '@mui/material'; // Import Tabs and Tab
import {
    ChevronLeft,
    ChevronRight,
    AlignHorizontalLeft,
    AlignHorizontalRight,
    AlignVerticalTop,
    AlignVerticalBottom,
    ExpandLess, // For collapsing top panel
    ExpandMore, // For collapsing bottom panel
} from '@mui/icons-material';
import { styled } from '@mui/material/styles'; // Import styled if you are putting the handle here
import { DataGridDisplay } from '../../components/pipeline/DataDisplays';

// Define the types for the props
type PanelPosition = 'left' | 'right' | 'top' | 'bottom';

interface SidePanelProps {
    isPanelOpen: boolean;
    panelPosition: PanelPosition;
    togglePanel: () => void;
    setPanelPosition: (position: PanelPosition) => void;
    displayData: any[];
    displayInfo: any;
}

interface CollapsedPanelHandleProps {
    position: 'left' | 'right' | 'top' | 'bottom';
    onToggle: () => void;
}

export const CollapsedPanelHandle: React.FC<CollapsedPanelHandleProps> = ({ position, onToggle }) => {
    const getIcon = () => {
        switch (position) {
            case 'left': return <ChevronRight />;
            case 'right': return <ChevronLeft />;
            case 'top': return <ExpandMore />;
            case 'bottom': return <ExpandLess />;
        }
    };

    const getPositionStyles = () => {
        const baseStyles = {
            position: 'absolute',
            zIndex: 10, // Ensure it's above the React Flow canvas
            bgcolor: 'background.paper',
            borderRadius: '0 12px 12px 0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
        };

        switch (position) {
            case 'left':
                return { ...baseStyles, top: '50%', left: 0, transform: 'translateY(-50%)', borderRadius: '0 12px 12px 0' };
            case 'right':
                return { ...baseStyles, top: '50%', right: 0, transform: 'translateY(-50%)', borderRadius: '12px 0 0 12px' };
            case 'top':
                return { ...baseStyles, top: 0, left: '50%', transform: 'translateX(-50%)', borderRadius: '0 0 12px 12px' };
            case 'bottom':
                return { ...baseStyles, bottom: 0, left: '50%', transform: 'translateX(-50%)', borderRadius: '12px 12px 0 0' };
        }
    };

    return (
        <Box sx={getPositionStyles()}>
            <IconButton onClick={onToggle} size="small">
                {getIcon()}
            </IconButton>
        </Box>
    );
};

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
    noPadding?: boolean;
}

// A simple component to render the content of a tab
function TabPanel(props: TabPanelProps) {
    const { children, value, index, noPadding, ...other } = props;

    return (
        <div
        role="tabpanel"
        hidden={value !== index}
        id={`side-panel-tabpanel-${index}`}
        aria-labelledby={`side-panel-tab-${index}`}
        {...other}
        style={{ height: '100%' }} // Ensure panel takes full height
        >
        {value === index && (
            <Box sx={{ p: noPadding ? 0 : 2, height: '100%' }}>
                {children}
            </Box>
        )}
        </div>
    );
}

// Helper function for accessibility props
function a11yProps(index: number) {
    return {
        id: `side-panel-tab-${index}`,
        'aria-controls': `side-panel-tabpanel-${index}`,
    };
}

export const SidePanel: React.FC<SidePanelProps> = ({
    isPanelOpen,
    panelPosition,
    togglePanel,
    setPanelPosition,
    displayData,
    displayInfo,
}) => {
    // State to manage the active tab. 0 = Properties, 1 = Dataset, 2 = Charts
    const [activeTab, setActiveTab] = useState(1); // Default to "Dataset" tab

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setActiveTab(newValue);
    };

    // This now correctly chooses the icon based on all four positions
    const getCollapseIcon = () => {
        switch (panelPosition) {
            case 'left':
                return <ChevronLeft />;
            case 'right':
                return <ChevronRight />;
            case 'top':
                return <ExpandLess />;
            case 'bottom':
                return <ExpandMore />;
            default:
                return <ChevronLeft />;
        }
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 1, backgroundColor: 'background.paper' }}>
            {/* Your panel content goes here */}
            {/* Header with Tabs and Controls */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider' }}>
                {/* Tab Navigation */}
                <Tabs value={activeTab} onChange={handleTabChange} aria-label="side panel tabs" sx={{ flexGrow: 1 }}>
                    <Tab label="Properties" {...a11yProps(0)} />
                    <Tab label="Dataset" {...a11yProps(1)} />
                    <Tab label="Charts" {...a11yProps(2)} />
                </Tabs>

                {/* Docking and Collapse Controls */}
                <Box sx={{ display: 'flex', alignItems: 'center', pr: 1 }}>
                    <Tooltip title="Dock Top"><IconButton size="small" onClick={() => setPanelPosition('top')}><AlignVerticalTop fontSize="small" /></IconButton></Tooltip>
                    <Tooltip title="Dock Bottom"><IconButton size="small" onClick={() => setPanelPosition('bottom')}><AlignVerticalBottom fontSize="small" /></IconButton></Tooltip>
                    <Tooltip title="Dock Left"><IconButton size="small" onClick={() => setPanelPosition('left')}><AlignHorizontalLeft fontSize="small" /></IconButton></Tooltip>
                    <Tooltip title="Dock Right"><IconButton size="small" onClick={() => setPanelPosition('right')}><AlignHorizontalRight fontSize="small" /></IconButton></Tooltip>
                    <Tooltip title={isPanelOpen ? "Collapse Panel" : "Expand Panel"}><IconButton onClick={togglePanel} size="small" sx={{ ml: 1 }}>{getCollapseIcon()}</IconButton></Tooltip>
                </Box>
            </Box>

            {/* Tab Content Panels */}
            <Box sx={{ flexGrow: 1, overflowY: 'auto' }}>
                <TabPanel value={activeTab} index={0}>
                    <Typography>Properties content for the selected node will be displayed here.</Typography>
                </TabPanel>
                <TabPanel value={activeTab} index={1} noPadding>
                    {/* The DataGridDisplay now lives inside this tab panel */}
                    <DataGridDisplay
                        key={displayInfo?.["Data Points"]} 
                        data={displayData}
                        info={displayInfo} 
                        title="Features Data"
                    />
                </TabPanel>
                <TabPanel value={activeTab} index={2}>
                    <Typography>Charts and visualizations will be displayed here.</Typography>
                </TabPanel>
            </Box>
        </Box>
    );
};

// Custom styled component for the resize handle (remains the same)
export const StyledResizeHandle = styled(Box)(({ theme }) => ({
    width: '8px',
    height: '8px',
    backgroundColor: theme.palette.divider,
    position: 'relative',
    outline: 'none',
    '&[data-resize-handle-active], &:hover': {
        backgroundColor: theme.palette.primary.main,
    },
    '&[data-panel-group-direction="horizontal"]': {
        width: '8px',
        height: '100%',
        margin: '0 -4px',
        borderLeft: `1px solid ${theme.palette.background.default}`,
        borderRight: `1px solid ${theme.palette.background.default}`,
        cursor: 'col-resize',
    },
    '&[data-panel-group-direction="vertical"]': {
        height: '8px',
        width: '100%',
        margin: '-4px 0',
        borderTop: `1px solid ${theme.palette.background.default}`,
        borderBottom: `1px solid ${theme.palette.background.default}`,
        cursor: 'row-resize',
    },
}));