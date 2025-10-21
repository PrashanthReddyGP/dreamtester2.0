// src/views/ChartingView.tsx
import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { TradingChartPanel } from '../components/chartingview/TradingChartPanel';
import { IndicatorChartPanel } from '../components/chartingview/IndicatorChartPanel';
import { ChartSettingsPanel } from '../components/chartingview/ChartSettingsPanel';
import type { IndicatorConfig } from '../components/chartingview/ChartSettingsPanel';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import type { LegendData } from '../components/chartingview/ChartLegend';
import type { LogicalRange, Time, MouseEventParams } from 'lightweight-charts';
import { useChart } from '../context/ChartContext';

const COLOR_PALETTE = [
    '#2962FF', '#E91E63', '#673AB7', '#FF9800', '#4CAF50', 
    '#00BCD4', '#FF5722', '#9C27B0', '#3F51B5', '#8BC34A', 
    '#03A9F4', '#FFEB3B', '#795548', '#607D8B', '#e2e2e2ff'
];

const isOverlayIndicator = (name: string): boolean => {
    const lowerCaseName = name.toLowerCase();
    return lowerCaseName.startsWith('sma') || lowerCaseName.startsWith('ema') || lowerCaseName.startsWith('zlema') || lowerCaseName.startsWith('bollinger') || lowerCaseName.startsWith('donchain') || lowerCaseName.startsWith('stoploss') || lowerCaseName.startsWith('supertrend') || lowerCaseName.startsWith('conversion_line') || lowerCaseName.startsWith('base_line') || lowerCaseName.startsWith('leading_span') || lowerCaseName.startsWith('lagging_span');
};

const ResizeHandle = () => (<PanelResizeHandle style={{ height: '4px', background: '#333', cursor: 'row-resize' }} />);

export const ChartingView: React.FC = () => {
    const { chartData, isChartDataLoading, indicatorConfigs, setIndicatorConfigs } = useChart();

    const [legendData, setLegendData] = useState<LegendData[]>([]);
    const [syncedTimeRange, setSyncedTimeRange] = useState<{ range: LogicalRange | null; sourceId: string | null }>({
        range: null,
        sourceId: null,
    });

    const [syncedCrosshairTime, setSyncedCrosshairTime] = useState<{ time: Time | undefined, sourceId: string | null }>({
        time: undefined,
        sourceId: null,
    });

    const [layout, setLayout] = useState<number[]>([]);

    useEffect(() => {
        if (!chartData?.indicators) {
             // If chartData is null when the component loads, we should clear any stale configs
            if(Object.keys(indicatorConfigs).length > 0) {
                setIndicatorConfigs({});
            }
            return;
        }

        const allIndicatorNames = ['Volume', ...Object.keys(chartData.indicators)];
        
        const newConfigs: Record<string, IndicatorConfig> = {};
        allIndicatorNames.forEach((name, index) => {
            const existingConfig = indicatorConfigs[name]; 
            newConfigs[name] = {
                name,
                isVisible: existingConfig?.isVisible ?? (name.toLowerCase() === 'volume'),
                color: existingConfig?.color ?? COLOR_PALETTE[index % COLOR_PALETTE.length],
            };
        });

        // To prevent unnecessary re-renders, only update if the configs have actually changed
        if (JSON.stringify(newConfigs) !== JSON.stringify(indicatorConfigs)) {
            setIndicatorConfigs(newConfigs);
        }

    // We only want this to run when the data itself changes, not when configs are toggled.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [chartData, setIndicatorConfigs]);


    const handleTimeRangeChange = useCallback((range: LogicalRange | null, sourceId: string) => {
        setSyncedTimeRange({ range, sourceId });
    }, []);

    const handleCrosshairMove = useCallback(
        (
            param: MouseEventParams,
            sourceId: string,
            legendUpdate?: LegendData[] // Optional: only the main chart provides this
        ) => {
            // Update the synced time, marking which chart it came from
            setSyncedCrosshairTime({ time: param.time, sourceId });

            // If the mouse is off the chart, clear the legend
            if (param.time === undefined) {
                setLegendData([]);
                return;
            }
            
            // The main chart is responsible for updating the rich legend data
            if (legendUpdate) {
                setLegendData(legendUpdate);
            }
        },
        []
    );

    const handleConfigChange = useCallback((name: string, newConfig: Partial<IndicatorConfig>) => {
        setIndicatorConfigs(prev => ({ ...prev, [name]: { ...prev[name], ...newConfig } }));
    }, []);

    const { overlayIndicators, paneIndicators } = useMemo(() => {
        const overlays: Record<string, IndicatorConfig> = {};
        const panes: Record<string, IndicatorConfig> = {};
        Object.values(indicatorConfigs).forEach(config => {
            if (isOverlayIndicator(config.name)) {
                overlays[config.name] = config;
            } else if (config.isVisible) {
                panes[config.name] = config;
            }
        });
        return { overlayIndicators: overlays, paneIndicators: panes };
    }, [indicatorConfigs]);

    const formattedVolumeData = useMemo(() => {
        if (!chartData) return [];
        return chartData.ohlcv.map(d => ({ time: d[0], value: d[5], color: d[4] > d[1] ? 'rgba(0, 150, 136, 0.7)' : 'rgba(255, 82, 82, 0.7)', }));
    }, [chartData]);


    // Condition 1: The data is currently being fetched
    if (isChartDataLoading) {
        return (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', width: '100vw' }}>
                <CircularProgress />
                <Typography variant="h6">Preparing Chart Data...</Typography>
                <Typography variant="body1" color="text.secondary">
                    Fetching OHLCV and calculating indicators.
                </Typography>
            </Box>
        );
    }
    
    // Condition 2: Loading is finished, but we have no data (or an error occurred)
    if (!chartData) {
        return (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', width: '100vw' }}>
                <Typography variant="h5">No Chart Data</Typography>
                <Typography variant="body1" color="text.secondary">
                    Please generate chart data from the Strategy Lab.
                </Typography>
            </Box>
        );
    }
    
    // --- DYNAMICALLY CALCULATE PANEL SIZES ---
    const visiblePaneCount = Object.keys(paneIndicators).length;
    const mainPanelSize = visiblePaneCount > 0 ? 100 - (visiblePaneCount * 20) : 100; // e.g. give each indicator 20%
    const indicatorPanelSize = visiblePaneCount > 0 ? (100 - mainPanelSize) / visiblePaneCount : 0;

    return (      
        <Box sx={{ height: `calc(100vh - 88px)`, width: '100vw' }}>
            <PanelGroup direction="horizontal">
                <Panel defaultSize={85} minSize={50} style={{ height: '100%' }}>
                    <PanelGroup direction="vertical" onLayout={setLayout}>
                        <Panel defaultSize={mainPanelSize} minSize={30}>
                            <TradingChartPanel
                                id="main-chart" // Give it a unique ID
                                chartData={chartData}
                                overlayIndicatorConfigs={overlayIndicators}
                                onCrosshairMove={handleCrosshairMove}
                                legendData={legendData}
                                onTimeRangeChange={handleTimeRangeChange}
                                syncedTimeRange={syncedTimeRange} // Pass the whole sync object
                                syncedCrosshairTime={syncedCrosshairTime} // Pass new state
                                panelSize={layout[0]}
                            />
                        </Panel>
                            {Object.values(paneIndicators).map((config, index) => (
                            <React.Fragment key={config.name}>
                                <ResizeHandle />
                                <Panel defaultSize={indicatorPanelSize} minSize={10}>
                                    <IndicatorChartPanel
                                        id={`indicator-chart-${config.name}`} // Give it a unique ID
                                        indicatorData={config.name === 'Volume' ? formattedVolumeData : chartData.indicators[config.name]}
                                        indicatorConfig={config}
                                        onTimeRangeChange={handleTimeRangeChange}
                                        onCrosshairMove={handleCrosshairMove}
                                        syncedTimeRange={syncedTimeRange} // Pass the whole sync object
                                        syncedCrosshairTime={syncedCrosshairTime} // Pass new state
                                        panelSize={layout[index + 1]}
                                    />
                                </Panel>
                            </React.Fragment>
                        ))}
                    </PanelGroup>
                </Panel>
                <PanelResizeHandle style={{ width: '4px', background: '#333' }} />
                <Panel defaultSize={15} minSize={15}>
                    <ChartSettingsPanel configs={indicatorConfigs} onConfigChange={handleConfigChange} />
                </Panel>
            </PanelGroup>
        </Box>
    );
};