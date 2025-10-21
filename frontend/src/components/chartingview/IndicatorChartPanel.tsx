// src/components/chartingview/IndicatorChartPanel.tsx
import React, { useEffect, useRef } from 'react';
import { createChart, LineSeries, HistogramSeries, CrosshairMode } from 'lightweight-charts';
import type { IChartApi, ISeriesApi, UTCTimestamp, LogicalRange, MouseEventParams, Time } from 'lightweight-charts';
import { Box } from '@mui/material';

// (Props interface is unchanged)
interface IndicatorChartPanelProps {
    id: string; // New prop
    indicatorData: { time: number; [key: string]: any }[];
    indicatorConfig: { name: string, color: string };
    onTimeRangeChange: (range: LogicalRange | null, sourceId: string) => void; // Update signature
    onCrosshairMove: (params: MouseEventParams, sourceId: string) => void;
    syncedTimeRange: { range: LogicalRange | null; sourceId: string | null }; // Update signature
    syncedCrosshairTime: { time: Time | undefined; sourceId: string | null };
    panelSize?: number;
}

export const IndicatorChartPanel: React.FC<IndicatorChartPanelProps> = ({ id, indicatorData, indicatorConfig, onTimeRangeChange, onCrosshairMove, syncedTimeRange, syncedCrosshairTime, panelSize }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<any> | null>(null);
    const isSyncing = useRef(false); // To prevent feedback loops

    useEffect(() => {
        const container = chartContainerRef.current;
        if (!container) return;
        
        const chart = createChart(container, { 
            width: container.clientWidth, 
            height: container.clientHeight, 
            layout: { 
                background: { color: '#131722' }, 
                textColor: '#D1D4DC', 
                attributionLogo: false, // Disable the "TradingView" logo
            }, 
                grid: { vertLines: { color: '#2A2E39' }, horzLines: { visible: false } }, 
                crosshair: { mode: CrosshairMode.Normal }, 
                timeScale: { visible: false, borderVisible: false }, 
                rightPriceScale: { borderVisible: false }, 
            });
        chartRef.current = chart;

        chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
            if (isSyncing.current) {
                isSyncing.current = false;
                return;
            }
            onTimeRangeChange(range, id);
        });

        // This is a user move, so propagate it up with our ID
        chart.subscribeCrosshairMove(param => {
            onCrosshairMove(param, id);
        });
        
        return () => { chart.remove(); seriesRef.current = null; };
    }, [id, onTimeRangeChange, onCrosshairMove]);

    // --- NEW useEffect to sync the crosshair position ---
    useEffect(() => {
        const chart = chartRef.current;
        if (!chart) return;

        // If the update came from this chart, do nothing.
        if (syncedCrosshairTime.sourceId === id) {
            return;
        }

        // If time is undefined, clear the crosshair.
        if (syncedCrosshairTime.time === undefined) {
            chart.clearCrosshairPosition();
            return;
        }

        // Get this panel's indicator series to anchor the crosshair.
        const indicatorSeries = seriesRef.current;
        if (!indicatorSeries) return; // Don't do anything if the series isn't ready
        
        // Set the crosshair position.
        chart.setCrosshairPosition(0, syncedCrosshairTime.time, indicatorSeries);
        
    }, [id, syncedCrosshairTime]);

    useEffect(() => {
        const chart = chartRef.current;
        const container = chartContainerRef.current;
        if (chart && container) {
            chart.resize(container.clientWidth, container.clientHeight);
        }
    }, [panelSize]);

    useEffect(() => {
        const chart = chartRef.current;
        // Apply update only if this chart was NOT the source
        if (chart && syncedTimeRange.range && syncedTimeRange.sourceId !== id) {
            // Set the flag right before programmatically updating the range
            isSyncing.current = true;
            chart.timeScale().setVisibleLogicalRange(syncedTimeRange.range);
        }
    }, [id, syncedTimeRange]);


    useEffect(() => {
            const chart = chartRef.current;
            if (!chart || !indicatorData) {
                return; // Guard clause: exit if the chart isn't ready or there's no data
            }

            // 1. CLEANUP: If a series from a previous render exists, remove it.
            if (seriesRef.current) {
                chart.removeSeries(seriesRef.current);
            }

            let series; // A variable to hold the new series we're about to create

            // 2. CREATE: Check the indicator name to decide which type of series to create.
            if (indicatorConfig.name.toLowerCase() === 'volume') {
                // Create a Histogram for Volume
                series = chart.addSeries(HistogramSeries, {
                    title: indicatorConfig.name,
                    priceFormat: { type: 'volume' },
                    lastValueVisible: false, // Shows the last value on the price scale
                    priceLineVisible: false,  // Shows a dotted line for the last value
                });
                // Set data for the Histogram, which includes color
                series.setData(indicatorData.map(d => ({ time: d.time / 1000 as UTCTimestamp, value: d.value, color: d.color })));
            } else {
                // Create a Line Series for all other indicators (like RSI, ATR, etc.)
                series = chart.addSeries(LineSeries, {
                    title: indicatorConfig.name,
                    color: indicatorConfig.color,
                    lineWidth: 2,
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
                // Set data for the Line Series
                series.setData(indicatorData.map(d => ({ time: d.time / 1000 as UTCTimestamp, value: d.value })));
            }

            // 3. STORE: Save the reference to the newly created series in our ref.
            // This is crucial so that on the *next* render, we can find and remove it in step 1.
            seriesRef.current = series;

        }, [indicatorData, indicatorConfig]); // Dependencies: This whole process re-runs if data or config changes.


    return <Box ref={chartContainerRef} sx={{ height: '100%', width: '100%' }} />;
};