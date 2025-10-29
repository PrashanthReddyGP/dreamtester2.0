// src/components/chartingview/TradingChartPanel.tsx
import React, { useEffect, useRef } from 'react';
import { createChart, CandlestickSeries, LineSeries, CrosshairMode } from 'lightweight-charts';
import type { IChartApi, ISeriesApi, UTCTimestamp, Time, LogicalRange, MouseEventParams, SeriesDataItemTypeMap } from 'lightweight-charts';
import { Box } from '@mui/material';
import type { ChartDataPayload } from '../../context/ChartContext';
import type { IndicatorConfig } from './ChartSettingsPanel';
import { ChartLegend } from './ChartLegend';
import type { LegendData } from './ChartLegend';

// (Props interface and helper functions)
interface TradingChartPanelProps {
    id: string; // New prop for identification
    chartData: ChartDataPayload;
    overlayIndicatorConfigs: Record<string, IndicatorConfig>;
    onCrosshairMove: (params: MouseEventParams, sourceId: string, legend: LegendData[]) => void;
    legendData: LegendData[];
    onTimeRangeChange: (range: LogicalRange | null, sourceId: string) => void; // Update signature
    syncedTimeRange: { range: LogicalRange | null; sourceId: string | null }; // Update signature
    syncedCrosshairTime: { time: Time | undefined; sourceId: string | null };
    panelSize?: number;
}

const formatLegendNumber = (value: number | undefined): string => {
    if (value === undefined) return 'N/A';
    // Large number formatting (for volume, etc.)
    if (value > 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
    if (value > 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
    if (value > 10_000) return `${(value / 1_000).toFixed(1)}K`;

    // High precision for instruments like Forex (e.g., < 100)
    if (Math.abs(value) < 100) {
        return value.toFixed(5);
    }
    // Standard precision for stocks
    return value.toFixed(2);
};

export const TradingChartPanel: React.FC<TradingChartPanelProps> = ({ id, chartData, overlayIndicatorConfigs, onCrosshairMove, legendData, onTimeRangeChange, syncedTimeRange, syncedCrosshairTime, panelSize }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<{ [key: string]: ISeriesApi<any> | ISeriesApi<any>[] }>({});
    const isSyncing = useRef(false); // To prevent feedback loops

    const formatCandlestickData = (rawData: any[]) => {
        return rawData.map(d => ({ time: d[0] / 1000 as UTCTimestamp, open: d[1], high: d[2], low: d[3], close: d[4], }));
    };

    useEffect(() => {
        const container = chartContainerRef.current;
        if (!container) return;

        const chart = createChart(
            container, 
            { 
                width: container.clientWidth, 
                height: container.clientHeight, 
                layout: {
                    background: { color: '#131722' }, 
                    textColor: '#D1D4DC',
                    attributionLogo: false, // Disable the "TradingView" logo
                }, 
                grid: {
                    vertLines: { color: '#2A2E39' }, 
                    horzLines: { color: '#2A2E39' } 
                }, 
                crosshair: { mode: CrosshairMode.Normal }, 
                rightPriceScale: { borderColor: '#485158' }, 
                timeScale: { borderColor: '#485158', timeVisible: true, secondsVisible: false },
            });
            
        chartRef.current = chart;

        chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
            if (isSyncing.current) {
                isSyncing.current = false;
                return;
            }
            onTimeRangeChange(range, id);
        });


        chart.subscribeCrosshairMove((param: MouseEventParams) => {
            const newLegendData: LegendData[] = [];
            if (param.seriesData) {
                for (const [series, data] of param.seriesData.entries()) {
                    const title = series.options().title || ''; const seriesType = series.seriesType();
                    if (seriesType === 'Candlestick') { const ohlc = data as SeriesDataItemTypeMap['Candlestick']; if (!ohlc) continue; newLegendData.push({ name: 'O', value: formatLegendNumber(ohlc.open), color: '#9B9EA3' }); newLegendData.push({ name: 'H', value: formatLegendNumber(ohlc.high), color: '#9B9EA3' }); newLegendData.push({ name: 'L', value: formatLegendNumber(ohlc.low), color: '#9B9EA3' }); newLegendData.push({ name: 'C', value: formatLegendNumber(ohlc.close), color: '#9B9EA3' });
                    } else if (seriesType === 'Line') { const singleValueData = data as SeriesDataItemTypeMap['Line']; if (singleValueData?.value === undefined) continue; const color = (series.options() as any).color || '#FFF'; newLegendData.push({ name: title, value: formatLegendNumber(singleValueData.value), color }); }
                }
            }
            onCrosshairMove(param, id, newLegendData);
        });

        return () => { chart.remove(); seriesRef.current = {}; };

    }, [id, onTimeRangeChange, onCrosshairMove]);

    useEffect(() => {
        const chart = chartRef.current;
        const container = chartContainerRef.current;
        if (chart && container) {
            chart.resize(container.clientWidth, container.clientHeight);
        }
    }, [panelSize]);

    // --- NEW useEffect to sync the crosshair position ---
    useEffect(() => {
        const chart = chartRef.current;
        if (!chart) return;

        // If the update came from this chart, do nothing.
        if (syncedCrosshairTime.sourceId === id) {
            return;
        }

        // If time is undefined, it means the mouse left the chart area. Clear the crosshair.
        if (syncedCrosshairTime.time === undefined) {
            chart.clearCrosshairPosition();
            return;
        }

        // Get the main candlestick series to anchor the crosshair to.
        const mainSeries = seriesRef.current['main'] as ISeriesApi<'Candlestick'>;
        if (!mainSeries) return; // Don't do anything if the series isn't ready

        // Set the crosshair position. The price '0' is a placeholder;
        // the 'time' is what matters for the vertical line sync.
        chart.setCrosshairPosition(0, syncedCrosshairTime.time, mainSeries);

    }, [id, syncedCrosshairTime]); // Dependency array is correct

    useEffect(() => {
        const chart = chartRef.current;
        // Apply update only if this chart was NOT the source
        if (chart && syncedTimeRange.range && syncedTimeRange.sourceId !== id) {
            // Set the flag right before programmatically updating the range
            isSyncing.current = true;
            chart.timeScale().setVisibleLogicalRange(syncedTimeRange.range);
        }
    }, [id, syncedTimeRange]);

    // Effect 1: Handles the main candlestick series and resets zoom.
    // This runs ONLY when chartData itself is replaced.
    useEffect(() => {
        const chart = chartRef.current;
        if (!chart || !chartData) return;
        
        // When new data arrives, we assume a full reset. Clear old main series.
        if (seriesRef.current['main']) {
            chart.removeSeries(seriesRef.current['main'] as ISeriesApi<"Candlestick">);
        }

        const isHighPrecision = chartData.ohlcv.length > 0 && chartData.ohlcv[0][4] < 100;
        const priceFormat = isHighPrecision
            ? { precision: 5, minMove: 0.00001 }
            : { precision: 2, minMove: 0.01 };

        // Add the main candlestick series
        const candleSeries = chart.addSeries(CandlestickSeries, {
            title: chartData.strategy_name, 
            upColor: '#26a69a', 
            downColor: '#ef5350',
            lastValueVisible: false,
            priceLineVisible: false,
            priceFormat: priceFormat,
        });
        candleSeries.setData(formatCandlestickData(chartData.ohlcv));
        seriesRef.current['main'] = candleSeries;

        // Reset the zoom to fit the new data
        if (chartData.ohlcv.length > 0) {
            chart.timeScale().fitContent();
        }

    }, [chartData]); // <-- DEPENDENCY IS ONLY chartData

    // Effect 2: Manages overlay indicators.
    // This runs when configs change, but DOES NOT reset zoom.
    useEffect(() => {
        const chart = chartRef.current;
        if (!chart || !chartData) return;

        // 1. Clean up ALL existing overlay series from the chart
        Object.keys(seriesRef.current).forEach(key => {
            if (key !== 'main') {
                const series = seriesRef.current[key];
                // Check if series is an array or single object and remove
                (Array.isArray(series) ? series : [series]).forEach(s => {
                    if(s) chart.removeSeries(s);
                });
                delete seriesRef.current[key];
            }
        });
        
        // 2. Add the currently visible overlay indicators
        Object.values(overlayIndicatorConfigs).forEach(config => {
            if (config.isVisible) {
                const data = chartData.indicators[config.name];
                if(data) {
                    const lineSeries = chart.addSeries(LineSeries, { 
                        title: config.name, 
                        color: config.color, 
                        lineWidth: 2,
                        lastValueVisible: false,
                        priceLineVisible: false,
                    });
                    lineSeries.setData(data.map(d => ({ time: d.time / 1000 as UTCTimestamp, value: d.value })));
                    seriesRef.current[config.name] = lineSeries;
                }
            }
        });

    }, [chartData, overlayIndicatorConfigs]); // <-- DEPENDS on configs and data

    return (
        <Box sx={{ height: '100%', width: '100%', position: 'relative' }}>
            <ChartLegend legendData={legendData} />
            <Box ref={chartContainerRef} sx={{ height: '100%', width: '100%' }} />
        </Box>
    );
};