// src/components/analysishub/CorrelationMatrixTab.tsx

import React, { useMemo, useState } from 'react';
import type { FC } from 'react';
import { Box, Typography, useTheme, Switch, FormControlLabel } from '@mui/material';
import ReactECharts from 'echarts-for-react';
import { sampleCorrelation } from 'simple-statistics';

import type { StrategyResult } from '../../services/api';

// --- HELPER FUNCTION FOR DYNAMIC TEXT COLOR ---
/**
 * Calculates the perceived luminance of a color.
 * @param hex The color in hex format (e.g., "#RRGGBB").
 * @returns A value between 0 (black) and 1 (white).
 */
const getLuminance = (hex: string): number => {
    // Remove the '#' if present
    const color = hex.substring(1); 
    const rgb = parseInt(color, 16);
    const r = (rgb >> 16) & 0xff;
    const g = (rgb >> 8) & 0xff;
    const b = (rgb >> 0) & 0xff;
    
    // Standard luminance formula
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};


const getDailyPnlByStrategy = (results: StrategyResult[]): Record<string, Record<string, number>> => {
    const dailyPnl: Record<string, Record<string, number>> = {};

    results.forEach(result => {
        if (result.strategy_name === 'Portfolio' || !result.trades) {
            return;
        }

        const pnlForStrategy: Record<string, number> = {};
        result.trades.forEach(trade => {
            if (!trade.Exit_Time || trade.Result === undefined) return;
            const dateStr = new Date(trade.Exit_Time * 1000).toISOString().split('T')[0];
            pnlForStrategy[dateStr] = (pnlForStrategy[dateStr] || 0) + trade.Result;
        });
        dailyPnl[result.strategy_name] = pnlForStrategy;
    });

    return dailyPnl;
};


export const CorrelationMatrixTab: FC<{ results: StrategyResult[] }> = ({ results }) => {
    const theme = useTheme();
    const [showLabels, setShowLabels] = useState(true);

    const { chartOptions, hasEnoughData } = useMemo(() => {
        const strategiesWithTrades = results.filter(
            r => r.strategy_name !== 'Portfolio' && r.trades && r.trades.length > 0
        );

        if (strategiesWithTrades.length < 2) {
            return { chartOptions: {}, hasEnoughData: false };
        }

        const dailyPnlByStrategy = getDailyPnlByStrategy(strategiesWithTrades);
        const strategyNames = Object.keys(dailyPnlByStrategy);
        const allDates = new Set<string>();
        Object.values(dailyPnlByStrategy).forEach(pnlMap => {
            Object.keys(pnlMap).forEach(date => allDates.add(date));
        });
        const sortedDates = Array.from(allDates).sort();
        
        const alignedPnlSeries: Record<string, number[]> = {};
        strategyNames.forEach(name => {
            alignedPnlSeries[name] = sortedDates.map(date => dailyPnlByStrategy[name][date] || 0);
        });

        const heatmapData: [number, number, number][] = [];
        for (let i = 0; i < strategyNames.length; i++) {
            for (let j = 0; j < strategyNames.length; j++) {
                const seriesA = alignedPnlSeries[strategyNames[i]];
                const seriesB = alignedPnlSeries[strategyNames[j]];
                const corr = sampleCorrelation(seriesA, seriesB);
                const finalCorr = isNaN(corr) ? 0 : parseFloat(corr.toFixed(2));
                heatmapData.push([i, j, finalCorr]);
            }
        }
        
        const colors = {
            very_negative: '#141414ff',
            mid_negative: '#a03d3dff',
            negative: '#9e4b4bff', // Warm Amber/Brown-Grey
            neutral: theme.palette.background.default,
            positive: '#3d9565ff', // Cool Slate-Blue
            mid_positive: '#25ba5bff',
            very_positive: '#141414ff'
        };

        const options = {
            grid: {
                height: '85%',
                top: '5%',
                bottom: '10%',
                left: '10%',
                right: '5%'
            },
            tooltip: {
                position: 'top',
                formatter: (params: any) => {
                    const nameX = strategyNames[params.data[0]];
                    const nameY = strategyNames[params.data[1]];
                    return `${nameX.replace('.py','')} vs ${nameY.replace('.py','')}<br/><strong>Correlation: ${params.data[2]}</strong>`;
                },
                backgroundColor: theme.palette.background.paper,
                borderColor: theme.palette.divider,
                borderWidth: 1,
                textStyle: {
                    color: theme.palette.text.primary,
                    fontWeight: 'normal'
                },
                extraCssText: 'box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);'
            },
            xAxis: {
                type: 'category',
                data: strategyNames.map(name => name.replace('.py','')),
                splitArea: { show: false }, 
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: {
                    color: theme.palette.text.secondary
                }
            },
            yAxis: {
                type: 'category',
                data: strategyNames.map(name => name.replace('.py','')),
                splitArea: { show: false },
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: {
                    color: theme.palette.text.secondary
                }
            },
            visualMap: {
                min: -1,
                max: 1,
                show: false,
                calculable: true,
                orient: 'vertical',
                left: 'right',
                bottom: '4%',
                inRange: {
                    color: [colors.very_negative, colors.mid_negative, colors.negative, colors.neutral, colors.positive,colors.mid_positive, colors.very_positive]
                },
                textStyle: {
                    color: theme.palette.text.secondary,
                }
            },
            series: [{
                name: 'Correlation',
                type: 'heatmap',
                data: heatmapData,
                label: {
                    show: showLabels,
                    color: 'hsla(0, 0%, 100%, 0.35)'
                    // --- DYNAMIC LABEL CONTRAST ---
                    // color: (params: any) => {
                    //     // params.color is the background color of the cell provided by echarts
                    //     // Use a threshold of 128 for luminance (0-255 scale)
                    //     return getLuminance(params.color) > 128 ? '#111' : '#fff';
                    // }
                },
                // --- REFINED HOVER EFFECT ---
                emphasis: {
                    itemStyle: {
                        borderColor: '#fff',
                        borderWidth: 2,
                        shadowBlur: 0
                    }
                }
            }]
        };

        return { chartOptions: options, hasEnoughData: true };
    }, [results, theme, showLabels]);

    if (!hasEnoughData) {
        return (
            <Box sx={{ p: 4, textAlign: 'center', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography variant="body1" color="text.secondary">
                    At least two strategies with trades are required to calculate correlation.
                </Typography>
            </Box>
        );
    }
    
    return (
        <Box sx={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', pr: 2 }}>
                <FormControlLabel
                    control={
                        <Switch
                            checked={showLabels}
                            onChange={(event) => setShowLabels(event.target.checked)}
                            size="small"
                        />
                    }
                    label="Show Values"
                    labelPlacement="start"
                />
            </Box>
            <Box sx={{ flex: 1, minHeight: 0 }}>
                <ReactECharts
                    option={chartOptions}
                    style={{ height: '100%', width: '100%' }}
                    notMerge={true}
                    lazyUpdate={true}
                />
            </Box>
        </Box>
    );
};