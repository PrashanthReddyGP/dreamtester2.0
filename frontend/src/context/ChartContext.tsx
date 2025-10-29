import React, { createContext, useState, useContext, useCallback, useMemo } from 'react';
import type { ReactNode } from 'react';
import type { IndicatorConfig } from '../components/chartingview/ChartSettingsPanel';
import type { DataSourceConfig } from '../components/strategylab/SettingsPanel';

const API_URL = 'http://127.0.0.1:8000';

// A flexible type for any indicator data point. It MUST have a time, but can have other properties.
export type IndicatorDataPoint = {
  time: number; // The backend sends a millisecond timestamp
  [key: string]: any; // Allows for 'value', 'upper', 'lower', etc.
};

export interface ChartDataPayload {
    ohlcv: [number, number, number, number, number, number][]; // [timestamp, o, h, l, c, v]
    indicators: Record<string, IndicatorDataPoint[]>; 
    strategy_name: string;
}
// --- 1. EXPAND THE CONTEXT STATE SHAPE ---
interface ChartContextState {
    indicatorConfigs: Record<string, IndicatorConfig>;
    setIndicatorConfigs: React.Dispatch<React.SetStateAction<Record<string, IndicatorConfig>>>;

    chartData: ChartDataPayload | null;
    isChartDataLoading: boolean;
    fetchChartData: (strategyCode: string, dataSourceConfig: DataSourceConfig) => Promise<void>;
    clearChartData: () => void;
}

const ChartContext = createContext<ChartContextState | undefined>(undefined);

interface ChartContextProviderProps {
    children: ReactNode;
}

export const ChartContextProvider: React.FC<ChartContextProviderProps> = ({ children }) => {
    // --- UI State ---
    const [indicatorConfigs, setIndicatorConfigs] = useState<Record<string, IndicatorConfig>>({});

    // --- DATA STATE (MOVED FROM AnalysisContext) ---
    const [chartData, setChartData] = useState<ChartDataPayload | null>(null);
    const [isChartDataLoading, setIsChartDataLoading] = useState<boolean>(false);

    // --- LOGIC (MOVED FROM AnalysisContext) ---
    const clearChartData = useCallback(() => {
        setChartData(null);
        // Also clear indicator configs when chart data is cleared
        setIndicatorConfigs({});
    }, []);

    const fetchChartData = useCallback(async (strategyCode: string, dataSourceConfig: DataSourceConfig) => {
        setIsChartDataLoading(true);
        setChartData(null); // Clear previous data immediately

        try {
            const response = await fetch(`${API_URL}/api/charting/prepare-data`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: strategyCode,
                    config: dataSourceConfig,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to fetch chart data.');
            }

            const data: ChartDataPayload = await response.json();
            setChartData(data);

        } catch (error) {
            console.error("Error fetching chart data:", error);
            alert(`Error preparing chart: ${error instanceof Error ? error.message : 'Unknown error'}`);
        } finally {
            setIsChartDataLoading(false);
        }
    }, []);


    const value = useMemo(() => ({
        indicatorConfigs,
        setIndicatorConfigs,
        chartData,
        isChartDataLoading,
        fetchChartData,
        clearChartData,
    }), [
        indicatorConfigs, 
        chartData, 
        isChartDataLoading, 
        fetchChartData, 
        clearChartData
    ]);


    return (
        <ChartContext.Provider value={value}>
            {children}
        </ChartContext.Provider>
    );
};

export const useChart = (): ChartContextState => {
    const context = useContext(ChartContext);
    if (context === undefined) {
        throw new Error('useChart must be used within a ChartContextProvider');
    }
    return context;
};