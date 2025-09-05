// frontend/src/services/api.tsx

// The base URL for your backend
const API_URL = 'http://127.0.0.1:8000';

// Define the shape of the backtest configuration object for type safety
export interface BacktestConfig {
    strategyCode: string;
    asset: string;
    timeframe: string;
    startDate: string | null; // Use string for dates to send as JSON
    endDate: string | null;
}

// Define the expected response (for now, a job ID)
export interface BacktestJobResponse {
    job_id: string;
}

// export interface BulkJobResponse {
//     job_ids: string[];
// }

export interface StrategyFilePayload {
    id: string;
    name: string;
    content: string;
}

// Define the shape of the expected result payload for type safety
// This should match the structure you create in `run_batch_manager`
export type EquityCurvePoint = [number, number]; // e.g., [1688673600000, 10500.50]

export type StrategyMetrics = {
    Net_Profit: number;
    Gross_Profit: number;
    Profit_Percentage: number;
    Annual_Return: number;
    Avg_Monthly_Return: number;
    Total_Trades: number;
    Open_Trades: number;
    Closed_Trades: number;
    Max_Drawdown: number;
    Avg_Drawdown: number;
    Profit_Factor: number;
    Sharpe_Ratio: number;
    Calmar_Ratio: number;
    Equity_Efficiency_Rate: number;
    Strategy_Quality: string;
    Max_Drawdown_Duration_days: number;
    Total_Wins: number;
    Total_Losses: number;
    Consecutive_Wins: number;
    Consecutive_Losses: number;
    Largest_Win: number;
    Largest_Loss: number;
    Avg_Win: number;
    Avg_Loss: number;
    Avg_Trade: number;
    Avg_Trade_Time: string;
    Avg_Win_Time: string;
    Avg_Loss_Time: string;
    Max_Runup: number;
    Avg_Runup: number;
    Winrate: number;
    RR: number;
    Max_Open_Trades: number;
    Avg_Open_Trades: number;
    Commission: number
};

export interface Trades {
    [key: string]: any;
}

export interface MonthlyReturns {
    Month: string;
    'Profit ($)': number;
    'Returns (%)': number;
}


export interface StrategyResult {
    strategy_name: string;
    equity_curve: EquityCurvePoint[];
    metrics: StrategyMetrics; 
    trades: Trades[];
    monthly_returns: MonthlyReturns;
}

export interface EquityType {
    strategy_name: string;
    equity_curve: EquityCurvePoint[];
}
export interface MetricsType {
    strategy_name: string;
    metrics: StrategyMetrics;
    monthly_returns: MonthlyReturns;
}
export interface TradesType {
    strategy_name: string;
    trades: Trades[];
}

export interface BacktestResultPayload {
    name: string;
    strategies_results: StrategyResult[];
    initial_capital: number;
}

export interface BatchSubmitResponse {
    message: string;
    batch_id: string;
}

export interface SubmissionResponse {
    message: string;
    batch_id: string;
}


// --- Define the ML-specific analysis types ---
export interface FeatureImportance {
    feature: string;
    importance: number;
}

export interface ClassificationReport {
  // e.g., "class_-1", "class_0", "class_1", "accuracy", "macro_avg"
    [key: string]: {
        precision: number;
        recall: number;
        'f1-score': number;
        support: number;
    } | number; // For accuracy
}

export interface ModelAnalysis {
    feature_importance: FeatureImportance[];
    classification_report: ClassificationReport;
    confusion_matrix: number[][];
}


// --- The main ML Result interface ---
// It's like a regular StrategyResult, but with an added `model_analysis` section.
export interface MLResult extends StrategyResult {
    strategy_name: 'ML Model'; // The name is fixed for an ML run
    model_analysis: ModelAnalysis;
    // We can also add the run config for display purposes
    run_config: {
        model: string;
        features: string[];
        labeling_method: string;
        symbol: string;
        timeframe: string;
    }
}


/**
 * Submits a new backtest job to the backend.
 */
export const submitBacktest = async (config: BacktestConfig): Promise<BacktestJobResponse> => {
    // We'll use the asynchronous job submission pattern we discussed.
    const response = await fetch(`${API_URL}/api/backtest/submit`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
    });

    if (!response.ok) {
        // Try to parse error details from the backend for better feedback
        const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
};


/**
 * Submits a batch of strategy files for backtesting.
 */
export const submitBatchBacktest = async (files: StrategyFilePayload[]): Promise<BatchSubmitResponse> => { 
    // This will call a new endpoint designed for batch submissions
    const response = await fetch(`${API_URL}/api/backtest/batch-submit`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(files),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
};


export const fetchAvailableSymbols = async (exchange: string): Promise<string[]> => {
    try {
        const response = await fetch(`${API_URL}/api/exchange/symbols/${exchange}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch symbols from server.`);
        }
        const symbols: string[] = await response.json();
        // ccxt returns symbols like "BTC/USDT", which is perfect.
        return symbols;
    } catch (error) {
        console.error("Error fetching available symbols:", error);
        // Return a fallback list on error so the UI doesn't crash
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'];
    }
};


export const clearOhlcvCache = async (): Promise<{ status: string; message: string }> => {
  const response = await fetch(`${API_URL}/api/data/cache`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to clear cache on the server.');
  }

  return response.json();
};


// /**
//  * Fetches the most recently completed backtest result.
//  */
// export const getLatestBacktestResult = async (): Promise<BacktestResultPayload | null> => {
//     const response = await fetch(`${API_URL}/api/backtest/latest`);
    
//     if (!response.ok) {
//         throw new Error("Failed to fetch latest result.");
//     }
    
//     const data = await response.json();
    
//     // If the backend returns an empty object, it means no result is ready yet.
//     if (!data || Object.keys(data).length === 0) {
//         return null;
//     }
    
//     return data;
// };

// export const clearLatestBacktestResult = async (): Promise<void> => {
//     try {
//         const response = await fetch(`${API_URL}/api/backtest/latest`, {
//             method: 'DELETE',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//         });

//         // If the response is not ok (e.g., 500 server error), throw an error.
//         if (!response.ok) {
//             const errorData = await response.json().catch(() => ({})); // Try to get error detail
//             throw new Error(errorData.detail || `Server responded with status: ${response.status}`);
//         }

//         console.log("Successfully cleared previous backtest result on the server.");
//         // The function resolves without a value, so the return type is Promise<void>
        
//     } catch (error) {
//         console.error("Error clearing latest backtest result:", error);
//         // Re-throw the error so the calling function (handleRunBacktest) can catch it.
//         throw error;
//     }
// };

// You can (and should) also move your other API calls here to centralize them!
// For example:
// export const getStrategies = async () => { ... };
// export const createStrategyItem = async (item) => { ... };
