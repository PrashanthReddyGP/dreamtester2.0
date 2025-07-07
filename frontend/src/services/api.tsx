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
    // You might add other initial data here later
}

export interface BulkJobResponse {
    job_ids: string[];
}

export interface StrategyFilePayload {
    id: string;
    name: string;
    content: string;
}

/**
 * Submits a new backtest job to the backend.
 * @param config The configuration for the backtest, including the strategy code.
 * @returns The response from the server, typically including a job ID.
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
 * @param files An array of strategy file objects.
 * @returns A response from the server, likely containing job IDs for each submitted file.
 */
export const submitBatchBacktest = async (files: StrategyFilePayload[]): Promise<BulkJobResponse> => {
    // This will call a new endpoint designed for batch submissions
    const response = await fetch(`${API_URL}/api/backtest/batch-submit`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(files), // Send the array of files directly
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
};


// You can (and should) also move your other API calls here to centralize them!
// For example:
// export const getStrategies = async () => { ... };
// export const createStrategyItem = async (item) => { ... };