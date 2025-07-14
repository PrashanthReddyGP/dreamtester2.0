// src/context/types.ts
import type { BacktestResultPayload } from '../services/api'; // Import the type

export interface ApiKeySet {
  exchange: string; // <-- ADD THIS
  apiKey: string;
  apiSecret: string;
}

// ... rest of the file is the same ...
export interface SettingsState {
  [exchange: string]: Omit<ApiKeySet, 'exchange'> | undefined; // Use Omit to not store redundant exchange name
}

export interface AppContextType {
  settings: SettingsState;
  isLoading: boolean;
  error: string | null;
  saveApiKeys: (exchange: string, apiKey: string, apiSecret: string) => Promise<any>;

  // --- ADD NEW PROPERTIES FOR ANALYSIS DATA ---
  latestBacktest: BacktestResultPayload | null;
  isBacktestLoading: boolean;
  backtestError: string | null;
  fetchLatestResults: () => void; // A function to trigger the fetch/poll
  clearLatestBacktest: () => void; // A function to trigger the fetch/poll
}