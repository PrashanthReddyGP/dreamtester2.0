export interface IndicatorParamDef {
  name: string;
  type: 'number';
  defaultValue: number;
}

export interface IndicatorDefinition {
  name: string;
  params: IndicatorParamDef[];
}

// This object is the single source of truth for all available indicators.
// The key (e.g., 'RSI') is a unique identifier.
export const INDICATOR_DEFINITIONS: { [key: string]: IndicatorDefinition } = {
  RSI: {
    name: 'Relative Strength Index (RSI)',
    params: [{ name: 'length', type: 'number', defaultValue: 14 }],
  },
  MACD: {
    name: 'Moving Average Convergence Divergence (MACD)',
    params: [
      { name: 'fast', type: 'number', defaultValue: 12 },
      { name: 'slow', type: 'number', defaultValue: 26 },
      { name: 'signal', type: 'number', defaultValue: 9 },
    ],
  },
  ATR: {
    name: 'Average True Range (ATR)',
    params: [{ name: 'length', type: 'number', defaultValue: 14 }],
  },
  ADX: {
    name: 'Average Directional Index (ADX)',
    params: [{ name: 'length', type: 'number', defaultValue: 14 }],
  },
  SMA: {
    name: 'Simple Moving Average (SMA)',
    params: [{ name: 'length', type: 'number', defaultValue: 50 }],
  },
  BBANDS: {
    name: 'Bollinger Bands',
    params: [
      { name: 'length', type: 'number', defaultValue: 20 },
      { name: 'std', type: 'number', defaultValue: 2 },
    ],
  },
};