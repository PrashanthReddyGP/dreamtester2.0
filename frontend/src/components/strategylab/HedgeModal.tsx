import React, { useState, useEffect, useRef } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Box,
    TextField,
    Typography,
    CircularProgress,
    Alert,
    IconButton,
    Divider,
    Autocomplete,
    Chip,
    Select,
    MenuItem,
    FormControlLabel,
    Checkbox,
    ToggleButton,
    ToggleButtonGroup,
    Slider
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import { v4 as uuidv4 } from 'uuid';
import { fetchAvailableSymbols } from '../../services/api';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';

// --- Reusable Helper Components (assuming they are in the same folder or imported) ---
// Note: These are the same components used by your other modals.
const API_URL = 'http://127.0.0.1:8000'; 

// A simplified type for strategy files listed in the explorer
interface StrategyFile {
    id: string;
    name: string;
    content: string;
}

interface OptimizableParameter {
    id: string;
    type: 'strategy_param' | 'indicator_param';
    name: string;
    value: number;
    enabled: boolean;
    mode: 'range' | 'list';
    start: number;
    end: number;
    step: number;
    list_values: string;
    indicatorIndex?: number;
    paramIndex?: number;
}

interface CombinationRule {
    id: string;
    param1: string;
    operator: '<' | '>' | '<=' | '>=' | '===' | '!==';
    param2: string;
}

// --- Step 1: Add new types for the durability test settings ---
export interface DataSegmentationSettings {
    type: 'data_segmentation';
    training_pct: number;
    validation_pct: number;
    // We don't need top_n or optimization_metric here
}

export interface WalkForwardSettings {
    type: 'walk_forward';
    training_period_length: number;
    training_period_unit: 'days' | 'weeks' | 'months';
    testing_period_length: number;
    testing_period_unit: 'days' | 'weeks' | 'months';
    step_forward_size_pct: number;
}

// --- Type Definitions for Hedge Optimization ---

export interface SingleStrategyHedgeConfig {
    strategy_code: string;
    parameters_to_optimize: OptimizableParameter[];
    combination_rules: Omit<CombinationRule, 'id'>[];
}

export interface HedgeOptimizationConfig {
    test_type: 'hedge_optimization';
    strategy_a: SingleStrategyHedgeConfig;
    strategy_b: SingleStrategyHedgeConfig;
    symbols_to_screen: string[];
    top_n_candidates: number;
    portfolio_metric: string;
    num_results_to_return: number;
    final_analysis: 'none' | DataSegmentationSettings | WalkForwardSettings;
}

// --- Component Props ---

interface HedgeModalProps {
    open: boolean;
    onClose: () => void;
    onSubmit: (config: HedgeOptimizationConfig) => void;
    isSubmitting: boolean;
    initialStrategy: StrategyFile | null; // This will be Strategy A
    availableStrategies: StrategyFile[];   // List of all other files for Strategy B dropdown
}

const fetchAllParametersAndSettings = async (code: string): Promise<any> => {
    const response = await fetch(`${API_URL}/api/strategies/parse-parameters`, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: code,
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "An unknown error" }));
        throw new Error(errorData.detail);
    }
    return response.json();
};

// --- The Main Modal Component ---

export const HedgeModal: React.FC<HedgeModalProps> = ({
    open, onClose, onSubmit, isSubmitting, initialStrategy, availableStrategies
}) => {

    // --- Global State ---
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [symbolList, setSymbolList] = useState<string[]>([]);
    const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
    
    // --- Hedge-Specific State ---
    const [topNCandidates, setTopNCandidates] = useState(20);
    const [portfolioMetric, setPortfolioMetric] = useState('Equity_Efficiency_Rate');

    // --- Strategy A State ---
    const [paramsA, setParamsA] = useState<OptimizableParameter[]>([]);
    const [rulesA, setRulesA] = useState<CombinationRule[]>([]);

    // --- Strategy B State ---
    const [selectedStrategyB, setSelectedStrategyB] = useState<StrategyFile | null>(null);
    const [paramsB, setParamsB] = useState<OptimizableParameter[]>([]);
    const [rulesB, setRulesB] = useState<CombinationRule[]>([]);
    const [isLoadingB, setIsLoadingB] = useState(false);

    const hasFetchedInitialData = useRef(false);

    // --- NEW: State for the final analysis step ---
    const [finalAnalysisType, setFinalAnalysisType] = useState<'none' | 'data_segmentation' | 'walk_forward'>('none');

    // State for Data Segmentation final analysis
    const [dsTrainingPct, setDsTrainingPct] = useState(70);
    const [dsValidationPct, setDsValidationPct] = useState(30);

    // State for Walk-Forward final analysis
    const [wfTrainingPeriod, setWfTrainingPeriod] = useState({ length: 180, unit: 'days' });
    const [wfTestingPeriod, setWfTestingPeriod] = useState({ length: 60, unit: 'days' });
    const [wfStepSize, setWfStepSize] = useState(25);

    // --- State for the number of results to return ---
    const [numResultsToReturn, setNumResultsToReturn] = useState(50);

    // Effect for initial data load (Symbols and Strategy A params)
    useEffect(() => {
        if (!open) {
            hasFetchedInitialData.current = false;
            // Reset states when modal closes
            setSelectedStrategyB(null);
            setParamsA([]);
            setParamsB([]);
            setRulesA([]);
            setRulesB([]);
            return;
        }

        if (open && initialStrategy && !hasFetchedInitialData.current) {
            setIsLoading(true);
            setError(null);
            Promise.all([
                fetchAvailableSymbols('binance'),
                fetchAllParametersAndSettings(initialStrategy.content)
            ]).then(([symbols, paramsDataA]) => {
                setSymbolList(symbols);
                const defaultSymbol = paramsDataA.settings?.symbol;
                if (defaultSymbol) setSelectedSymbols([defaultSymbol]);

                const optimizableParamsA = paramsDataA.optimizable_params.map((p: any) => ({
                    ...p, enabled: false, mode: 'range', start: p.value, end: p.value, step: 1, list_values: String(p.value),
                }));
                setParamsA(optimizableParamsA);
                hasFetchedInitialData.current = true;
            }).catch(err => setError(err.message))
            .finally(() => setIsLoading(false));
        }
    }, [open, initialStrategy]);

    // Effect for loading Strategy B's parameters when it's selected
    useEffect(() => {
        if (selectedStrategyB) {
            setIsLoadingB(true);
            fetchAllParametersAndSettings(selectedStrategyB.content)
                .then(paramsDataB => {
                    const optimizableParamsB = paramsDataB.optimizable_params.map((p: any) => ({
                        ...p, enabled: false, mode: 'range', start: p.value, end: p.value, step: 1, list_values: String(p.value),
                    }));
                    setParamsB(optimizableParamsB);
                    setRulesB([]); // Reset rules for new strategy
                })
                .catch(err => setError(`Failed to load Strategy B: ${err.message}`))
                .finally(() => setIsLoadingB(false));
        } else {
            setParamsB([]); // Clear params if B is deselected
        }
    }, [selectedStrategyB]);


    // --- Generic Handlers ---
    const handleParamChange = (setter: React.Dispatch<React.SetStateAction<OptimizableParameter[]>>) => 
        (id: string, field: keyof OptimizableParameter, value: any) => {
            setter(prev => prev.map(p => p.id === id ? { ...p, [field]: value } : p));
    };
    const addRule = (setter: React.Dispatch<React.SetStateAction<CombinationRule[]>>) => () => {
        setter(prev => [...prev, { id: uuidv4(), param1: '', operator: '<', param2: '' }]);
    };
    const updateRule = (setter: React.Dispatch<React.SetStateAction<CombinationRule[]>>) => 
        (id: string, field: keyof Omit<CombinationRule, 'id'>, value: string) => {
            setter(prev => prev.map(rule => rule.id === id ? { ...rule, [field]: value } : rule));
    };
    const removeRule = (setter: React.Dispatch<React.SetStateAction<CombinationRule[]>>) => (id: string) => {
        setter(prev => prev.filter(rule => rule.id !== id));
    };

    const handleSubmit = () => {
        if (!initialStrategy || !selectedStrategyB) {
            setError("Both Strategy A and Strategy B must be selected."); return;
        }
        if (selectedSymbols.length === 0) {
            setError("Please select at least one asset to test."); return;
        }

        let finalAnalysisConfig: HedgeOptimizationConfig['final_analysis'];

        if (finalAnalysisType === 'data_segmentation') {
            finalAnalysisConfig = {
                type: 'data_segmentation',
                training_pct: dsTrainingPct,
                validation_pct: dsValidationPct,
            };
        } else if (finalAnalysisType === 'walk_forward') {
            finalAnalysisConfig = {
                type: 'walk_forward',
                training_period_length: wfTrainingPeriod.length,
                training_period_unit: wfTrainingPeriod.unit as 'days', // Cast for type safety
                testing_period_length: wfTestingPeriod.length,
                testing_period_unit: wfTestingPeriod.unit as 'days',
                step_forward_size_pct: wfStepSize,
            };
        } else {
            finalAnalysisConfig = { type: 'none' };
        }

        const config: HedgeOptimizationConfig = {
            test_type: 'hedge_optimization',
            strategy_a: {
                strategy_code: initialStrategy.content,
                parameters_to_optimize: paramsA.filter(p => p.enabled),
                combination_rules: rulesA.map(({ id, ...rest }) => rest),
            },
            strategy_b: {
                strategy_code: selectedStrategyB.content,
                parameters_to_optimize: paramsB.filter(p => p.enabled),
                combination_rules: rulesB.map(({ id, ...rest }) => rest),
            },
            symbols_to_screen: selectedSymbols,
            top_n_candidates: topNCandidates,
            portfolio_metric: portfolioMetric,
            num_results_to_return: numResultsToReturn,
            final_analysis: finalAnalysisConfig,
        };
        onSubmit(config);
    };

    const enabledParamsForRulesA = paramsA.filter(p => p.enabled);
    const enabledParamsForRulesB = paramsB.filter(p => p.enabled);
    
    // Filter out initial strategy from the dropdown for B
    const availableStrategiesForB = availableStrategies.filter(s => s.id !== initialStrategy?.id);

    return (
        <Dialog open={open} onClose={onClose} sx={{ display:'flex', justifyContent:'center', backdropFilter: 'blur(4px)',
                '& .css-iuy973-MuiPaper-root-MuiDialog-paper': {
                    width: '90vw',
                    height: '90vh',
                    maxWidth: '100%',
                }
            }}>
            <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                Configure Hedge Optimization
                <IconButton aria-label="close" onClick={onClose}><CloseIcon /></IconButton>
            </DialogTitle>
            <DialogContent dividers>
                {isLoading && <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}><CircularProgress /></Box>}
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

                {!isLoading && !error && (
                    <Box sx={{ display: 'flex', gap: 3 }}>
                        
                        {/* --- STRATEGY A PANEL --- */}
                        <StrategyPanel
                            title="Strategy A (e.g., Long)"
                            strategy={initialStrategy}
                            params={paramsA}
                            rules={rulesA}
                            onParamChange={handleParamChange(setParamsA)}
                            onAddRule={addRule(setRulesA)}
                            onUpdateRule={updateRule(setRulesA)}
                            onRemoveRule={removeRule(setRulesA)}
                            enabledParamsForRules={enabledParamsForRulesA}
                            isSubmitting={isSubmitting}
                        />

                        <Divider orientation="vertical" flexItem />

                        {/* --- STRATEGY B PANEL --- */}
                        <StrategyPanel
                            title="Strategy B (e.g., Short)"
                            strategy={selectedStrategyB}
                            params={paramsB}
                            rules={rulesB}
                            onParamChange={handleParamChange(setParamsB)}
                            onAddRule={addRule(setRulesB)}
                            onUpdateRule={updateRule(setRulesB)}
                            onRemoveRule={removeRule(setRulesB)}
                            enabledParamsForRules={enabledParamsForRulesB}
                            isSubmitting={isSubmitting}
                            isLoading={isLoadingB}
                        >
                            <Autocomplete
                                options={availableStrategiesForB}
                                getOptionLabel={(option) => option.name}
                                value={selectedStrategyB}
                                onChange={(event, newValue) => setSelectedStrategyB(newValue)}
                                renderInput={(params) => <TextField {...params} label="Select Strategy B File" />}
                                sx={{ mb: 2 }}
                            />
                        </StrategyPanel>
                        
                        <Divider orientation="vertical" flexItem />

                        {/* --- GLOBAL SETTINGS PANEL --- */}
                        <Box sx={{ flex: 1, minWidth: '300px', display: 'flex', flexDirection: 'column', gap: 3 }}>
                            <Typography variant="h6">Global Settings</Typography>
                            <Autocomplete
                                multiple id="hedge-asset-autocomplete" options={symbolList} value={selectedSymbols}
                                onChange={(event, newValue) => setSelectedSymbols(newValue)}
                                renderTags={(value, getTagProps) => value.map((option, index) => <Chip key={index} label={option} {...getTagProps({ index })} />)}
                                renderInput={(params) => <TextField {...params} label="Symbols to Test" />}
                            />
                            <TextField
                                type="number"
                                label="Top N Candidates to Pair"
                                value={topNCandidates}
                                onChange={e => setTopNCandidates(parseInt(e.target.value) || 1)}
                                helperText="Number of best results from each strategy to combine."
                                InputProps={{ inputProps: { min: 1, max: 100 } }}
                            />
                            <TextField
                                type="number"
                                label="Number of Top Pairs to Return"
                                value={numResultsToReturn}
                                onChange={e => setNumResultsToReturn(parseInt(e.target.value) || 1)}
                                helperText="How many of the best hedge combos to show in the results."
                                InputProps={{ inputProps: { min: 1, max: 500 } }}
                            />
                            <Select
                                value={portfolioMetric}
                                onChange={e => setPortfolioMetric(e.target.value)}
                                displayEmpty
                            >
                                <MenuItem value="Equity_Efficiency_Rate">Maximize Equity Efficiency Rate</MenuItem>
                                <MenuItem value="Sharpe_Ratio">Maximize Sharpe Ratio</MenuItem>
                                <MenuItem value="Profit_Factor">Maximize Profit Factor</MenuItem>
                                <MenuItem value="Calmar_Ratio">Maximize Calmar Ratio</MenuItem>
                                <MenuItem value="Net_Profit">Maximize Net Profit</MenuItem>
                                <MenuItem value="Max_Drawdown">Minimize Max Drawdown</MenuItem>
                                <MenuItem value="Max_Drawdown_Duration_days">Minimize Max Drawdown Period</MenuItem>
                            </Select>
                            {/* --- Final Analysis Section --- */}
                            <Divider sx={{ my: 1 }} />
                            <Typography variant="h6">Final Analysis</Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                Optionally, run a durability test on the combined equity curve of the single best hedge pair found.
                            </Typography>
                            
                            <Select
                                value={finalAnalysisType}
                                onChange={e => setFinalAnalysisType(e.target.value as any)}
                            >
                                <MenuItem value="none">None (Standard Run)</MenuItem>
                                <MenuItem value="data_segmentation">Data Segmentation</MenuItem>
                                <MenuItem value="walk_forward">Walk-Forward Analysis</MenuItem>
                            </Select>

                            {/* Conditional UI for Data Segmentation */}
                            {finalAnalysisType === 'data_segmentation' && (
                                <Box sx={{ mt: 2, p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                                    <Typography gutterBottom>Data Segmentation on Best Pair</Typography>
                                    <Slider
                                        value={dsTrainingPct}
                                        onChange={(e, val) => {
                                            const newTrain = val as number;
                                            setDsTrainingPct(newTrain);
                                            setDsValidationPct(100 - newTrain);
                                        }}
                                        valueLabelFormat={(val) => `Train: ${val}% | Test: ${100 - val}%`}
                                        valueLabelDisplay="auto"
                                    />
                                </Box>
                            )}

                            {/* Conditional UI for Walk-Forward */}
                            {finalAnalysisType === 'walk_forward' && (
                                <Box sx={{ mt: 2, p: 2, border: 1, borderColor: 'divider', borderRadius: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
                                    <Typography>Walk-Forward on Best Pair</Typography>
                                    <Box sx={{ display: 'flex', gap: 2 }}>
                                        <TextField fullWidth type="number" label="Training Period" value={wfTrainingPeriod.length} onChange={e => setWfTrainingPeriod(p => ({ ...p, length: parseInt(e.target.value) || 0 }))}/>
                                        <Select value={wfTrainingPeriod.unit} onChange={e => setWfTrainingPeriod(p => ({ ...p, unit: e.target.value }))} sx={{ minWidth: 120 }}>
                                            <MenuItem value="days">Days</MenuItem>
                                            <MenuItem value="weeks">Weeks</MenuItem>
                                            <MenuItem value="months">Months</MenuItem>
                                        </Select>
                                    </Box>
                                    <Box sx={{ display: 'flex', gap: 2 }}>
                                        <TextField fullWidth type="number" label="Testing Period" value={wfTestingPeriod.length} onChange={e => setWfTestingPeriod(p => ({ ...p, length: parseInt(e.target.value) || 0 }))}/>
                                        <Select value={wfTestingPeriod.unit} onChange={e => setWfTestingPeriod(p => ({ ...p, unit: e.target.value }))} sx={{ minWidth: 120 }}>
                                            <MenuItem value="days">Days</MenuItem>
                                            <MenuItem value="weeks">Weeks</MenuItem>
                                            <MenuItem value="months">Months</MenuItem>
                                        </Select>
                                    </Box>
                                    <Typography variant="body2" color="text.secondary">Step Forward Size (% of Testing Period)</Typography>
                                    <Slider value={wfStepSize} onChange={(e, val) => setWfStepSize(val as number)} valueLabelDisplay="auto" step={5} marks min={5} max={100} />
                                </Box>
                            )}
                        </Box>
                    </Box>
                )}
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} disabled={isSubmitting}>Cancel</Button>
                <Button onClick={handleSubmit} variant="contained" disabled={isLoading || isSubmitting}>Run Hedge Optimization</Button>
            </DialogActions>
        </Dialog>
    );
};


// --- Sub-component for displaying a strategy's configuration panel ---
const StrategyPanel: React.FC<{
    title: string;
    strategy: StrategyFile | null;
    params: OptimizableParameter[];
    rules: CombinationRule[];
    onParamChange: (id: string, field: keyof OptimizableParameter, value: any) => void;
    onAddRule: () => void;
    onUpdateRule: (id: string, field: keyof Omit<CombinationRule, 'id'>, value: string) => void;
    onRemoveRule: (id: string) => void;
    enabledParamsForRules: OptimizableParameter[];
    isSubmitting: boolean;
    isLoading?: boolean;
    children?: React.ReactNode;
}> = ({ title, strategy, params, rules, onParamChange, onAddRule, onUpdateRule, onRemoveRule, enabledParamsForRules, isSubmitting, isLoading, children }) => {
    return (
        <Box sx={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>{title}</Typography>
            {children || <TextField label="Strategy A File" value={strategy?.name || ''} disabled fullWidth sx={{ mb: 2 }} />}
            
            {isLoading ? <CircularProgress sx={{ alignSelf: 'center', my: 4 }}/> :
            strategy && (
                <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 2, overflow: 'hidden' }}>
                    <Box sx={{ flex: 1, pr: 1 }}>
                        <Typography variant="overline">Parameters to Optimize</Typography>
                        {params.map(param => (
                            <ParameterInputRow key={param.id} param={param} handleParamChange={onParamChange} isSubmitting={isSubmitting} />
                        ))}
                    </Box>
                    <Divider />
                    <Box sx={{ flex: 1, overflowY: 'auto', pr: 1 }}>
                        <Typography variant="overline">Combination Rules</Typography>
                        {rules.map(rule => (
                            <CombinationRuleRow key={rule.id} rule={rule} availableParams={enabledParamsForRules} onUpdate={onUpdateRule} onRemove={onRemoveRule} />
                        ))}
                        <Button startIcon={<AddCircleOutlineIcon />} onClick={onAddRule} disabled={enabledParamsForRules.length < 2} sx={{ mt: 1 }}>Add Rule</Button>
                    </Box>
                </Box>
            )}
        </Box>
    );
}

const CombinationRuleRow: React.FC<{rule: CombinationRule, availableParams: OptimizableParameter[], onUpdate: Function, onRemove: Function}> = ({rule, availableParams, onUpdate, onRemove}) => {
    const operators = ['<', '>', '<=', '>=', '===', '!=='];
    return (
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 1, p: 1.5, borderRadius: 2, bgcolor: 'action.hover' }}>
            <Select size="small" value={rule.param1} onChange={e => onUpdate(rule.id, 'param1', e.target.value)} displayEmpty fullWidth>
                <MenuItem value="" disabled>Select Param...</MenuItem>
                {availableParams.map(p => <MenuItem key={p.id} value={p.id}>{p.name}</MenuItem>)}
            </Select>
            <Select size="small" value={rule.operator} onChange={e => onUpdate(rule.id, 'operator', e.target.value)}>
                {operators.map(op => <MenuItem key={op} value={op}>{op}</MenuItem>)}
            </Select>
            <Select size="small" value={rule.param2} onChange={e => onUpdate(rule.id, 'param2', e.target.value)} displayEmpty fullWidth>
                <MenuItem value="" disabled>Select Param...</MenuItem>
                {availableParams.map(p => <MenuItem key={p.id} value={p.id}>{p.name}</MenuItem>)}
            </Select>
            <IconButton onClick={() => onRemove(rule.id)} color="error"><DeleteOutlineIcon /></IconButton>
        </Box>
    );
}

// --- Helper Component for each parameter's input row (from previous step) ---
const ParameterInputRow: React.FC<{param: OptimizableParameter, handleParamChange: Function, isSubmitting: boolean}> = ({param, handleParamChange, isSubmitting}) => {
    return (
        <Box sx={{ p: 2, mb: 1, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <FormControlLabel
                    control={<Checkbox checked={param.enabled} onChange={e => handleParamChange(param.id, 'enabled', e.target.checked)} />}
                    label={<Typography sx={{ fontWeight: 500 }}>{param.name}</Typography>}
                />
                <ToggleButtonGroup
                    size="small"
                    value={param.mode}
                    exclusive
                    disabled={!param.enabled || isSubmitting}
                    onChange={(e, newMode) => { if (newMode) handleParamChange(param.id, 'mode', newMode) }}
                >
                    <ToggleButton value="range">Range</ToggleButton>
                    <ToggleButton value="list">List</ToggleButton>
                </ToggleButtonGroup>
            </Box>
            
            {param.mode === 'range' ? (
                <Box sx={{ display: 'flex', gap: 2, pl: 6 }}>
                    <TextField fullWidth label="Start" type="number" variant="outlined" size="small" value={param.start} onChange={e => handleParamChange(param.id, 'start', parseFloat(e.target.value) || 0)} disabled={!param.enabled || isSubmitting} InputLabelProps={{ shrink: true }}/>
                    <TextField fullWidth label="End" type="number" variant="outlined" size="small" value={param.end} onChange={e => handleParamChange(param.id, 'end', parseFloat(e.target.value) || 0)} disabled={!param.enabled || isSubmitting} InputLabelProps={{ shrink: true }}/>
                    <TextField fullWidth label="Step" type="number" variant="outlined" size="small" value={param.step} onChange={e => handleParamChange(param.id, 'step', parseFloat(e.target.value) || 1)} disabled={!param.enabled || isSubmitting} InputLabelProps={{ shrink: true }}/>
                </Box>
            ) : (
                <Box sx={{ pl: 6 }}>
                    <TextField
                        fullWidth
                        label="Values (comma-separated)"
                        variant="outlined"
                        size="small"
                        value={param.list_values}
                        onChange={e => handleParamChange(param.id, 'list_values', e.target.value)}
                        placeholder="e.g., 20, 50, 100, 155"
                        disabled={!param.enabled || isSubmitting}
                        InputLabelProps={{ shrink: true }}
                    />
                </Box>
            )}
        </Box>
    );
}