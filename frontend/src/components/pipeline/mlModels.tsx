// Configuration for rendering hyperparameter inputs dynamically
export const HYPERPARAMETER_CONFIG: Record<string, { name: string; label: string; type: 'number' | 'text' | 'select'; options?: string[]; defaultValue: any; }[]> = {
    'LogisticRegression': [
        { name: 'C', label: 'Inverse Regularization (C)', type: 'number', defaultValue: 1.0 },
        { name: 'penalty', label: 'Penalty', type: 'select', options: ['l1', 'l2', 'elasticnet', 'none'], defaultValue: 'l2' },
        { name: 'solver', label: 'Solver', type: 'select', options: ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'], defaultValue: 'lbfgs' },
    ],
    'RandomForestClassifier': [
        { name: 'n_estimators', label: 'Number of Trees', type: 'number', defaultValue: 100 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 10 },
        { name: 'min_samples_split', label: 'Min Samples to Split', type: 'number', defaultValue: 2 },
        { name: 'min_samples_leaf', label: 'Min Samples per Leaf', type: 'number', defaultValue: 1 },
        { name: 'class_weight', label: 'Class Weight', type: 'select', options: ['balanced', 'balanced_subsample', 'none'], defaultValue: 'balanced' },
    ],
    'LightGBMClassifier': [
        { name: 'n_estimators', label: 'Number of Estimators', type: 'number', defaultValue: 100 },
        { name: 'learning_rate', label: 'Learning Rate', type: 'number', defaultValue: 0.1 },
        { name: 'num_leaves', label: 'Number of Leaves', type: 'number', defaultValue: 31 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: -1 },
    ],
    'XGBoostClassifier': [
        { name: 'n_estimators', label: 'Number of Estimators', type: 'number', defaultValue: 100 },
        { name: 'learning_rate', label: 'Learning Rate', type: 'number', defaultValue: 0.1 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 3 },
        { name: 'subsample', label: 'Subsample Ratio', type: 'number', defaultValue: 1.0 },
        { name: 'colsample_bytree', label: 'Colsample by Tree', type: 'number', defaultValue: 1.0 },
    ],
    'SVC': [ // Support Vector Classifier
        { name: 'C', label: 'Regularization (C)', type: 'number', defaultValue: 1.0 },
        { name: 'kernel', label: 'Kernel', type: 'select', options: ['rbf', 'linear', 'poly', 'sigmoid'], defaultValue: 'rbf' },
        { name: 'gamma', label: 'Kernel Coefficient (gamma)', type: 'select', options: ['scale', 'auto'], defaultValue: 'scale' },
    ],
    'KNeighborsClassifier': [
        { name: 'n_neighbors', label: 'Number of Neighbors (K)', type: 'number', defaultValue: 5 },
        { name: 'weights', label: 'Weighting', type: 'select', options: ['uniform', 'distance'], defaultValue: 'uniform' },
        { name: 'algorithm', label: 'Algorithm', type: 'select', options: ['auto', 'ball_tree', 'kd_tree', 'brute'], defaultValue: 'auto' },
    ],
    'DecisionTreeClassifier': [
        { name: 'criterion', label: 'Criterion', type: 'select', options: ['gini', 'entropy'], defaultValue: 'gini' },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 10 },
        { name: 'min_samples_split', label: 'Min Samples to Split', type: 'number', defaultValue: 2 },
        { name: 'min_samples_leaf', label: 'Min Samples per Leaf', type: 'number', defaultValue: 1 },
    ],

    // --- Regression Models ---
    'LinearRegression': [
        { name: 'fit_intercept', label: 'Fit Intercept', type: 'select', options: ['true', 'false'], defaultValue: 'true' },
    ],
    'RandomForestRegressor': [
        { name: 'n_estimators', label: 'Number of Trees', type: 'number', defaultValue: 100 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 10 },
        { name: 'min_samples_split', label: 'Min Samples to Split', type: 'number', defaultValue: 2 },
        { name: 'criterion', label: 'Criterion', type: 'select', options: ['squared_error', 'absolute_error', 'poisson'], defaultValue: 'squared_error' },
    ],
    'XGBoostRegressor': [
        { name: 'n_estimators', label: 'Number of Estimators', type: 'number', defaultValue: 100 },
        { name: 'learning_rate', label: 'Learning Rate', type: 'number', defaultValue: 0.1 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 3 },
        { name: 'subsample', label: 'Subsample Ratio', type: 'number', defaultValue: 1.0 },
    ],
    'SVR': [ // Support Vector Regressor
        { name: 'C', label: 'Regularization (C)', type: 'number', defaultValue: 1.0 },
        { name: 'kernel', label: 'Kernel', type: 'select', options: ['rbf', 'linear', 'poly'], defaultValue: 'rbf' },
        { name: 'epsilon', label: 'Epsilon (margin of tolerance)', type: 'number', defaultValue: 0.1 },
    ],

    // --- Unsupervised Learning Models ---
    'KMeans': [
        { name: 'n_clusters', label: 'Number of Clusters (K)', type: 'number', defaultValue: 3 },
        { name: 'init', label: 'Initialization Method', type: 'select', options: ['k-means++', 'random'], defaultValue: 'k-means++' },
        { name: 'n_init', label: 'Number of Initializations', type: 'number', defaultValue: 10 },
        { name: 'max_iter', label: 'Max Iterations', type: 'number', defaultValue: 300 },
    ],
    'PCA': [ // Principal Component Analysis
        { name: 'n_components', label: 'Number of Components', type: 'number', defaultValue: 2 },
        { name: 'svd_solver', label: 'SVD Solver', type: 'select', options: ['auto', 'full', 'arpack', 'randomized'], defaultValue: 'auto' },
    ],
    'IsolationForest': [
        { name: 'n_estimators', label: 'Number of Estimators', type: 'number', defaultValue: 100 },
        { name: 'contamination', label: 'Contamination (Outlier %)', type: 'number', defaultValue: 0.1 },
        { name: 'max_features', label: 'Max Features per Tree', type: 'number', defaultValue: 1.0 },
    ],
    'DBSCAN': [
        { name: 'eps', label: 'Max Distance (eps)', type: 'number', defaultValue: 0.5 },
        { name: 'min_samples', label: 'Min Samples in Neighborhood', type: 'number', defaultValue: 5 },
        { name: 'algorithm', label: 'Algorithm', type: 'select', options: ['auto', 'ball_tree', 'kd_tree', 'brute'], defaultValue: 'auto' },
    ]
};