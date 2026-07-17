from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
import os
from werkzeug.utils import secure_filename
import json
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_data_info(df):
    numeric_df = df.select_dtypes(include=[np.number])
    missing_values = df.isnull().sum().to_dict()
    missing_percent = (df.isnull().mean() * 100).round(2).to_dict()
    zero_variance_columns = []
    if not numeric_df.empty:
        variances = numeric_df.var(ddof=0)
        zero_variance_columns = [col for col, var in variances.items() if float(var) == 0.0]

    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    return {
        'rows': len(df),
        'columns': list(df.columns),
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': missing_values,
        'missing_percent': missing_percent,
        'numeric_columns': numeric_df.columns.tolist(),
        'categorical_columns': categorical_cols,
        'zero_variance_columns': zero_variance_columns,
        'sample': df.head(3).to_dict(orient='records')
    }

# Store uploaded data in memory for analysis
uploaded_data = None
data_info = None
latest_analysis = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global uploaded_data, data_info
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Only CSV and Excel files allowed'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the file
        if filename.endswith('.csv'):
            uploaded_data = pd.read_csv(filepath)
        else:
            uploaded_data = pd.read_excel(filepath)
        
        # Get data info
        data_info = compute_data_info(uploaded_data)
        
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully',
            'data_info': data_info
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clean', methods=['POST'])
def clean_data():
    global uploaded_data, data_info
    try:
        if uploaded_data is None:
            return jsonify({'success': False, 'error': 'No data uploaded'}), 400

        if not data_info:
            return jsonify({'success': False, 'error': 'Dataset information unavailable'}), 500

        zero_cols = data_info.get('zero_variance_columns', [])
        if not zero_cols:
            return jsonify({'success': True, 'message': 'No zero-variance columns found', 'data_info': data_info, 'removed_columns': []})

        uploaded_data = uploaded_data.drop(columns=zero_cols)
        data_info = compute_data_info(uploaded_data)

        return jsonify({'success': True, 'message': 'Zero-variance columns removed', 'removed_columns': zero_cols, 'data_info': data_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    global uploaded_data, latest_analysis
    
    try:
        if uploaded_data is None:
            return jsonify({'success': False, 'error': 'No data uploaded'}), 400
        
        data = json.loads(request.data)
        algorithms = data.get('algorithms', [])
        # Support both single and multi-select target columns from frontend
        target_columns = data.get('target_columns', None)
        if target_columns is None:
            single = data.get('target_column', None)
            if single:
                target_columns = [single]
        
        if not algorithms:
            return jsonify({'success': False, 'error': 'No algorithms selected'}), 400
        
        results = {}
        best_model = None
        best_score = None
        best_metric = None
        
        for algo in algorithms:
            try:
                if algo in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Linear Regression']:
                    # Supervised learning
                    if not target_columns:
                        results[algo] = {'error': 'Please specify a valid target column for supervised learning'}
                        continue
                    # validate columns
                    invalid = [c for c in target_columns if c not in uploaded_data.columns]
                    if invalid:
                        results[algo] = {'error': f'Invalid target columns: {invalid}'}
                        continue
                    
                    result = run_supervised_algorithm(algo, target_columns)
                    results[algo] = result
                
                elif algo in ['K-Means', 'PCA', 'Hierarchical Clustering']:
                    # Unsupervised learning
                    result = run_unsupervised_algorithm(algo)
                    results[algo] = result
                else:
                    result = {'error': 'Unknown algorithm'}
                    results[algo] = result
                
                # Track best model based on the most relevant metric
                if 'error' not in result:
                    score = None
                    metric = None
                    if result.get('type') == 'Classification' and 'accuracy' in result:
                        score = result['accuracy']
                        metric = 'accuracy'
                    elif result.get('type') == 'Regression' and 'r2_score' in result:
                        score = result['r2_score']
                        metric = 'r2_score'
                    elif result.get('type') == 'Clustering' and 'silhouette_score' in result:
                        score = result['silhouette_score']
                        metric = 'silhouette_score'
                    elif result.get('type') == 'Dimensionality Reduction' and 'explained_variance_ratio' in result:
                        score = result['explained_variance_ratio']
                        metric = 'explained_variance_ratio'
                    
                    if score is not None and (best_score is None or score > best_score):
                        best_model = algo
                        best_score = score
                        best_metric = metric
            
            except Exception as e:
                results[algo] = {'error': str(e)}
        
        latest_analysis = {
            'target_columns': target_columns or [],
            'algorithms': algorithms,
            'results': results,
            'best_model': {
                'name': best_model,
                'metric': best_metric,
                'score': float(best_score) if best_score is not None else None
            } if best_model else None,
            'data_info': data_info
        }
        
        return jsonify({'success': True, 'results': results, 'best_model': latest_analysis['best_model']})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def run_supervised_algorithm(algorithm, target_columns):
    try:
        df = uploaded_data.copy()
        # Prepare data
        # allow single or multiple target columns
        if isinstance(target_columns, (list, tuple)) and len(target_columns) > 1:
            y = df[target_columns]
        else:
            # single target
            col = target_columns[0] if isinstance(target_columns, (list, tuple)) else target_columns
            y = df[col]
        X = df.drop(columns=target_columns if isinstance(target_columns, (list, tuple)) else [target_columns])
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Check if target is categorical or continuous
        # If multi-output, detect per-column; we'll treat it as classification if y has object dtype or low cardinality
        is_multi_output = isinstance(y, pd.DataFrame)
        if not is_multi_output:
            classification = (y.dtype == 'object') or (len(y.unique()) < len(y) * 0.5)
        else:
            # if any target column looks categorical, use multioutput estimator for classification
            classification = any((y[c].dtype == 'object') or (len(y[c].unique()) < len(y[c]) * 0.5) for c in y.columns)

        # Force regression for Linear Regression, even if target cardinality is low
        if algorithm == 'Linear Regression':
            classification = False

        if classification:
            # Classification
            if not is_multi_output:
                y_encoded = LabelEncoder().fit_transform(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            else:
                # encode each column
                y_encoded = y.copy()
                for col in y_encoded.columns:
                    if y_encoded[col].dtype == 'object' or y_encoded[col].dtype == 'bool':
                        y_encoded[col] = LabelEncoder().fit_transform(y_encoded[col].astype(str))
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            
            if algorithm == 'Logistic Regression':
                base = LogisticRegression(max_iter=1000, random_state=42)
            elif algorithm == 'Decision Tree':
                base = DecisionTreeClassifier(random_state=42)
            elif algorithm == 'Random Forest':
                base = RandomForestClassifier(random_state=42, n_estimators=100)
            elif algorithm == 'SVM':
                base = SVC(random_state=42)
            else:
                raise ValueError(f"Unsupported classification algorithm: {algorithm}")

            if not is_multi_output:
                model = base
            else:
                model = MultiOutputClassifier(base)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # For multioutput, compute average accuracy
            if not is_multi_output:
                accuracy = accuracy_score(y_test, y_pred)
            else:
                # y_test is DataFrame; y_pred is array
                accs = []
                for i, col in enumerate(y_test.columns):
                    accs.append(accuracy_score(y_test.iloc[:, i], y_pred[:, i]))
                accuracy = float(np.mean(accs))
            
            return {
                'accuracy': float(accuracy),
                'type': 'Classification',
                'train_size': len(X_train),
                'test_size': len(X_test),
                'features_used': len(X.columns)
            }
        else:
            # Regression
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if algorithm == 'Linear Regression':
                base = LinearRegression()
            elif algorithm == 'Decision Tree':
                base = DecisionTreeRegressor(random_state=42)
            elif algorithm == 'Random Forest':
                base = RandomForestRegressor(random_state=42)
            elif algorithm == 'SVM':
                base = SVR(kernel='rbf')
            else:
                raise ValueError(f"Unsupported regression algorithm: {algorithm}")

            # For regression multioutput use MultiOutputRegressor
            if not is_multi_output:
                model = base
            else:
                model = MultiOutputRegressor(base)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # compute multioutput metrics by averaging
            if not is_multi_output:
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
            else:
                # y_test is DataFrame; y_pred is array
                r2s = []
                mses = []
                for i, col in enumerate(y_test.columns):
                    r2s.append(r2_score(y_test.iloc[:, i], y_pred[:, i]))
                    mses.append(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
                r2 = float(np.mean(r2s))
                mse = float(np.mean(mses))
            
            return {
                'r2_score': float(r2),
                'mse': float(mse),
                'type': 'Regression',
                'train_size': len(X_train),
                'test_size': len(X_test),
                'features_used': len(X.columns)
            }
    
    except Exception as e:
        return {'error': str(e)}

def run_unsupervised_algorithm(algorithm):
    try:
        df = uploaded_data.copy()
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'error': 'No numeric columns found in dataset'}
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)
        
        if algorithm == 'K-Means':
            optimal_k = min(10, max(2, len(numeric_df) // 10))
            model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            silhouette = silhouette_score(X_scaled, labels)
            
            return {
                'clusters': int(optimal_k),
                'silhouette_score': float(silhouette),
                'type': 'Clustering',
                'samples': len(numeric_df),
                'features_used': X_scaled.shape[1]
            }
        
        elif algorithm == 'PCA':
            n_components = min(3, X_scaled.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
            explained_variance = float(pca.explained_variance_ratio_.sum())
            
            return {
                'components': int(n_components),
                'explained_variance_ratio': float(explained_variance),
                'type': 'Dimensionality Reduction',
                'samples': len(numeric_df),
                'original_features': X_scaled.shape[1]
            }
        
        elif algorithm == 'Hierarchical Clustering':
            from scipy.cluster.hierarchy import dendrogram, linkage
            optimal_k = min(10, max(2, len(numeric_df) // 10))
            linkage_matrix = linkage(X_scaled, method='ward')
            
            return {
                'clusters': int(optimal_k),
                'type': 'Hierarchical Clustering',
                'samples': len(numeric_df),
                'features_used': X_scaled.shape[1]
            }
    
    except Exception as e:
        return {'error': str(e)}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = json.loads(request.data)
        user_message = data.get('message', '').lower()
        
        response = generate_chat_response(user_message)
        
        return jsonify({'success': True, 'response': response})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_chat_response(user_message):
    """Generate response to user queries about ML algorithms and analysis"""
    global uploaded_data, data_info, latest_analysis
    
    lower = user_message.lower()
    
    if uploaded_data is None:
        return "I can answer general ML questions, but please upload a dataset first to get analysis-specific guidance."
    
    if 'best model' in lower or 'most accurate' in lower or 'best algorithm' in lower or 'accurate model' in lower:
        best = latest_analysis.get('best_model')
        if best and best.get('name'):
            score = best.get('score')
            metric = best.get('metric')
            selected_models = latest_analysis.get('algorithms') or []
            selected_text = ', '.join(selected_models) if selected_models else 'the selected models'
            return f"Based on your uploaded dataset and the selected models ({selected_text}), the most accurate model was {best['name']} with a {metric.replace('_', ' ')} of {score:.4f}."
        return 'The analysis has not produced a best model yet. Please run the analysis first.'
    
    if 'dataset' in lower or 'data' in lower or 'uploaded' in lower or 'file' in lower:
        if data_info:
            columns_display = ', '.join(data_info['columns'][:10])
            if len(data_info['columns']) > 10:
                columns_display += ', ...'
            selected_models = latest_analysis.get('algorithms') or []
            selected_text = f" You analyzed: {', '.join(selected_models)}." if selected_models else ''
            return f"Your uploaded dataset has {data_info['rows']} rows and {len(data_info['columns'])} columns. The first columns are: {columns_display}.{selected_text}"
        return 'I cannot read the dataset details yet. Please upload a file first.'
    
    if 'target column' in lower or 'target columns' in lower or 'target' in lower:
        cols = latest_analysis.get('target_columns') or []
        if cols:
            return f"The analysis is using target column(s): {', '.join(cols)}. This allows multi-output prediction when more than one target is selected."
        return 'No target columns are selected yet. Please choose one or more target columns before running supervised analysis.'
    
    if 'result' in lower or 'score' in lower or 'accuracy' in lower or 'r2' in lower or 'silhouette' in lower or 'analysis' in lower:
        results = latest_analysis.get('results') or {}
        selected_models = latest_analysis.get('algorithms') or []
        if selected_models:
            for model_name in selected_models:
                if model_name.lower() in lower:
                    result = results.get(model_name)
                    if result and not result.get('error'):
                        if result.get('type') == 'Classification' and 'accuracy' in result:
                            return f"{model_name} achieved an accuracy score of {result['accuracy']:.4f} on the uploaded dataset."
                        if result.get('type') == 'Regression' and 'r2_score' in result:
                            return f"{model_name} achieved an R2 score of {result['r2_score']:.4f} and MSE of {result['mse']:.4f}."
                        if result.get('type') == 'Clustering' and 'silhouette_score' in result:
                            return f"{model_name} achieved a silhouette score of {result['silhouette_score']:.4f}."
                        if result.get('type') == 'Dimensionality Reduction' and 'explained_variance_ratio' in result:
                            return f"{model_name} explained {result['explained_variance_ratio']:.4f} of the variance."
            best = latest_analysis.get('best_model')
            if best and best.get('name'):
                return f"The best performing model in the latest analysis was {best['name']} with a {best['metric'].replace('_', ' ')} score of {best['score']:.4f}."
        return 'Analysis results are available after running the selected algorithms. Upload a dataset, select your models, and click Analyze.'
    
    responses = {
        'supervised learning': 'Supervised learning uses labeled data to train models. Common algorithms include Logistic Regression, Decision Trees, Random Forests, and SVM.',
        'unsupervised learning': 'Unsupervised learning finds patterns in unlabeled data. Common algorithms include K-Means clustering, PCA for dimensionality reduction, and Hierarchical Clustering.',
        'logistic regression': 'Logistic Regression is a supervised learning algorithm used for binary or multiclass classification. It uses a logistic function to model the probability.',
        'decision tree': 'Decision Trees are supervised learning algorithms that split data based on features to create a tree-like model of decisions.',
        'random forest': 'Random Forest is an ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.',
        'svm': 'Support Vector Machine (SVM) is a powerful supervised learning algorithm that finds the optimal boundary between classes.',
        'k-means': 'K-Means is an unsupervised learning algorithm that partitions data into K clusters by minimizing within-cluster variance.',
        'pca': 'Principal Component Analysis (PCA) is a dimensionality reduction technique that identifies the main variance directions in data.',
        'hierarchical clustering': 'Hierarchical Clustering builds a hierarchy of clusters, useful for understanding data relationships at different levels.',
        'accuracy': 'Accuracy is the proportion of correct predictions. For classification: (TP + TN) / (TP + TN + FP + FN).',
        'silhouette score': 'Silhouette Score measures how similar an object is to its own cluster compared to other clusters (range: -1 to 1).',
        'overfitting': 'Overfitting occurs when a model learns the training data too well, performing poorly on new data. Use techniques like cross-validation and regularization.',
        'underfitting': 'Underfitting occurs when a model is too simple to capture patterns. Use more complex models or collect more features.',
        'cross validation': 'Cross-validation divides data into folds to better estimate model performance and reduce variance.',
        'help': 'I can help you with questions about ML algorithms, accuracy scores, data analysis techniques, and best practices. Ask about any algorithm or concept!',
    }
    
    for key, value in responses.items():
        if key in lower:
            return value
    
    return "I can help with ML algorithms and analysis! Ask me about supervised/unsupervised learning, specific algorithms, accuracy metrics, or best practices."

if __name__ == '__main__':
    app.run(debug=True, port=5555)
