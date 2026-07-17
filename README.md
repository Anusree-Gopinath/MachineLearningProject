# ML Algorithm Analysis & Chat Interface

A comprehensive web application for machine learning algorithm analysis with an integrated AI chat assistant.

## Features

### 1. **File Upload**
   - Upload CSV or Excel datasets
   - Supports files up to 16MB
   - Displays data summary (rows, columns, data types)

### 2. **ML Algorithm Selection**
   - **Supervised Learning**: Logistic Regression, Decision Tree, Random Forest, SVM, Linear Regression
   - **Unsupervised Learning**: K-Means Clustering, PCA, Hierarchical Clustering
   - Select multiple algorithms at once
   - Specify target column for supervised learning

### 3. **Accuracy Scoring & Analysis**
   - Automatic model training and evaluation
   - Displays accuracy metrics for classification
   - R² and MSE scores for regression
   - Silhouette scores for clustering
   - Explained variance ratio for dimensionality reduction

### 4. **Floating AI Chat Assistant**
   - Compact chat window anchored at the bottom-right of the screen
   - Ask questions about the uploaded dataset, detected issues, and completed analysis
   - Get explanations of algorithms, metrics, target columns, and cleaning decisions
   - Useful for instant guidance while reviewing the analysis results

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Navigate to the project directory**:
   ```bash
   cd /path/to/MLEndOfModeuleProject
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Use the application**:
   - Upload your dataset (CSV or Excel)
   - Review the pre-analysis summary and decide whether to remove zero-variance columns
   - Select your target column (for supervised learning)
   - Choose the algorithms you want to test
   - Click "Analyze" to run the models
   - Use the floating chat icon at the bottom-right to ask questions about the file or the analysis results

## Project Structure

```
MLEndOfModeuleProject/
├── app.py                    # Flask backend application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── templates/
│   └── index.html           # Main webpage
└── uploads/                 # Uploaded files storage (auto-created)
```

## Algorithms Explained

### Supervised Learning
- **Logistic Regression**: Binary/multiclass classification using logistic function
- **Decision Tree**: Tree-based classification/regression
- **Random Forest**: Ensemble of decision trees
- **Support Vector Machine (SVM)**: Kernel-based boundary finding
- **Linear Regression**: Continuous value prediction

### Unsupervised Learning
- **K-Means**: Partitions data into K clusters
- **PCA**: Dimensionality reduction through variance analysis
- **Hierarchical Clustering**: Creates hierarchy of clusters

## Performance Metrics

### Classification
- **Accuracy**: (TP + TN) / Total predictions
- **Train/Test Split**: 80/20 by default

### Regression
- **R² Score**: Coefficient of determination (0-1)
- **MSE**: Mean Squared Error

### Clustering
- **Silhouette Score**: -1 to 1 (higher is better)
- **Optimal K**: Automatically calculated

## Tips for Best Results

1. **Data Quality**: Clean your data before uploading
2. **Feature Engineering**: Consider preprocessing before analysis
3. **Target Column**: Required for supervised learning
4. **Algorithm Selection**: Start with simpler algorithms first
5. **Large Datasets**: May take longer to process

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify `app.py`:
```python
app.run(debug=True, port=5001)  # Change to different port
```

### File Upload Issues
- Ensure file is in CSV or Excel format
- Check file size (max 16MB)
- Verify data contains valid values

### Analysis Errors
- Ensure target column is selected for supervised learning
- Check for missing or invalid data in dataset
- Verify columns match data types

## Requirements

See `requirements.txt` for all dependencies:
- Flask: Web framework
- pandas: Data manipulation
- scikit-learn: ML algorithms
- numpy: Numerical computing
- scipy: Scientific computing
- openpyxl: Excel file support

## License

Free to use for educational purposes.

## Support

For issues or questions:
1. Check the chat assistant in the application
2. Review the Troubleshooting section
3. Verify data format and requirements

Enjoy analyzing your data with multiple ML algorithms!
