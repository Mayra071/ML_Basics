# Breast Cancer Prediction ML Pipeline

A comprehensive machine learning pipeline for breast cancer prediction using Logistic Regression and XGBoost classifiers. This project implements a complete MLOps workflow with data ingestion, preprocessing, model training, evaluation, and experiment tracking using MLflow.

## 🎯 Project Overview

This project predicts whether a breast tumor is benign (B) or malignant (M) based on features extracted from digitized images of fine needle aspirates (FNA) of breast masses. The dataset contains 569 samples with 30 numerical features derived from cell nuclei characteristics.

## ✨ Key Features

- **Modular Architecture**: Clean separation of concerns with dedicated components for each ML pipeline stage
- **Automated Data Pipeline**: Automated data ingestion, preprocessing, and train-test splitting
- **Multiple ML Models**: Logistic Regression and XGBoost with hyperparameter tuning
- **Comprehensive Evaluation**: Classification reports, confusion matrices, and ROC AUC metrics
- **Experiment Tracking**: MLflow integration for tracking experiments, parameters, and models
- **Docker Support**: Containerized deployment with Dockerfile
- **Jupyter Notebook**: Interactive analysis and visualization
- **Logging System**: Comprehensive logging throughout the pipeline
- **Artifact Management**: Automated saving of models, preprocessors, and visualizations

## 🏗️ Project Structure

```
week1_p1/
├── src/
│   └── Breast_Cancer_prediction/
│       ├── components/           # Core ML pipeline components
│       │   ├── data_ingestion.py      # Data loading and splitting
│       │   ├── data_transformation.py # Data preprocessing pipeline
│       │   ├── model_trainer.py       # Model training and evaluation
│       │   └── model_monitering.py    # Model monitoring utilities
|       |
│       ├── pipelines/            # End-to-end pipeline scripts
│       │   ├── data_preprocessing.py
│       │   ├── model_training.py
│       │   └── model_evaluation.py
│       ├── exception.py          # Custom exception handling
│       ├── logger.py             # Logging configuration
│       ├── mlflow_utils.py       # MLflow integration utilities
│       └── utils.py              # Utility functions
|
├── artifacts/                    # Generated artifacts and outputs
│   ├── data.csv                 # Raw dataset
│   ├── train.csv                # Training data
│   ├── test.csv                 # Test data
│   ├── model.pkl                # Best trained model
│   ├── preprocessor.pkl         # Data preprocessing pipeline
│   └── confussion/              # Confusion matrix visualizations
|
├── mlruns/                      # MLflow experiment tracking data
├── mlartifacts/                 # MLflow model artifacts
├── logs/                        # Application logs
├── notebook/
│   ├── Breast_Cancer.ipynb      # Interactive analysis notebook
│   └── data/                    # Notebook data files
├── app.py                       # Main application entry point
├── main.py                      # Model inspection utility
├── template.py                  # Template script
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Git (for cloning)

### Installation

1. **Create project from cookiecutter template:**
   ```bash
   cookiecutter https://github.com/drivendata/cookiecutter-data-science
   cd <your-project-name>
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Using venv
   python -m venv .venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n breast_cancer python=3.8
   conda activate breast_cancer
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MLflow (optional):**
   ```bash
   # Start MLflow tracking server (optional)
   mlflow server --host 0.0.0.0 --port 5000
   ```

### Running the Pipeline

1. **Run the complete ML pipeline:**
   ```bash
   python app.py
   ```

2. **Inspect the trained model:**
   ```bash
   python main.py
   ```

3. **Explore the Jupyter notebook:**
   ```bash
   jupyter notebook notebook/Breast_Cancer.ipynb
   ```

## 📊 Dataset Information

The dataset contains 569 samples with the following characteristics:
- **Target Variable**: `diagnosis` (B = Benign, M = Malignant)
- **Features**: 30 numerical features derived from cell nuclei measurements
- **Feature Categories**:
  - Mean values (radius, texture, perimeter, area, etc.)
  - Standard errors (se)
  - Worst values (worst)

### Sample Features:
- `radius_mean`, `texture_mean`, `perimeter_mean`
- `area_mean`, `smoothness_mean`, `compactness_mean`
- `concavity_mean`, `concave points_mean`
- And their corresponding `_se` and `_worst` variants

## 🔧 Pipeline Components

### 1. Data Ingestion (`data_ingestion.py`)
- Loads raw dataset from CSV
- Performs train-test split (80/20)
- Saves processed datasets to artifacts directory

### 2. Data Transformation (`data_transformation.py`)
- Handles missing values with median imputation
- Applies MinMax scaling for feature normalization
- Encodes target variable (B=0, M=1)
- Creates preprocessing pipeline for consistent transformations

### 3. Model Training (`model_trainer.py`)
- Implements two ML algorithms:
  - **Logistic Regression** with hyperparameter tuning
  - **XGBoost** with comprehensive parameter optimization
- Performs GridSearchCV for optimal hyperparameters
- Generates classification reports and confusion matrices
- Integrates with MLflow for experiment tracking

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrix visualization
- ROC AUC metrics
- Model comparison and selection

## 🤖 Machine Learning Models

### Logistic Regression
- **Hyperparameters tuned:**
  - C: [0.1, 1.0, 3.0]
  - solver: ['liblinear']
  - max_iter: [1000]
  - class_weight: ['balanced']

### XGBoost
- **Hyperparameters tuned:**
  - learning_rate: [0.05, 0.1]
  - subsample: [0.8, 1.0]
  - max_depth: [4, 6]
  - n_estimators: [150]
  - objective: ['binary:logistic']

## 📈 MLflow Integration

The project uses MLflow for comprehensive experiment tracking:

- **Experiment Management**: Organized experiments with clear naming
- **Parameter Logging**: All hyperparameters tracked automatically
- **Metric Tracking**: Performance metrics logged for each run
- **Model Artifacts**: Models saved with proper versioning
- **Visualization**: Confusion matrices and plots logged as artifacts

### MLflow Commands:
```bash
# Start MLflow UI
mlflow ui

# View experiments
mlflow experiments list

# Run specific experiment
mlflow run . --experiment-name "Breast_Cancer_Prediction"
```

## 🐳 Docker Deployment

The project includes Docker support for containerized deployment:

```bash
# Build Docker image
docker build -t breast-cancer-prediction .

# Run container
docker run -p 5000:5000 breast-cancer-prediction
```

## 📋 Dependencies

### Core Dependencies:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn==1.6.1` - Machine learning algorithms
- `xgboost` - Gradient boosting framework
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization

### MLOps Dependencies:
- `mlflow` - Experiment tracking and model management
- `dotenv` - Environment variable management

## 📊 Output Artifacts

After running the pipeline, the following artifacts are generated:

### Models and Preprocessors:
- `artifacts/model.pkl` - Best performing model with metadata
- `artifacts/preprocessor.pkl` - Data preprocessing pipeline

### Data Files:
- `artifacts/train.csv` - Training dataset
- `artifacts/test.csv` - Test dataset
- `artifacts/data.csv` - Original dataset

### Visualizations:
- `artifacts/confussion/logistic_regression_confusion.jpg`
- `artifacts/confussion/xgboost_confusion.jpg`

### Logs:
- `logs/` - Detailed execution logs with timestamps

## 🔍 Usage Examples

### Basic Usage:
```python
from src.Breast_Cancer_prediction.components.data_ingestion import DataIngestion
from src.Breast_Cancer_prediction.components.data_transformation import DataTransformation
from src.Breast_Cancer_prediction.components.model_trainer import ModelTrainer

# Run complete pipeline
data_ingestion = DataIngestion()
train_path, test_path = data_ingestion.initiate_data_ingestion()

data_transformation = DataTransformation()
train_data, test_data, _ = data_transformation.initiate_data_transformation(train_path, test_path)

model_trainer = ModelTrainer()
reports = model_trainer.initiate_model_trainer(train_data, test_data)
```

### Model Inspection:
```python
import pickle

# Load trained model
with open("artifacts/model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
report = model_data["report"]
print(f"Model accuracy: {report['report_dict']['accuracy']}")
```

## 🧪 Testing and Validation

The project includes comprehensive validation:
- Cross-validation during hyperparameter tuning
- Train-test split validation
- Classification report generation
- Confusion matrix analysis
- ROC AUC evaluation

## 📝 Logging

Comprehensive logging system tracks:
- Data ingestion progress
- Preprocessing steps
- Model training iterations
- Evaluation metrics
- Error handling and debugging information

Logs are stored in timestamped files in the `logs/` directory.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Manish Kumar** - *Initial work* - [YourGitHub](https://github.com/mayra071)

## 🙏 Acknowledgments

- University of Wisconsin Hospitals for the breast cancer dataset
- Scikit-learn community for excellent ML tools
- MLflow team for experiment tracking capabilities
- XGBoost developers for the gradient boosting framework

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: [aryam7842@gmail.com]
- Documentation: [Link to detailed docs]

---
