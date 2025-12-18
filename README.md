# Lung Cancer Prediction using Random Forest

## Project Overview
A machine learning project that uses Random Forest to predict lung cancer risk based on patient data.

## Link to the dataset
https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer/data

```text
.
├── data/
│   ├── processed/              # Cleaned and preprocessed datasets
│   └── raw/                    # Original, immutable data
├── models/                     # Trained model files
├── notebooks/                  # Jupyter notebooks
│   └── 01_data_review.ipynb
└── src/                        # Source code
    ├── model.py                # Model training code
    ├── predict.py              # Prediction interface
    ├── preprocess.py           # Data preprocessing
    └── utils.py                # Utility functions
```

## Setup
1. Clone the repository
2. Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
1. Data Preprocessing:
```sh
python3 src/preprocess.py
```

2. Train Model:
```sh
python3 src/model.py
```

3. Make Predictions:
```sh
python3 src/predict.py
```

## Project Structure
- `data/`: Contains raw and processed datasets
- `models/`: Stores trained model files
- `notebooks/`: Jupyter notebooks for data analysis
- `src/`: Source code for the project
  - `model.py`: Implementation of the Random Forest model
  - `predict.py`: Functions for making predictions
  - `preprocess.py`: Data preprocessing pipeline
  - `utils.py`: Utility functions
