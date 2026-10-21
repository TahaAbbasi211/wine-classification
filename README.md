# Wine Classification

## Overview
A machine learning project that classifies wine samples into three classes using chemical properties.

## Dataset
The project uses the Wine dataset provided by scikit-learn.

- 178 samples
- 13 numerical features
- 3 target classes

## Models
The following classification algorithms are compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

## Technologies
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Evaluation
Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The evaluation results are saved to `model_results.csv`.

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

## Project Structure

- `main.py`: Data processing, model training, and evaluation
- `requirements.txt`: Project dependencies
- `model_results.csv`: Model comparison results
- `.gitignore`: Files excluded from Git

## Limitations

This is an educational project using a built-in dataset. Model performance may vary depending on the train-test split and model configuration.