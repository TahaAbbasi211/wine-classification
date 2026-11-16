# Wine Classification

A machine learning project that classifies wine samples into three classes (cultivars) from 13 chemical properties, comparing four classifiers.

## Dataset

The Wine dataset from scikit-learn (`load_wine`, originally from the UCI Machine Learning Repository):

- 178 samples
- 13 numerical features (chemical properties such as alcohol, magnesium, flavanoids and color intensity)
- 3 target classes

## Method

- Stratified train/test split: 75% train, 25% test (`random_state=42`)
- Pipelines with `StandardScaler` for Logistic Regression and KNN; the tree models need no scaling
- Models compared: Logistic Regression, K-Nearest Neighbors (k=5), Decision Tree, Random Forest (100 trees)
- Evaluation: accuracy comparison across models, plus a classification report and confusion matrix for the best model
- Exploratory plots: class counts, alcohol vs color intensity, feature correlation heatmap

## Technologies

- Python
- Pandas, Matplotlib, Seaborn
- Scikit-learn

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run the project

```bash
python main.py
```

`main.py` prints the dataset summary, trains and compares the models, and shows the plots.

## Project structure

- `main.py` — data loading, exploration, model training and evaluation
- `Jupiter.ipynb` — exploratory notebook
- `requirements.txt` — project dependencies
- `.gitignore` — files excluded from Git

## Limitations

An educational project on a small, clean, built-in dataset. Results depend on the train/test split, and picking the best model on the test set is optimistic — cross-validation would be the more rigorous choice for comparing models.
