# GEMINI Analysis: SERS-ML Bacteria Classification

## Project Overview

This project is a machine learning pipeline for classifying two types of bacteria, Staphylococcus and Pseudomonas, based on their Surface-Enhanced Raman Spectroscopy (SERS) spectra. The project uses Python with libraries such as pandas, scikit-learn, and matplotlib. The overall goal is to build a classifier that can accurately distinguish between the two bacteria types from their spectral data.

The project follows a standard machine learning workflow:
1.  **Data Loading and Preprocessing:** Raw spectral data is loaded, cleaned, and transformed into a suitable format for machine learning. This includes interpolation, binning, and normalization.
2.  **Feature Engineering:** Relevant features, such as spectral peaks, are extracted from the preprocessed data.
3.  **Model Training:** A Random Forest classifier is trained on the extracted features. `GridSearchCV` is used to find the optimal hyperparameters for the model.
4.  **Model Evaluation and Saving:** The trained model is evaluated and saved for future use.

## Key Files

*   `main.py`: This is the main script that orchestrates the model training process. It loads the preprocessed data, performs feature engineering, and trains the final classifier.
*   `analysis.ipynb`: A Jupyter notebook that contains the initial data exploration, preprocessing, and various modeling experiments. This notebook is crucial for understanding the data and the development process.
*   `dataset.csv`: The raw spectral data in a long format, with columns for Wavenumber, Intensity, and Class.
*   `FINAL.csv`: The preprocessed and cleaned dataset used for training the model in `main.py`. This file is generated from the raw data after cleaning and restructuring.
*   `model.joblib`: The saved, trained scikit-learn machine learning model (a Random Forest classifier).
*   `data/`: This directory contains the original raw data files in various formats (`.csv`, `.xlsx`).
*   `analysis of spectrums/`: This directory contains various plots and images generated during the data analysis phase, such as spectra plots, peak analysis, and comparison of processing stages.

## Workflow

1.  **Data Ingestion:** The process starts with the raw spectral data located in the `data/` directory.
2.  **Preprocessing and Exploration:** The `analysis.ipynb` notebook is used to load the raw data, combine it into a single dataset, and perform extensive preprocessing. This includes:
    *   Pivoting the data from a long to a wide format.
    *   Interpolating the spectra to a common wavenumber grid.
    *   Binning the spectra to reduce dimensionality.
    *   Normalizing the data.
    *   The cleaned and restructured data is saved as `FINAL.csv`.
3.  **Model Training:** The `main.py` script loads the `FINAL.csv` file. It then:
    *   Extracts peak features from the spectra.
    *   Uses a `RandomForestClassifier` within a `GridSearchCV` pipeline to find the best model hyperparameters.
    *   Trains the final model on the feature-engineered data.
4.  **Model Persistence:** The trained model is saved to `model.joblib` for later use in prediction tasks.

## Building and Running

To run the main training pipeline, execute the following command in your terminal:

```bash
python main.py
```

This will load the preprocessed data, train the model, and print the best hyperparameters and cross-validation score.
