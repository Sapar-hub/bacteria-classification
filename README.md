# SERS Bacteria Classification

This project uses machine learning to classify two types of bacteria, *Staphylococcus* (Gram-positive) and *Pseudomonas* (Gram-negative), based on their Surface-Enhanced Raman Spectroscopy (SERS) spectra.

> **Note:** It was developed for the "Classification of bacteria by combination scattering spectra" project within the Scientific and Technical Conference of RTU MIREA, section "Applied Physics", May 2025. I do not plan to maintain or continue this project unless practical use is defined and real raw data is acquired. The project achieved an accuracy of 83% in separating two classes of bacteria.

## Overview

The project implements a full machine learning pipeline in Python:

1.  **Data Loading and Preprocessing:** Raw spectral data is loaded, cleaned, interpolated, binned, and normalized.
2.  **Feature Engineering:** Spectral peaks are identified and used as features for the model.
3.  **Model Training:** A `RandomForestClassifier` is trained using `GridSearchCV` to find the optimal hyperparameters.
4.  **Model Evaluation and Saving:** The best model is evaluated and saved to `model.joblib` for future use.

## Data Acquisition and Processing

The experimental data was collected using the following methods:

-   **Bacterial Strains:** The studies analyzed *Staphylococcus aureus* (Gram-positive) and *Klebsiella pneumoniae* (Gram-negative). The project dataset includes *Pseudomonas*.
-   **Sample Preparation:** Bacterial colonies were grown in a nutrient medium. A few drops were then placed on a SERS-active substrate, inactivated, and dried at room temperature.
-   **SERS Substrates:** The spectra were enhanced using two types of substrates: gold nanoparticles (~40 nm) on a silver-coated glass slide, and carbon nanowalls coated with a thin gold film.
-   **Instrumentation:** A Nicolet Almega XR spectrometer was used to record the spectra.
-   **Measurement Parameters:** A 532 nm laser (20 mW) was used for excitation. The spectral range was 400-3100 cm⁻¹. To minimize fluorescence, a 30-second photobleaching step was performed before each measurement.

## Project Structure

-   `main.py`: The main script to run the model training pipeline.
-   `analysis.ipynb`: A Jupyter Notebook with the initial data exploration, preprocessing, and modeling experiments.
-   `requirements.txt`: A list of Python dependencies required for the project.
-   `data/`: Contains the raw spectral data.
-   `model.joblib`: The final, trained machine learning model.
-   `FINAL.csv`: The preprocessed dataset used for training (generated from `analysis.ipynb`).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Sapar-hub/bacteria-classification.git
    cd SERS-ML
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The raw data from the `data/` directory is processed in the `analysis.ipynb` notebook, which generates the `FINAL.csv` file used for training.

To train the model, run the main script:

```bash
python main.py
```

This will load the preprocessed data from `FINAL.csv`, train the classifier, and save the final model as `model.joblib`. The script will also print the best hyperparameters found by GridSearchCV and the cross-validation score.
