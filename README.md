# SERS Bacteria Classification

This project uses machine learning to classify two types of bacteria, *Staphylococcus* and *Pseudomonas*, based on their Surface-Enhanced Raman Spectroscopy (SERS) spectra.

## Overview

The project implements a full machine learning pipeline in Python:

1.  **Data Loading and Preprocessing:** Raw spectral data is loaded, cleaned, interpolated, binned, and normalized.
2.  **Feature Engineering:** Spectral peaks are identified and used as features for the model.
3.  **Model Training:** A `RandomForestClassifier` is trained using `GridSearchCV` to find the optimal hyperparameters.
4.  **Model Evaluation and Saving:** The best model is evaluated and saved to `model.joblib` for future use.

## Data Acquisition and Processing

The experimental data was collected using the following methods, based on the provided documentation:

-   **Bacterial Strains:** The studies analyzed *Staphylococcus aureus* (Gram-positive) and *Klebsiella pneumoniae* (Gram-negative). The project dataset also includes *Pseudomonas*.
-   **Sample Preparation:** Bacterial colonies were grown in a nutrient medium. A few drops were then placed on a SERS-active substrate, inactivated, and dried at room temperature.
-   **SERS Substrates:** The spectra were enhanced using two types of substrates: gold nanoparticles (~40 nm) on a silver-coated glass slide, and carbon nanowalls coated with a thin gold film.
-   **Instrumentation:** A Nicolet Almega XR spectrometer was used to record the spectra.
-   **Measurement Parameters:** A 532 nm laser (20 mW) was used for excitation. The spectral range was 400-3100 cm⁻¹. To minimize fluorescence, a 30-second photobleaching step was performed before each measurement.

## How It Works: The Science Behind SERS Classification

The classification is possible due to the ability of Surface-Enhanced Raman Spectroscopy (SERS) to generate a unique "molecular fingerprint" for different types of bacteria.

1.  **Molecular Fingerprinting:** When a laser interacts with a bacterium, it scatters off the various molecules in the cell wall (lipids, proteins, peptidoglycans, etc.). The Raman spectroscopy technique measures the energy shifts in the scattered light, which correspond to the specific vibrational modes of these molecules. The resulting spectrum is a unique fingerprint of the bacterium's surface chemistry.

2.  **Signal Enhancement (The "SERS" part):** The standard Raman signal is extremely weak. SERS overcomes this by placing the bacteria on a nanostructured metallic surface (like the gold substrates used here). The laser excites plasmons in the metal, creating a powerful electromagnetic field that amplifies the Raman signal of nearby molecules by many orders of magnitude. This makes it possible to get a strong, clear fingerprint from a small number of bacterial cells.

3.  **Distinguishing Bacteria:** *Staphylococcus* and *Pseudomonas* have significantly different cell wall compositions. For example, *Staphylococcus* (Gram-positive) has a thick peptidoglycan layer, while *Pseudomonas* (Gram-negative) has a thin one plus an outer membrane of lipopolysaccharides. These structural differences result in measurably different SERS spectra.

4.  **Machine Learning:** While the spectra are distinct, the differences can be complex and subtle. The machine learning model (a Random Forest) is trained to recognize these complex patterns across the entire spectrum. It learns the specific spectral features that reliably distinguish one bacterial class from the other, enabling automated and accurate classification.

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
    git clone <your-repo-url>
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
