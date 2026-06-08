# Convex Optimization Project
*Comprehensive mathematical implementation of linear and non-linear supervised learning algorithms applied to the 
"Student Depression and Lifestyle" dataset (100k rows) from Kaggle.*

---

## Project Structure

```
Convex-Optimization/
├── data/
│   ├── raw/                          # Raw data downloaded from Kaggle
│   └── processed/                    # Cleaned and transformed data (clean_student_data.csv)
├── notebooks/
│   └── 1.0_data_preprocessing.ipynb  # Full EDA and preprocessing pipeline
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── pipeline.py               # Script to automate preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── 2.0_linear_svm.ipynb      # Convex SVM implementation and training
│   │   ├── 3.0_NN_training.ipynb     # Non-convex neural network implementation and training
│   │   └── binary_mlp_pytorch.pt     # Pre-trained PyTorch model weights
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

### Directory Roles

- `data/raw/`: Temporary storage for raw data downloaded from Kaggle
- `data/processed/`: Final preprocessed data, ready for modeling
- `notebooks/`: Jupyter notebooks for exploration, validation, and documentation
- `src/preprocessing/`: Python module to automate the preprocessing pipeline (modular)
- `src/models/`: Mathematical implementations of the convex SVM and neural network models

---

## Dataset Schema

| Column Name             | Description                                   | Data Type | Range/Values                       |
|-------------------------|-----------------------------------------------|-----------|------------------------------------|
| **Student_ID**          | Unique identifier for each student            | Integer   | Unique IDs                         |
| **Age**                 | Age of the student                            | Integer   | 18-24                              |
| **Gender**              | Gender of the student                         | String    | Male, Female                       |
| **Department**          | Field of study                                | String    | Engineering, Business, Arts, etc.  |
| **CGPA**                | Cumulative Grade Point Average                | Float     | 0.0 - 4.0                          |
| **Sleep_Duration**      | Average hours of sleep per night              | Float     | Continuous                         |
| **Study_Hours**         | Average hours spent studying per day          | Float     | Continuous                         |
| **Social_Media_Hours**  | Average hours spent on social media per day   | Float     | Continuous                         |
| **Physical_Activity**   | Average minutes of physical activity per week | Integer   | Continuous                         |
| **Stress_Level**        | Self-reported stress level                    | Integer   | 0-10                               |
| **Depression**          | Mental health status                          | Boolean   | True (Depression), False (Healthy) |
| **Transportation_Time** | Time from house to school                     | Integer   | 0-120                              |
| **Student_debt**        | Does the student have a debt                  | Integer   | 1 (Debt), 0 (No debt)              |
| **Part_Time_Job**       | Does the student have a partial Job           | Integer   | 1 (Job), 0 (No Job)                |
| **Living_status**       | How does the student live                     | Strng     | Alone, Family                      |

---

## Mathematical Justifications of Preprocessing Steps

### Why Standardize Features?

Standardization (centering + scaling) is critical for SGD convergence:

1. Prevents weight divergence: features with large amplitude (e.g., `Physical_Activity`) produce disproportionately 
large gradients compared to features with small ranges (e.g., `Age`).
2. Accelerates convergence: standardization makes the cost surface more symmetric, enabling more efficient exploration by SGD.
3. Mathematical formulation:
$$x_{\text{standardized}} = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}(x)}}$$

After transformation: $\mathbb{E}[x'] = 0$ and $\mathrm{Var}(x') = 1$ for each feature.

---

### Why Use Labels in {-1, 1}?

The hinge loss for linear SVM depends on a multiplicative label formulation:

$$\text{Hinge Loss} = \max(0, 1 - y_i (w^T x_i + b))$$

where:
- $y_i \in \{-1, +1\}$
- $w^T x_i + b$ is the decision score

Key property:
- If $y_i (w^T x_i + b) > 1$ → no penalty (correct classification with margin)
- If $y_i (w^T x_i + b) \le 1$ → positive loss

Using labels in {0, 1} breaks this multiplicative sign interpretation and does not align with the hinge loss formulation.

---

## Execution Commands


### **Prerequisites**
Recommended: Python >= 3.10.

1) Create and activate a virtual environment:


```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```


2) Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Run the preprocessing notebook end-to-end
Open the notebook to execute the preprocessing interactively:

```bash
jupyter notebook notebooks/1.0_data_preprocessing.ipynb
```

Run cells sequentially from the first cell through the export. The notebook downloads the Kaggle dataset, injects 
socio-economic variables, encodes categorical variables, standardizes numeric features, transforms the target, and exports 
the final CSV.

### 2. Run the automated pipeline
Execute the pipeline script from the project root to run the full flow non-interactively:

```bash
python src/preprocessing/pipeline.py
```

This script reproduces the same sequence as the notebook: Kaggle download, socio-economic feature injection, quality 
control, one-hot encoding, standardization, target transformation, and CSV export.

### 3. Reuse the pipeline in another Python script
Import pipeline functions for integration into model training or other scripts:

```python
from src.preprocessing.pipeline import (
    load_dataset,
    inject_socio_economic_features,
    quality_control,
    encode_categorical,
    standardize_numeric_features,
    transform_target_variable,
    save_processed_data
)

df = load_dataset()
df = inject_socio_economic_features(df)
df = quality_control(df)
df = encode_categorical(df)
df = standardize_numeric_features(df)
df = transform_target_variable(df)
save_processed_data(df)
```

### Which mode to use?
- Use the **notebook** to inspect and validate each step interactively.
- Use the **`pipeline.py`** script to run the entire preprocessing and produce `data/processed/clean_student_data.csv`.
- Import the pipeline functions when integrating preprocessing into model training workflows.

---

## `src/models/` Structure (Planned)

The `src/models/` directory will contain the following mathematical implementations:

### Notebook 2: Convex Linear SVM
- Optimizer: Stochastic Gradient Descent (SGD)
- Primal formulation:
$$\min_{w,b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i$$
with soft-margin constraints $y_i (w^T x_i + b) \ge 1 - \xi_i$.
- KKT conditions for optimality and Lagrangian dual formulation for convergence analysis.

### Notebook 3: Non-Convex Neural Network
- Architecture: Feedforward layers with ReLU activations
- Loss: Binary cross-entropy
- Optimizer: Adam
- No global convergence guarantees (non-convex loss surface)

---

### Required Python Dependencies

```
pandas>=1.5.0
numpy>=1.22.0
scikit-learn>=1.0.0
kagglehub>=0.1.0
jupyter>=1.0.0
```

(Install via `pip install -r requirements.txt`)

---

## Validation and Verification

After running the pipeline, verify the presence of `data/processed/clean_student_data.csv`:

```bash
# Print final dimensions and columns
python -c "import pandas as pd; df = pd.read_csv('data/processed/clean_student_data.csv'); print(f'Shape: {df.shape}'); print(f'Columns: {list(df.columns)}')"
```

---

**This project demonstrates a rigorous understanding of convex optimization mathematics and modern machine learning techniques.**

## Dataset Source
https://www.kaggle.com/datasets/aldinwhyudii/student-depression-and-lifestyle-100k-data
