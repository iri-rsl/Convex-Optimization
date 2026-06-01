Comprehensive mathematical implementation of linear and non-linear supervised learning algorithms applied to the "Student Depression and Lifestyle" dataset (100k rows) from Kaggle.

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
│   └── models/                       # Placeholder for SVM and neural network implementations
├── README.md                         # This file
└── [requirements.txt]               # (Optional) Python dependencies
```

### Directory Roles

- `data/raw/`: Temporary storage for raw data downloaded from Kaggle
- `data/processed/`: Final preprocessed data, ready for modeling
- `notebooks/`: Jupyter notebooks for exploration, validation, and documentation
- `src/preprocessing/`: Python module to automate the preprocessing pipeline (modular)
- `src/models/`: Mathematical implementations of the convex SVM and neural network models

---

## Dataset Schema

| Column Name | Description | Data Type | Range/Values | Processing |
|---|---|---|---|---|
| **Student_ID** | Unique identifier | Integer | Unique IDs | Removed |
| **Age** | Student age | Integer | 18–24 | Standardized |
| **Gender** | Gender | String | Male, Female | One-hot encoded |
| **Department** | Field of study | String | Engineering, Business, Arts | One-hot encoded |
| **CGPA** | Cumulative Grade Point Average | Float | 0.0–4.0 | Standardized |
| **Sleep_Duration** | Hours of sleep per night | Float | Continuous | Standardized |
| **Study_Hours** | Study hours per day | Float | Continuous | Standardized |
| **Social_Media_Hours** | Hours on social media per day | Float | Continuous | Standardized |
| **Physical_Activity** | Minutes of activity per week | Integer | Continuous | Standardized |
| **Stress_Level** | Self-reported stress level | Integer | 0–10 | Standardized |
| **Depression** | Mental health status | Boolean | True / False | Converted to {-1, 1} |

---

The pipeline follows exactly six sequential steps:

### Step 1: Loading via Kagglehub
- Direct download using the Kaggle Dataset API
- Dataset: `aldinwhyudii/student-depression-and-lifestyle-100k-data`
- Format: CSV with 100,000 observations

### Step 2: Quality Control
- Check for missing values (`isnull().sum()`) per column
- Remove the non-informative identifier `Student_ID` (unique integer)
  - Rationale: The identifier carries no predictive signal and should not be used for model training

### Step 3: One-Hot Encoding (Categorical)
- Affected columns: `Gender`, `Department`, etc.
- Parameters: `drop='first'` to avoid multicollinearity
- Result: Independent binary indicator variables (0/1)

### Step 4: Numeric Standardization (Scaling)
- Columns standardized (7 features):
  - `Age`, `CGPA`, `Sleep_Duration`, `Study_Hours`, `Social_Media_Hours`, `Physical_Activity`, `Stress_Level`
- Operation: $x_{\text{scaled}} = \frac{x - \mu}{\sigma}$
- Guarantees: mean = 0, std = 1 for each feature

### Step 5: Target Transformation
- Source column: `Depression` (Boolean)
  - `True` = Probable depression
  - `False` = Healthy
- Convert to {**-1, 1**} binary labels:
  - `True` → `1`
  - `False` → `-1`
- Rationale: Required for the mathematical formulation of the SVM hinge loss

### Step 6: Export
- Final format: CSV (`data/processed/clean_student_data.csv`)
- Indices removed to avoid redundancy

---

## Mathematical Justifications

### Why Standardize Features?

Standardization (centering + scaling) is critical for SGD convergence:

1. Prevents weight divergence: features with large amplitude (e.g., `Physical_Activity`) produce disproportionately large gradients compared to features with small ranges (e.g., `Age`).
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

### Prerequisites
Install required Python packages:

```bash
pip install pandas numpy scikit-learn kagglehub jupyter
```

Activate your virtual environment if applicable.

### 1. Run the notebook end-to-end
Open the notebook to execute the preprocessing interactively:

```bash
jupyter notebook notebooks/1.0_data_preprocessing.ipynb
```

Run cells sequentially from the first cell through the export. The notebook downloads the Kaggle dataset, injects socio-economic variables, encodes categorical variables, standardizes numeric features, transforms the target, and exports the final CSV.

### 2. Run the automated pipeline
Execute the pipeline script from the project root to run the full flow non-interactively:

```bash
python src/preprocessing/pipeline.py
```

This script reproduces the same sequence as the notebook: Kaggle download, socio-economic feature injection, quality control, one-hot encoding, standardization, target transformation, and CSV export.

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

### Module 1: Convex Linear SVM
- Optimizer: Stochastic Gradient Descent (SGD)
- Primal formulation:
$$\min_{w,b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i$$
with soft-margin constraints $y_i (w^T x_i + b) \ge 1 - \xi_i$.
- KKT conditions for optimality and Lagrangian dual formulation for convergence analysis.

### Module 2: Non-Convex Neural Network
- Architecture: Feedforward layers with ReLU activations
- Loss: Binary cross-entropy
- Optimizers: SGD with momentum / Adam
- No global convergence guarantees (non-convex loss surface)

---

## Future Integration Notes

### Additional Features (Phase 2)
The following socio-economic features will be added later:
- Student debt
- Transportation costs
- Housing type and costs
- Employment status and income

Strategy: Add these columns after Step 2 (Quality Control) and apply the same encoding and scaling as current features. The SVM and neural network architectures will remain unchanged.

### Required Python Dependencies

```
pandas>=1.5.0
numpy>=1.22.0
scikit-learn>=1.0.0
kagglehub>=0.1.0
jupyter>=1.0.0
```

(Install via `pip install -r requirements.txt` if available.)

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
