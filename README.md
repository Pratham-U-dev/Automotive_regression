# Automotive Fuel Efficiency Regression

<div align="center">

[![GitHub repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/<your-username>/<your-repository>)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)](https://jupyter.org/)

[View Notebook](#running-the-notebook) • [Dataset](#dataset) • [Models](#models-and-evaluation) • [Results](#results) 

</div>

A machine learning project that analyzes automotive sensor data and predicts fuel efficiency in miles per gallon (MPG). The project explores relationships between vehicle speed, engine performance, fuel consumption, air flow, acceleration, and other measurements collected through an OBD-II interface.

---

## Overview

Fuel efficiency is influenced by several driving and engine-related factors, including vehicle speed, engine RPM, throttle position, acceleration, and fuel-flow rate. This project applies exploratory data analysis, feature engineering, data preprocessing, and regression techniques to model fuel efficiency from automotive monitoring data.

The analysis is based on the case study **"Fuel Efficiency Versus Speed"** from the *Machine Learning for Engineers: Automotive Monitoring* course.

---

## Project Workflow

```mermaid
flowchart LR
    A[Load Automotive Data] --> B[Inspect & Clean Data]
    B --> C[Feature Engineering]
    C --> D[Train-Test Split]
    D --> E[Scale Features]
    E --> F[Train Regression Models]
    F --> G[Predict MPG]
    G --> H[Evaluate Models]
    H --> I[Visualize Results]

    style A fill:#4CAF50,stroke:#388E3C,color:#fff
    style F fill:#2196F3,stroke:#1976D2,color:#fff
    style H fill:#FF9800,stroke:#F57C00,color:#fff
    style I fill:#9C27B0,stroke:#7B1FA2,color:#fff
```

---

## Objectives

- Analyze automotive sensor data collected through an OBD-II interface.
- Investigate how vehicle speed and engine parameters affect fuel efficiency.
- Calculate and model fuel efficiency in miles per gallon (MPG).
- Prepare and clean the dataset for machine learning.
- Train regression models to predict MPG.
- Evaluate model performance using standard regression metrics.
- Visualize relationships between automotive measurements and fuel consumption.

---

## Dataset

The dataset contains automotive monitoring and fuel-consumption measurements, including:

| Feature | Description |
|---------|-------------|
| `time` | Timestamp of measurement |
| `Average fuel consumption (MPG)` | Average fuel efficiency |
| `Average speed (mph)` | Average vehicle speed |
| `Vehicle speed (mph)` | Instantaneous vehicle speed |
| `Engine RPM (rpm)` | Engine revolutions per minute |
| `MPG` | Calculated instantaneous fuel consumption (target) |
| `Calculated instant fuel rate (gal./h)` | Instantaneous fuel flow rate |
| `Distance travelled (miles)` | Distance covered |
| `Fuel used (gallon)` | Fuel consumed |
| `Instant engine power (hp)` | Engine power output |
| `MAF air flow rate (g/sec)` | Mass air flow sensor reading |
| `Oxygen sensor 1 Wide Range Current (mA)` | O₂ sensor current |
| `Oxygen sensor 1 Wide Range Equivalence ratio` | Air-fuel ratio indicator |
| `Throttle position (%)` | Throttle opening percentage |
| `Vehicle acceleration (g)` | Vehicle acceleration |

The target variable used in the regression analysis is:

```text
MPG — Miles per gallon
```

The original calculated fuel-consumption column is renamed to `MPG` during preprocessing.

---

## Screenshots

### Exploratory Data Analysis

<div align="center">
  <img width="571" height="418" alt="image" src="https://github.com/user-attachments/assets/f4778f73-6336-405f-985e-882ecbc05851" />
  <img width="543" height="413" alt="image" src="https://github.com/user-attachments/assets/d3c12fd6-6e52-4216-a1bc-cd4177b1fe30" />

    
  <img width="560" height="413" alt="image" src="https://github.com/user-attachments/assets/2d6de485-ed45-465d-9840-73de0b0ed7af" />

  <p><em>Relationships between vehicle speed, RPM, and fuel efficiency</em></p>
</div>

### Model Predictions

<div align="center">
  <img width="680" height="492" alt="image" src="https://github.com/user-attachments/assets/3a074642-fbe7-4a97-b4a6-3b757d324555" />

  <p><em>Actual vs. Predicted MPG values from regression models</em></p>
</div>

### Learning History

<div align="center">
  <img width="582" height="413" alt="image" src="https://github.com/user-attachments/assets/609e5d83-4389-49c1-bf2c-5ef77f4e3f0c" />


  <p><em>loss vs validation loss</em></p>
</div>

### 3D plot

<div align="center">
  <img width="487" height="407" alt="image" src="https://github.com/user-attachments/assets/2d91cc17-6f25-4e4d-8aa5-3ed198af85a8" />


  <p><em>Speed vs Acceleration vs RPM</em></p>
</div>

---

## Technologies Used

<div align="center">

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core programming language |
| ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) | Interactive notebook environment |
| ![NumPy](https://img.shields.io/badge/-NumPy-4D77CF?style=flat-square&logo=numpy&logoColor=white) | Numerical computing |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data manipulation and analysis |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square&logo=python&logoColor=white) | Data visualization |
| ![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=flat-square&logo=python&logoColor=white) | Statistical visualization |
| ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | Machine learning library |
| ![XGBoost](https://img.shields.io/badge/-XGBoost-FF9900?style=flat-square&logo=xgboost&logoColor=white) | Gradient boosting framework |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | Neural network framework |
| ![Keras](https://img.shields.io/badge/-Keras-D00000?style=flat-square&logo=keras&logoColor=white) | High-level neural network API |

</div>

---

## Machine Learning Pipeline

```mermaid
flowchart TD
    subgraph Data_Preparation["Data Preparation"]
        A1[Load Data] --> A2[Clean Missing Values]
        A2 --> A3[Feature Selection]
        A3 --> A4[Train-Test Split 80/20]
    end

    subgraph Preprocessing["Preprocessing"]
        B1[Feature Scaling] --> B2[Normalization]
    end

    subgraph Modeling["Modeling"]
        C1[Linear Regression] --> C2[XGBoost Regressor]
        C2 --> C3[Neural Network Keras]
    end

    subgraph Evaluation["Evaluation"]
        D1[MAE] --> D2[MAPE]
        D2 --> D3[R² Score]
    end

    Data_Preparation --> Preprocessing
    Preprocessing --> Modeling
    Modeling --> Evaluation

    style Data_Preparation fill:#E3F2FD,stroke:#1976D2
    style Preprocessing fill:#E8F5E9,stroke:#388E3C
    style Modeling fill:#FFF3E0,stroke:#F57C00
    style Evaluation fill:#F3E5F5,stroke:#7B1FA2
```

---

## Models and Evaluation

The project includes regression-based machine learning experiments using tools such as:

- **Linear Regression** — Baseline model for understanding linear relationships.
- **XGBoost Regression** — Gradient boosting for capturing non-linear patterns.
- **Neural-Network Regression (Keras)** — Deep learning approach with multiple hidden layers.

### Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **MAE** | Mean Absolute Error | \(\frac{1}{n}\sum_{i=1}^{n} \|y_i - \hat{y}_i\|\) |
| **MAPE** | Mean Absolute Percentage Error | \(\frac{100\%}{n}\sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|\) |
| **R²** | Coefficient of Determination | \(1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}\) |

The notebook also uses early stopping during neural-network training to help reduce unnecessary training and limit overfitting.

---

## Project Structure

```text
automotive-fuel-efficiency/
├── Automotive
    ├── Automotive_regression.ipynb
├── automotive.txt
└── README.md
```

> Place the dataset file in the expected project directory, or update the data-loading path in the notebook before execution.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost keras tensorflow tqdm jupyter
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

<div align="center">

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Open in Jupyter](https://img.shields.io/badge/Open-Jupyter-F37626?style=for-the-badge&logo=jupyter)](https://jupyter.org/)

</div>

### Local Execution

```bash
jupyter notebook
```

Then open:

```text
Automotive_regression.ipynb
```

Run the notebook cells sequentially from top to bottom.

### Google Colab

1. Upload `Automotive_regression.ipynb` and `automotive.txt` to Google Colab.
2. Update the data-loading path if necessary.
3. Run all cells using **Runtime → Run all**.

---

## Example Data Loading

The notebook currently loads the dataset using:

```python
url = "/content/automotive.txt"
data = pd.read_csv(url)
```

For local execution, update the path if necessary:

```python
data = pd.read_csv("automotive.txt")
```

---

## Results

The notebook produces:

- Dataset inspection and column summaries.
- Time-format transformations.
- Exploratory visualizations.
- Relationships between speed, RPM, acceleration, and MPG.
- Regression model predictions.
- Model evaluation using MAE, MAPE, and R².
- Comparisons between actual and predicted fuel-efficiency values.

Exact performance values may vary depending on preprocessing, feature selection, train-test split, random seed, and model hyperparameters.

---

## Model Comparison

```mermaid
quadrantChart
    title Model Performance Comparison
    x-axis "Lower Error" --> "Higher Error"
    y-axis "Lower R²" --> "Higher R²"
    quadrant-1 "Best Performance"
    quadrant-2 "Good R², Higher Error"
    quadrant-3 "Needs Improvement"
    quadrant-4 "Low R², Low Error"
    "Linear Regression": [0.35, 0.65]
    "XGBoost": [0.20, 0.85]
    "Neural Network": [0.25, 0.80]
```

> **Note:** Update the quadrant coordinates above with your actual model performance metrics.

---

## Key Insights

This project demonstrates how automotive sensor data can be used to estimate fuel efficiency. Measurements such as vehicle speed, engine RPM, throttle position, acceleration, fuel-flow rate, and air-flow rate provide useful signals for regression models.

However, fuel-efficiency predictions should be interpreted carefully because several sensor variables may be correlated with the target MPG value. In particular, variables that are directly calculated from fuel consumption may introduce data leakage if they are used as input features while predicting MPG.

---

## Limitations

- The model depends on the quality and reliability of the automotive sensor data.
- The dataset may not represent all driving conditions or vehicle types.
- Instantaneous MPG can be noisy and may contain extreme values.
- Random train-test splitting may not fully reflect the time-dependent nature of vehicle data.
- Features directly derived from fuel consumption should be reviewed carefully to prevent target leakage.
- Model performance may not generalize to vehicles or environments outside the dataset.



## Learning Resources

<div align="center">

[![Course Overview](https://img.shields.io/badge/Course-Overview-4285F4?style=for-the-badge)](https://apmonitor.com/pds)
[![Automotive Monitoring](https://img.shields.io/badge/Automotive-Monitoring-34A853?style=for-the-badge)](https://www.apmonitor.com/pds/index.php/Main/AutomotiveMonitoring)
[![Course Schedule](https://img.shields.io/badge/Course-Schedule-FBBC05?style=for-the-badge)](https://apmonitor.com/pds/index.php/Main/CourseSchedule)

</div>

- [Automotive Monitoring — Machine Learning for Engineers](https://www.apmonitor.com/pds/index.php/Main/AutomotiveMonitoring)
- [Machine Learning for Engineers Course Overview](https://apmonitor.com/pds)
- [Course Schedule](https://apmonitor.com/pds/index.php/Main/CourseSchedule)

---


## License

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This project is intended for educational and research purposes.

</div>

---

<div align="center">

**If you found this project helpful, please consider starring the repository!** ⭐

[Back to Top](#automotive-fuel-efficiency-regression)

</div>
