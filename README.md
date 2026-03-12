# 🔺 Topological Data Analysis for Customer Segmentation & Behavioral Analysis in E-Commerce

> Using **Mapper** and **Persistent Homology** to uncover multi-scale customer behavioral patterns and build interpretable segmentation graphs for e-commerce purchase forecasting.

---

## 🎯 Problem

Traditional clustering methods like K-Means struggle to capture complex, non-linear relationships in customer behavior. This project applies **Topological Data Analysis (TDA)** - a mathematically rigorous approach that preserves the shape and structure of data - to segment e-commerce customers and evaluate whether topological features improve purchase prediction.

---

## 🔍 Approach

### 1. Exploratory Data Analysis
- Analyzed customer behavioral signals: purchase history, browsing patterns, cart activity, and session metadata
- Identified distributional patterns and feature correlations relevant to segmentation

### 2. Topological Feature Extraction
- **Mapper Algorithm**: Constructed low-dimensional graph representations of customer data to reveal clusters, bridges, and outlier groups not visible with standard methods
- **Persistent Homology**: Captured multi-scale topological features (connected components, loops) that encode behavioral structure across resolutions

### 3. Customer Segmentation
- Built interpretable segmentation graphs using Mapper outputs
- Identified distinct behavioral archetypes (e.g., high-intent browsers, one-time buyers, loyal repeat customers)

### 4. Predictive Modeling
- Evaluated whether TDA-derived topological features add predictive value for forecasting customer purchases
- Compared models with and without topological features to quantify their contribution

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| TDA | KeplerMapper, Ripser, Persim |
| Data Processing | Python, Pandas, NumPy |
| ML & Modeling | Scikit-learn |
| Visualization | Matplotlib, Seaborn, KeplerMapper visualizations |

---

## 📁 Repository Structure

```
├── tda_customer_segmentation.ipynb       # Full pipeline notebook with outputs
├── tda_customer_segmentation.py          # Clean Python script version
├── Dataset/                                   # Dataset files
├── Report.pdf       # Full project report
├── Presentation.mp4 # Video presentation
├── requirements.txt                           # Python dependencies
└── README.md
```

---

## 🎬 Presentation

A full video walkthrough of the project methodology and findings is included in [`Keertana Kappuram-DSC 214-Presentation.mp4`](./Keertana%20Kappuram-DSC%20214-%20Presentation.mp4).

---

## 📦 Dataset

The e-commerce behavioral dataset used in this project can be downloaded [here](https://docs.google.com/spreadsheets/d/1PslHPKdA8mj0TodGBR974fSVD2xLObaj/edit?usp=share_link&ouid=111531593127164351024&rtpof=true&sd=true).

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/keertanakappuram/TOPOLOGICAL-DATA-ANALYSIS-FOR-CUSTOMER-SEGMENTATION-AND-BEHAVIORAL-ANALYSIS-IN-E--COMMERCE.git
cd TOPOLOGICAL-DATA-ANALYSIS-FOR-CUSTOMER-SEGMENTATION-AND-BEHAVIORAL-ANALYSIS-IN-E--COMMERCE
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the notebook (with outputs) or the Python script
```bash
jupyter notebook Keertana_Kappuram_DSC_214_Code.ipynb
# or
python keertana_kappuram_dsc_214_code.py
```
