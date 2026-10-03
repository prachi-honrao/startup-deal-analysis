# Startup Funding & Deal Outcome Analysis – Streamlit Dashboard

An interactive **Exploratory Data Analysis (EDA)** project focused on understanding startup characteristics, financial metrics, and deal outcomes.

The project uses **Python for data analysis and visualization** and **Streamlit** to convert the analysis into an interactive web application.

## 📊 Project Overview

This project explores startup funding data to identify patterns and relationships between startup characteristics and funding deal outcomes.

The analysis covers factors such as:

- Industry
- City
- Business Stage
- Sales Channel
- Season
- Annual Sales
- Profit Margin
- Asking Amount
- Equity Offered
- Deal Amount
- Deal Made (Yes/No)

## 🔍 EDA Analysis

The Streamlit application includes:

### Univariate Analysis
Analysis of individual variables using appropriate charts and statistical summaries.

### Bivariate Analysis
Exploration of relationships between two variables to identify patterns and differences across startup categories.

### Multivariate Analysis
Analysis of multiple variables together to understand more complex relationships within the dataset.

## 🎛️ Interactive Filters

The dashboard provides sidebar filters for:

- **Season**
- **Industry**
- **City**
- **Business Stage**
- **Profit/Loss**
- **Sales Channel**

These filters allow users to dynamically explore different segments of the startup dataset.

## 📈 Dashboard Pages

The application contains the following pages:

- 🏠 **Home** – Project overview and key information
- 📊 **Univariate** – Individual variable analysis
- 📈 **Bivariate** – Relationship analysis between variables
- 🔎 **Multivariate** – Analysis involving multiple variables

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Plotly**
- **Streamlit**

## 📁 Project Structure

```text
startup-deal-analysis/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── shark_tank_corrected.csv
│
└── ...
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the project folder

```bash
cd startup-deal-analysis
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📂 Dataset

The dataset used for the analysis is located at:

```text
data/shark_tank_corrected.csv
```

## 🎯 Key Learning

This project helped strengthen practical skills in:

- Data cleaning and preprocessing
- Handling missing values and duplicate records
- Exploratory Data Analysis
- Data visualization
- Identifying trends and relationships
- Business-oriented data analysis
- Interactive dashboard development
- Building Streamlit applications

## 🚀 Project Goal

The main goal of this project is to transform raw startup funding data into meaningful visual insights and provide an interactive platform for exploring **startup characteristics and deal outcomes**.

---

**Built with Python, Pandas, Plotly & Streamlit.**