
# 🛍️ Online Retail Customer Segmentation

## 📌 Project Overview

Online Retail Customer Segmentation is a data analysis and machine learning application that groups customers based on their purchasing behavior.

The application uses **RFM Analysis (Recency, Frequency, Monetary)** and the **K-Means Clustering** algorithm to identify different customer segments. It provides interactive visualizations, customer insights, and downloadable analysis results through a Streamlit dashboard.

## 🎯 Objectives

- Analyze customer purchasing behavior.
- Calculate RFM metrics for each customer.
- Group similar customers using K-Means clustering.
- Visualize customer segments through interactive charts.
- Explore individual customer information.
- Download customer-level results and segment summaries.

## ✨ Features

- **Interactive Dashboard:** View important business metrics and customer insights.
- **Dataset Upload:** Upload a retail transaction dataset in CSV format.
- **Data Cleaning:** Handle missing values, convert dates and prices, and filter invalid sales transactions.
- **RFM Analysis:** Calculate Recency, Frequency, and Monetary values.
- **K-Means Clustering:** Group customers into a user-selected number of segments.
- **Customer Segment Analysis:** Compare customer groups using charts and summary tables.
- **Customer Explorer:** Search, filter, and inspect individual customers.
- **Model Evaluation:** Display the silhouette score to help evaluate cluster separation.
- **Download Results:** Export customer segmentation results and segment summaries as CSV files.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly

## 📊 RFM Analysis

RFM analysis uses three important customer metrics:

| Metric | Meaning | Interpretation |
|---|---|---|
| Recency | Days since the customer's latest purchase | Lower values indicate more recent purchases |
| Frequency | Number of unique invoice dates associated with a customer | Higher values indicate purchases across more invoice dates |
| Monetary | Total revenue from the customer's retained transactions | Higher values indicate greater spending |

These metrics are standardized before applying K-Means clustering.

## 🤖 Machine Learning Algorithm

The application uses **K-Means Clustering**, an unsupervised machine learning algorithm.

### Workflow

1. Load the retail transaction dataset.
2. Clean the data and convert columns to appropriate data types.
3. Calculate RFM metrics for each customer.
4. Standardize the RFM features using `StandardScaler`.
5. Apply K-Means clustering.
6. Assign a segment to each customer.
7. Display charts, tables, and evaluation results.
8. Allow users to download the results.

The number of clusters can be adjusted through the application sidebar.

## 📁 Project Structure

```text
online-customer-segmentation/
│
├── app.py
├── requirements.txt
├── online_retail.csv
└── README.md
```

The dataset file is optional if users upload their own CSV through the application.

## 📋 Dataset Requirements

Upload a CSV file containing these columns:

- `CustomerID`
- `InvoiceDate`
- `Quantity`
- `UnitPrice`

Column names such as `UnitPrice ($)` are also supported by the application's column normalization.

The application calculates revenue as:

```python
Revenue = Quantity * UnitPrice
```

Customer IDs are treated as text. Invalid dates and missing required values are removed. Non-positive quantities and prices are excluded, and cancellation invoices are excluded when an invoice number column is available and begins with `C`.

**Note:** Review the dataset documentation to determine how returns, cancellations, and refunds should be handled for your analysis.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/online-customer-segmentation.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Navigate to the project folder

```bash
cd online-customer-segmentation
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```text
streamlit
pandas
scikit-learn
plotly
```

## 🚀 Deployment

This application can be deployed using **Streamlit Community Cloud**.

1. Push the project files to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the project repository.
5. Set `app.py` as the main file.
6. Deploy the application.

After successful deployment, you will receive a shareable application URL.

## 📈 Evaluation

The application displays the silhouette score as a measure of how well the customer clusters are separated.

A higher silhouette score generally indicates better-separated clusters. The score is not a prediction accuracy percentage, and the best number of clusters should also be evaluated based on business needs.

## 🔮 Future Enhancements

- Add customer segment labels based on their RFM profiles.
- Include additional customer behavior visualizations.
- Compare different cluster counts.
- Add automated recommendations for each customer segment.
- Improve data quality checks and reporting.
- Support larger datasets and additional retail transaction formats.

## 👩‍💻 Author

**Doddipatla Lakshmi Priyanka**

B.Tech – Electronics and Communication Engineering (ECE)

## 📄 License

This project is intended for educational and learning purposes. Add an appropriate open-source license if you want others to reuse or modify the code under defined terms.
