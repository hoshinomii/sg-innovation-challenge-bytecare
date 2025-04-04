# ZenSupply - Demand Prediction & Restock Recommendation System

This project implements a machine learning solution to predict product demand and provide restock recommendations based on historical sales data.

## Overview

The system analyzes historical sales data to identify patterns and trends, then uses these insights to predict future demand and recommend optimal restock quantities for each product. The recommendations take into account:

- Historical sales patterns
- Seasonal trends
- Day-of-week effects
- Safety stock requirements
- Lead time for ordering

## Files

- `training.py`: Trains the demand prediction model and generates initial restock recommendations
- `predict.py`: Uses the trained model to generate new recommendations on fresh data
- `restock_recommendations.csv`: Output file with recommended restock quantities
- `demand_prediction_model.pkl`: Serialized trained model
- `install_dependencies.sh`: Script to install required dependencies
- `ai_insights.py`: Generates AI-powered insights and reports
- `use_azure_openai.py`: Script to use Azure OpenAI for generating insights
- `azure_helpers.py`: Helper functions for Azure OpenAI integration

## Data

The system uses transaction data from the `supermarket_data.csv` file, which should contain the following columns:

- `Invoice`: Invoice number
- `StockCode`: Product code
- `Description`: Product description
- `Quantity`: Quantity sold
- `InvoiceDate`: Date and time of sale
- `Price`: Unit price
- `Customer ID`: Customer identifier
- `Country`: Country where sale was made

## How It Works

### Data Preprocessing

- Converts dates to datetime format
- Extracts time features (year, month, day, day of week)
- Filters out returns and canceled orders
- Calculates total price per transaction

### Feature Engineering

- Aggregates sales by product and date
- Creates lag features for time series analysis
- Calculates rolling averages to capture trends

### Model Training

The system trains multiple models and selects the best performing one:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Restock Recommendations

Recommendations are generated using:
1. Predicted demand for the next period
2. Safety stock calculation based on demand variability
3. Consideration of lead time

## Setup

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Install Dependencies

Run the following script to install the required dependencies:

```bash
./install_dependencies.sh
```

Alternatively, you can manually install the dependencies using pip:

```bash
pip install -r requirements.txt
pip install --force-reinstall --no-cache-dir google-generativeai>=0.7.0
```

### Environment Variables

Create a `.env` file in the project root directory and add the following environment variables:

```
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_DEPLOYMENT=<your_azure_openai_deployment>
```

### Training the Model

```bash
python training.py
```

This will:
- Load and preprocess the data
- Train the demand prediction model
- Save the model to disk
- Generate and save restock recommendations

### Generating New Recommendations

```bash
python predict.py
```

This will:
- Load the trained model
- Process new data
- Generate restock recommendations
- Save recommendations to a CSV file
- Generate HTML and Markdown reports

### Using Azure OpenAI for Insights

```bash
python use_azure_openai.py
```

This will:
- Load configuration and initialize Azure OpenAI client
- Generate AI-powered insights using Azure OpenAI
- Save insights to a file

## Results

The system outputs a CSV file with the following information for each product:

- `StockCode`: Product identifier
- `Description`: Product description
- `predicted_demand`: Predicted demand for the next period
- `safety_stock`: Recommended safety stock level
- `recommended_restock`: Recommended restock quantity
- `avg_price`: Average unit price
- `estimated_cost`: Estimated cost of restock

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- google-generativeai
- openai

Install requirements using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib google-generativeai openai
```

## Next Steps

Future improvements could include:
- More sophisticated models like ARIMA or Prophet for time series forecasting
- Integration with inventory management systems
- Web dashboard for visualizing recommendations
- Automatic alerts for products that need restocking