import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
import os
import argparse
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Import the AI insights generator
try:
    from ai_insights import AIInsightGenerator, generate_report_with_insights
    AI_INSIGHTS_AVAILABLE = True
except ImportError:
    AI_INSIGHTS_AVAILABLE = False
    print("AI insights module not available. Running without AI explanations.")

# Load the dataset
def load_data(file_path):
    print(f"Loading data from {file_path}")
    
    # Try to detect and report encoding or format issues
    try:
        data = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)
        
        # Check if any unexpected columns are present
        expected_columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 
                            'InvoiceDate', 'Price', 'Customer ID', 'Country']
        unexpected_cols = [col for col in data.columns if col not in expected_columns]
        if unexpected_cols:
            print(f"WARNING: Found unexpected columns: {unexpected_cols}")
        
        # Check for rows with unexpected format
        for col in expected_columns:
            if col not in data.columns:
                print(f"WARNING: Missing expected column: {col}")
        
        # Check for any malformed data in key columns
        malformed = data[data['Invoice'].astype(str).str.match(r'^[0-9C]+$') == False]
        if len(malformed) > 0:
            print(f"WARNING: Found {len(malformed)} rows with non-standard invoice format")
            print(malformed.head())
        
        print(f"Dataset shape: {data.shape}")
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try with different encoding or parsing options
        print("Attempting to load with different options...")
        data = pd.read_csv(file_path, encoding='latin1', sep=None, engine='python')
        print(f"Dataset shape: {data.shape}")
        return data

# Preprocess the data
def preprocess_data(data):
    print("Preprocessing data...")
    
    # Convert InvoiceDate to datetime
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d/%m/%y %H:%M')
    
    # Extract date components
    data['Year'] = data['InvoiceDate'].dt.year
    data['Month'] = data['InvoiceDate'].dt.month
    data['Day'] = data['InvoiceDate'].dt.day
    data['DayOfWeek'] = data['InvoiceDate'].dt.dayofweek
    data['Hour'] = data['InvoiceDate'].dt.hour
    
    # Calculate total price
    data['TotalPrice'] = data['Quantity'] * data['Price']
    
    # Filter out returns (negative quantities) and canceled orders
    data = data[data['Quantity'] > 0]
    data = data[~data['Invoice'].astype(str).str.startswith('C')]
    
    # Fix data types
    if 'Customer ID' in data.columns:
        data['Customer ID'] = pd.to_numeric(data['Customer ID'], errors='coerce')
    
    return data

# Exploratory Data Analysis
def perform_eda(data):
    print("Performing exploratory data analysis...")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(data[['Quantity', 'Price', 'TotalPrice']].describe())
    
    # Top selling products
    top_products = data.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 Products by Quantity Sold:")
    print(top_products)
    
    # Sales by country
    sales_by_country = data.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 Countries by Total Sales:")
    print(sales_by_country)
    
    # Time-based analysis
    monthly_sales = data.groupby('Month')['TotalPrice'].sum()
    day_of_week_sales = data.groupby('DayOfWeek')['TotalPrice'].sum()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    monthly_sales.plot(kind='bar')
    plt.title('Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    
    plt.subplot(1, 2, 2)
    day_of_week_sales.plot(kind='bar')
    plt.title('Sales by Day of Week')
    plt.xlabel('Day (0=Monday, 6=Sunday)')
    plt.ylabel('Total Sales')
    
    plt.tight_layout()
    plt.savefig('sales_analysis.png')
    
    return top_products, sales_by_country

# Feature Engineering
def engineer_features(data):
    print("Engineering features...")
    
    # Store the description mapping before aggregation
    descriptions = None
    if 'Description' in data.columns:
        descriptions = data[['StockCode', 'Description']].drop_duplicates()
    
    # Aggregate data by product and date
    product_daily = data.groupby(['StockCode', 'Year', 'Month', 'Day']).agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'Price': 'mean'  # Using mean in case price varies
    }).reset_index()
    
    # Create a proper date field
    product_daily['Date'] = pd.to_datetime(
        product_daily['Year'].astype(str) + '-' + 
        product_daily['Month'].astype(str) + '-' + 
        product_daily['Day'].astype(str)
    )
    
    # Calculate DayOfWeek from Date
    product_daily['DayOfWeek'] = product_daily['Date'].dt.dayofweek
    
    # Sort by product and date
    product_daily = product_daily.sort_values(['StockCode', 'Date'])
    
    # Create lagged features (previous day's sales)
    product_daily['Lag1_Quantity'] = product_daily.groupby('StockCode')['Quantity'].shift(1)
    
    # Create rolling mean features (sales trend)
    product_daily['Rolling7_Quantity'] = product_daily.groupby('StockCode')['Quantity'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Add back the descriptions if available
    if descriptions is not None and not descriptions.empty:
        product_daily = product_daily.merge(descriptions, on='StockCode', how='left')
    
    # Drop rows with NaN values after creating lagged features
    product_daily = product_daily.dropna(subset=['Lag1_Quantity'])
    
    return product_daily

# Build and train demand prediction model
def build_demand_model(data):
    print("Building demand prediction model...")
    
    # Define features and target
    features = ['Lag1_Quantity', 'Rolling7_Quantity', 'Month', 'Day', 'DayOfWeek']
    target = 'Quantity'
    
    # Split the data into features and target
    X = data[features]
    y = data[target]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = float('-inf')
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
        results[name] = {'model': model, 'mse': mse, 'mae': mae, 'r2': r2}
        
        if r2 > best_score:
            best_score = r2
            best_model = model
    
    print(f"\nBest model: {max(results.items(), key=lambda x: x[1]['r2'])[0]}")
    
    # Create directory if it doesn't exist
    model_dir = 'dist/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the best model to the models directory
    model_path = os.path.join(model_dir, 'demand_prediction_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
    
    return best_model, X_test, y_test, results

# Calculate restock recommendation
def calculate_restock_recommendations(data, model, last_n_days=30, safety_stock_factor=1.5, lead_time=7):
    print("Calculating restock recommendations...")
    
    # Get the latest date in the dataset
    latest_date = data['Date'].max()
    
    # Filter data for the last N days to analyze recent trends
    recent_data = data[data['Date'] >= (latest_date - timedelta(days=last_n_days))]
    
    # Group by product to get recent sales statistics
    product_stats = recent_data.groupby('StockCode').agg({
        'Quantity': ['sum', 'mean', 'std'],
        'Price': 'mean'
    })
    
    product_stats.columns = ['Total_Quantity', 'Avg_Daily_Quantity', 'Std_Daily_Quantity', 'Avg_Price']
    
    # Prepare data for prediction (next 7 days)
    forecast_dates = [latest_date + timedelta(days=i) for i in range(1, 8)]
    
    restock_recommendations = {}
    
    for stock_code in product_stats.index:
        # Filter data for this product
        product_data = data[data['StockCode'] == stock_code].copy()
        
        if len(product_data) < 10:  # Skip products with insufficient data
            continue
            
        # Get the latest product data for prediction
        latest_product_data = product_data.sort_values('Date').iloc[-1]
        
        # Prepare features for prediction
        forecast_features = []
        for date in forecast_dates:
            features = {
                'Lag1_Quantity': latest_product_data['Quantity'],
                'Rolling7_Quantity': product_stats.loc[stock_code, 'Avg_Daily_Quantity'],
                'Month': date.month,
                'Day': date.day,
                'DayOfWeek': date.dayofweek
            }
            forecast_features.append(features)
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecast_features)
        
        # Make predictions
        predicted_demand = model.predict(forecast_df)
        total_predicted_demand = np.sum(predicted_demand)
        
        # Calculate safety stock
        safety_stock = safety_stock_factor * product_stats.loc[stock_code, 'Std_Daily_Quantity'] * np.sqrt(lead_time)
        
        # Calculate recommended restock amount
        restock_amount = total_predicted_demand + safety_stock
        
        # Try to get the description from the product data
        description = "Unknown"
        if 'Description' in product_data.columns and not product_data['Description'].isna().all():
            description = product_data['Description'].iloc[0]
        
        # Store recommendations
        restock_recommendations[stock_code] = {
            'predicted_weekly_demand': total_predicted_demand,
            'safety_stock': safety_stock,
            'recommended_restock': restock_amount,
            'avg_price': product_stats.loc[stock_code, 'Avg_Price'],
            'Description': description  # Add description directly
        }
    
    # Convert to DataFrame
    restock_df = pd.DataFrame.from_dict(restock_recommendations, orient='index')
    
    # Round numerical columns
    restock_df['predicted_weekly_demand'] = restock_df['predicted_weekly_demand'].round(0)
    restock_df['safety_stock'] = restock_df['safety_stock'].round(0)
    restock_df['recommended_restock'] = restock_df['recommended_restock'].round(0)
    
    # Sort by recommended restock amount
    restock_df = restock_df.sort_values('recommended_restock', ascending=False)
    
    # Save recommendations to CSV
    restock_df.to_csv('restock_recommendations.csv')
    
    return restock_df

# Generate AI insights if available
def generate_ai_insights(restock_df, feature_data, gemini_api_key=None, gemini_model=None):
    """Generate AI insights if the module is available"""
    if not AI_INSIGHTS_AVAILABLE:
        print("AI insights module not available. Skipping AI analysis.")
        return None
    
    print(f"Generating AI-powered insights and analysis using Google Gemini...")
    try:
        # Ensure reports directory exists
        reports_dir = 'dist/reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate report with path to reports directory
        output_path = os.path.join(reports_dir, 'inventory_insights_report.html')
        report = generate_report_with_insights(restock_df, feature_data, 
                                              gemini_api_key=gemini_api_key,
                                              gemini_model=gemini_model,
                                              save_html=True,
                                              output_path=output_path)
        
        # Save the markdown report to the reports directory
        md_path = os.path.join(reports_dir, 'inventory_insights_report.md')
        with open(md_path, 'w') as f:
            f.write(report)
        
        print(f"AI insights generated and saved to '{md_path}' and '{output_path}'")
        return report
    except Exception as e:
        print(f"Error generating AI insights: {e}")
        return None

# Sample random rows from the dataset
def sample_dataset(data, n_samples=500, random_state=42):
    """
    Create a smaller dataset by sampling random rows.
    
    Args:
        data (pd.DataFrame): The original dataset
        n_samples (int): Number of samples to draw
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: A sampled subset of the data
    """
    print(f"Sampling {n_samples} random rows from dataset of {len(data)} rows")
    
    if len(data) <= n_samples:
        print("Dataset already smaller than requested sample size. Using full dataset.")
        return data
    
    # Sample randomly but with a fixed random seed for reproducibility
    sampled_data = data.sample(n=n_samples, random_state=random_state)
    
    print(f"Sampled dataset shape: {sampled_data.shape}")
    return sampled_data

# Main function
def main():
    # Set up argument parser for API keys
    parser = argparse.ArgumentParser(description='Inventory Management with AI Insights')
    parser.add_argument('--gemini_api_key', type=str, help='API key for Google Gemini')
    parser.add_argument('--file_path', type=str, default='dataset/supermarket_data.csv',
                        help='Path to the dataset file')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='Sample size to use (0 for full dataset)')
    args = parser.parse_args()
    
    # Set file path
    file_path = args.file_path
    
    # Load data
    data = load_data(file_path)
    
    # Sample data if requested
    if args.sample_size > 0:
        data = sample_dataset(data, n_samples=args.sample_size)
    
    # Preprocess data
    processed_data = preprocess_data(data)
    
    # Perform EDA
    top_products, top_countries = perform_eda(processed_data)
    
    # Engineer features
    feature_data = engineer_features(processed_data)
    
    # Build demand prediction model
    model, X_test, y_test, model_results = build_demand_model(feature_data)
    
    # Calculate restock recommendations
    restock_recommendations = calculate_restock_recommendations(feature_data, model)
    
    # Ensure directory exists for saving CSV
    reports_dir = 'dist/reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save recommendations to CSV in reports directory
    csv_path = os.path.join(reports_dir, 'restock_recommendations.csv')
    restock_recommendations.to_csv(csv_path)
    
    # Print top recommendations
    print("\nTop 10 Restock Recommendations:")
    print(restock_recommendations[['Description', 'predicted_weekly_demand', 'recommended_restock']].head(10))
    
    # Generate AI insights if available
    gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    
    if gemini_api_key and AI_INSIGHTS_AVAILABLE:
        insights = generate_ai_insights(restock_recommendations, feature_data, 
                                        gemini_api_key=gemini_api_key,
                                        gemini_model=gemini_model)
    
    print(f"\nDone! Restock recommendations saved to '{csv_path}'")
    if AI_INSIGHTS_AVAILABLE:
        print(f"AI-powered insights saved to '{os.path.join(reports_dir, 'inventory_insights_report.md')}'")

if __name__ == "__main__":
    main()
