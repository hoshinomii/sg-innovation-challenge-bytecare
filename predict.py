import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the AI insights generator
try:
    from ai_insights import AIInsightGenerator, generate_report_with_insights
    AI_INSIGHTS_AVAILABLE = True
except ImportError:
    AI_INSIGHTS_AVAILABLE = False
    print("AI insights module not available. Running without AI explanations.")

def load_model(model_path='dist/models/demand_prediction_model.pkl'):
    """Load the trained demand prediction model"""
    print(f"Loading model from {model_path}")
    
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        print("Please run training.py first to generate the model.")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def prepare_data(file_path):
    """Load and prepare new data for prediction"""
    print(f"Loading data from {file_path}")
    
    # Load the data
    data = pd.read_csv(file_path)
    
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
    
    return data

def engineer_prediction_features(data):
    """Engineer features for the new data"""
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
    
    # Sort by product and date
    product_daily = product_daily.sort_values(['StockCode', 'Date'])
    
    # Create lagged features (previous day's sales)
    product_daily['Lag1_Quantity'] = product_daily.groupby('StockCode')['Quantity'].shift(1)
    
    # Create rolling mean features (sales trend)
    product_daily['Rolling7_Quantity'] = product_daily.groupby('StockCode')['Quantity'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Drop rows with NaN values after creating lagged features
    product_daily = product_daily.dropna()
    
    return product_daily

def generate_restock_recommendations(data, model, forecast_days=7, safety_stock_factor=1.5, lead_time=7):
    """Generate restock recommendations based on the trained model"""
    # Store product descriptions in a separate DataFrame before they get lost
    try:
        # Check if original data contains descriptions
        raw_data = None
        if 'Description' not in data.columns:
            # If Description isn't in processed data, try to get it from original CSV
            try:
                raw_data = pd.read_csv('dataset/supermarket_data.csv')
                descriptions = raw_data[['StockCode', 'Description']].drop_duplicates()
                print(f"Loaded {len(descriptions)} product descriptions from original dataset")
            except Exception as e:
                print(f"Warning: Could not load descriptions from original data: {e}")
                # Create an empty descriptions frame with correct columns
                descriptions = pd.DataFrame(columns=['StockCode', 'Description'])
        else:
            # Extract descriptions from the provided data
            descriptions = data[['StockCode', 'Description']].drop_duplicates()
    except Exception as e:
        print(f"Warning: Error handling descriptions: {e}")
        # Create an empty descriptions frame as fallback
        descriptions = pd.DataFrame(columns=['StockCode', 'Description'])
    
    # Get the latest date in the dataset
    latest_date = data['Date'].max()
    
    # Calculate product statistics (last 30 days)
    recent_data = data[data['Date'] >= (latest_date - timedelta(days=30))]
    product_stats = recent_data.groupby('StockCode').agg({
        'Quantity': ['mean', 'std'],
        'Price': 'mean'
    })
    product_stats.columns = ['Avg_Daily_Quantity', 'Std_Daily_Quantity', 'Avg_Price']
    
    # Prepare data for forecast
    forecast_dates = [latest_date + timedelta(days=i) for i in range(1, forecast_days+1)]
    
    restock_recommendations = {}
    
    # Get unique stock codes
    stock_codes = data['StockCode'].unique()
    
    for stock_code in stock_codes:
        # Filter data for this product
        product_data = data[data['StockCode'] == stock_code].copy()
        
        if len(product_data) < 5:  # Skip products with insufficient data
            continue
            
        # Get the latest product data for prediction
        latest_product_data = product_data.sort_values('Date').iloc[-1]
        
        if stock_code not in product_stats.index:
            continue
            
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
        try:
            safety_stock = safety_stock_factor * product_stats.loc[stock_code, 'Std_Daily_Quantity'] * np.sqrt(lead_time)
            if np.isnan(safety_stock):
                safety_stock = total_predicted_demand * 0.2  # fallback: 20% of predicted demand
        except Exception:
            safety_stock = total_predicted_demand * 0.2  # fallback: 20% of predicted demand
        
        # Calculate recommended restock amount
        restock_amount = total_predicted_demand + safety_stock
        
        # Get average price from stats
        avg_price = product_stats.loc[stock_code, 'Avg_Price']
        
        # Try to get the description from the descriptions DataFrame
        description = "Unknown"
        if stock_code in descriptions['StockCode'].values:
            desc_row = descriptions[descriptions['StockCode'] == stock_code]
            if not desc_row.empty and not pd.isna(desc_row['Description'].iloc[0]):
                description = desc_row['Description'].iloc[0]
        
        # Calculate estimated cost
        estimated_cost = restock_amount * avg_price
        
        # Store recommendations
        restock_recommendations[stock_code] = {
            'predicted_weekly_demand': total_predicted_demand,  # Change key name for consistency
            'safety_stock': safety_stock,
            'recommended_restock': restock_amount,
            'avg_price': avg_price,
            'estimated_cost': estimated_cost,
            'Description': description
        }
    
    # Convert to DataFrame
    restock_df = pd.DataFrame.from_dict(restock_recommendations, orient='index')
    
    # Round numerical columns
    restock_df['predicted_weekly_demand'] = restock_df['predicted_weekly_demand'].round(0)  # Update key name
    restock_df['safety_stock'] = restock_df['safety_stock'].round(0)
    restock_df['recommended_restock'] = restock_df['recommended_restock'].round(0)
    restock_df['estimated_cost'] = restock_df['estimated_cost'].round(2)
    
    # Add a priority column
    restock_df['priority'] = pd.qcut(restock_df['recommended_restock'], 
                                     q=3, 
                                     labels=['Low', 'Medium', 'High'])
    
    # Sort by recommended restock amount
    restock_df = restock_df.sort_values('recommended_restock', ascending=False)
    
    return restock_df

def generate_ai_insights(restock_df, feature_data, gemini_api_key=None, gemini_model=None):
    """Generate AI insights on the prediction results"""
    if not AI_INSIGHTS_AVAILABLE:
        print("AI insights module not available. Skipping AI analysis.")
        return None
    
    print(f"Generating AI-powered insights using Google Gemini...")
    try:
        report = generate_report_with_insights(
            restock_df, 
            feature_data, 
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model
        )
        
        # Save the report to a file
        report_path = 'dist/reports/prediction_insights_report.md'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"AI insights generated and saved to '{report_path}'")
        return report
    except Exception as e:
        print(f"Error generating AI insights: {e}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate inventory restock predictions')
    parser.add_argument('--model_path', type=str, default='dist/models/demand_prediction_model.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--data_path', type=str, default='dataset/supermarket_data.csv',
                        help='Path to the dataset file')
    parser.add_argument('--output_path', type=str, default='dist/data/restock_recommendations.csv',
                        help='Path to save the recommendations file')
    parser.add_argument('--detailed_output', action='store_true',
                        help='Generate additional detailed CSV output files')
    parser.add_argument('--api_key', type=str, help='API key for OpenAI services')
    parser.add_argument('--gemini_api_key', type=str, help='API key for Google Gemini')
    parser.add_argument('--ai_provider', type=str, default=os.environ.get('DEFAULT_AI_PROVIDER', 'auto'), 
                        choices=['auto', 'gemini', 'openai', 'huggingface', 'template'],
                        help='AI provider to use for generating insights')
    parser.add_argument('--use_ai', action='store_true', help='Generate AI insights on predictions')
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    
    # Prepare data
    data = prepare_data(args.data_path)
    
    # Engineer features
    feature_data = engineer_prediction_features(data)
    
    # Generate recommendations
    recommendations = generate_restock_recommendations(feature_data, model)
    
    # Format output columns
    output_recommendations = recommendations.copy()
    
    # Convert numeric columns to integers where appropriate
    int_columns = ['predicted_weekly_demand', 'safety_stock', 'recommended_restock']
    output_recommendations[int_columns] = output_recommendations[int_columns].astype(int)
    
    # Save recommendations
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        output_recommendations.to_csv(args.output_path, index_label='StockCode')
        print(f"\n✅ Success: Restock recommendations CSV saved to '{args.output_path}'")
    except Exception as e:
        print(f"\n❌ Error saving CSV file: {e}")
    
    # Create additional detailed outputs if requested
    if args.detailed_output:
        # High priority items only
        try:
            high_priority = output_recommendations[output_recommendations['priority'] == 'High']
            high_priority_path = args.output_path.replace('.csv', '_high_priority.csv')
            high_priority.to_csv(high_priority_path, index_label='StockCode')
            print(f"✅ High priority items saved to '{high_priority_path}'")
            
            # Medium priority items
            medium_priority = output_recommendations[output_recommendations['priority'] == 'Medium']
            medium_priority_path = args.output_path.replace('.csv', '_medium_priority.csv')
            medium_priority.to_csv(medium_priority_path, index_label='StockCode')
            print(f"✅ Medium priority items saved to '{medium_priority_path}'")
            
            # Low priority items
            low_priority = output_recommendations[output_recommendations['priority'] == 'Low']
            low_priority_path = args.output_path.replace('.csv', '_low_priority.csv')
            low_priority.to_csv(low_priority_path, index_label='StockCode')
            print(f"✅ Low priority items saved to '{low_priority_path}'")
            
            # Summary statistics by priority
            summary_path = args.output_path.replace('.csv', '_summary.csv')
            summary = output_recommendations.groupby('priority').agg({
                'recommended_restock': ['count', 'sum'],
                'estimated_cost': 'sum'
            })
            summary.columns = ['item_count', 'total_units', 'total_cost']
            summary.to_csv(summary_path)
            print(f"✅ Summary statistics saved to '{summary_path}'")
        except Exception as e:
            print(f"❌ Error creating detailed outputs: {e}")
    
    print("\n" + "="*120)
    print("🔍 TOP RESTOCK RECOMMENDATIONS BY PRIORITY")
    print("="*120)
    
    # Display top 5 high priority items in table format
    high_priority = output_recommendations[output_recommendations['priority'] == 'High'].head(5)
    if not high_priority.empty:
        print("\n🔴 HIGH PRIORITY ITEMS")
        print("-"*120)
        # Print table header
        print(f"{'Stock Code':<12} {'Description':<40} {'Weekly Demand':<15} {'Restock Amt':<15} {'Est. Cost':<15}")
        print("-"*120)
        for idx, row in high_priority.iterrows():
            desc = row['Description'] if not pd.isna(row['Description']) else f"Product {idx}"
            # Truncate description if too long
            desc = desc[:37] + "..." if len(desc) > 37 else desc.ljust(37)
            print(f"{idx:<12} {desc:<40} {row['predicted_weekly_demand']:<15} {row['recommended_restock']:<15} ${row['estimated_cost']:<14.2f}")
    
    # Display top 5 medium priority items in table format
    medium_priority = output_recommendations[output_recommendations['priority'] == 'Medium'].head(5)
    if not medium_priority.empty:
        print("\n🟠 MEDIUM PRIORITY ITEMS")
        print("-"*120)
        # Print table header
        print(f"{'Stock Code':<12} {'Description':<40} {'Weekly Demand':<15} {'Restock Amt':<15} {'Est. Cost':<15}")
        print("-"*120)
        for idx, row in medium_priority.iterrows():
            desc = row['Description'] if not pd.isna(row['Description']) else f"Product {idx}"
            # Truncate description if too long
            desc = desc[:37] + "..." if len(desc) > 37 else desc.ljust(37)
            print(f"{idx:<12} {desc:<40} {row['predicted_weekly_demand']:<15} {row['recommended_restock']:<15} ${row['estimated_cost']:<14.2f}")
    
    # Display top 5 low priority items in table format
    low_priority = output_recommendations[output_recommendations['priority'] == 'Low'].head(5)
    if not low_priority.empty:
        print("\n🟢 LOW PRIORITY ITEMS")
        print("-"*120)
        # Print table header
        print(f"{'Stock Code':<12} {'Description':<40} {'Weekly Demand':<15} {'Restock Amt':<15} {'Est. Cost':<15}")
        print("-"*120)
        for idx, row in low_priority.iterrows():
            desc = row['Description'] if not pd.isna(row['Description']) else f"Product {idx}"
            # Truncate description if too long
            desc = desc[:37] + "..." if len(desc) > 37 else desc.ljust(37)
            print(f"{idx:<12} {desc:<40} {row['predicted_weekly_demand']:<15} {row['recommended_restock']:<15} ${row['estimated_cost']:<14.2f}")
    
    # Display summary statistics
    print("\n📊 SUMMARY STATISTICS")
    print("-"*80)
    for priority in ['High', 'Medium', 'Low']:
        priority_items = output_recommendations[output_recommendations['priority'] == priority]
        if not priority_items.empty:
            item_count = len(priority_items)
            total_units = priority_items['recommended_restock'].sum()
            total_cost = priority_items['estimated_cost'].sum()
            
            # Pick emoji based on priority
            emoji = "🔴" if priority == 'High' else "🟠" if priority == 'Medium' else "🟢"
            print(f"{emoji} {priority} Priority: {item_count} items, {int(total_units)} units, ${total_cost:.2f}")
    print("="*80)
    
    # Generate AI insights if requested
    if args.use_ai or os.environ.get('ALWAYS_USE_AI', '').lower() == 'true':
        print("\n🧠 Generating AI insights...")
        gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        
        if gemini_api_key and AI_INSIGHTS_AVAILABLE:
            insights = generate_ai_insights(
                recommendations, 
                feature_data,
                gemini_api_key=gemini_api_key,
                gemini_model=gemini_model
            )
            print("✅ AI insights generation complete!")
        else:
            if not gemini_api_key:
                print("❌ Missing Gemini API key. Set GEMINI_API_KEY environment variable or use --gemini_api_key")
            if not AI_INSIGHTS_AVAILABLE:
                print("❌ AI insights module not available. Make sure ai_insights.py is in the same directory.")

if __name__ == "__main__":
    main()
