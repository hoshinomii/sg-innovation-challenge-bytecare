import pandas as pd
import argparse
import os

def sample_dataset(file_path, output_path, n_samples=500, random_state=42):
    """
    Sample random rows from a dataset and save to a new file
    
    Args:
        file_path (str): Path to the original dataset
        output_path (str): Path to save the sampled dataset
        n_samples (int): Number of samples to draw
        random_state (int): Random seed for reproducibility
    """
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {data.shape}")
    
    if len(data) <= n_samples:
        print("Dataset already smaller than requested sample size. Using full dataset.")
        sampled_data = data
    else:
        # Sample randomly but with a fixed random seed for reproducibility
        sampled_data = data.sample(n=n_samples, random_state=random_state)
    
    print(f"Sampled dataset shape: {sampled_data.shape}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save the sampled data
    sampled_data.to_csv(output_path, index=False)
    print(f"Sampled data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Sample random rows from a dataset')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save the sampled CSV file')
    parser.add_argument('--samples', type=int, default=500,
                      help='Number of samples to draw (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    sample_dataset(args.input, args.output, args.samples, args.seed)

if __name__ == "__main__":
    main()
