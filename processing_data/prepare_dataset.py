import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

def load_data(csv_path):
    """
    Loads the CSV data into a pandas DataFrame.
    
    Args:
        csv_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv('/Users/dhruv590/Projects/Fetch/data/data.csv')
    return df

def map_labels(df):
    """
    Maps categorical text labels to integer labels.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
    
    Returns:
        pd.DataFrame, dict, dict: 
            - Updated DataFrame with integer labels.
            - Mapping dictionary for Sentiment.
            - Mapping dictionary for Category.
    """
    # Map Sentiment
    sentiment_mapping = {label: idx for idx, label in enumerate(df['Sentiment'].unique())}
    df['sentiment_label'] = df['Sentiment'].map(sentiment_mapping)
    
    # Map Category
    category_mapping = {label: idx for idx, label in enumerate(df['Category'].unique())}
    df['category_label'] = df['Category'].map(category_mapping)
    
    return df, sentiment_mapping, category_mapping

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.
    
    Returns:
        pd.DataFrame, pd.DataFrame: Training and Testing DataFrames.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[['sentiment_label', 'category_label']])
    return train_df, test_df

def save_data(train_df, test_df, output_dir, sentiment_mapping, category_mapping):
    """
    Saves the training and testing DataFrames to CSV files and label mappings to a JSON file.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        output_dir (str): Directory to save the processed data.
        sentiment_mapping (dict): Mapping from sentiment labels to integers.
        category_mapping (dict): Mapping from category labels to integers.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Save label mappings
    label_mappings = {
        'sentiment_mapping': sentiment_mapping,
        'category_mapping': category_mapping
    }
    
    with open(os.path.join(output_dir, 'label_mappings.json'), 'w') as f:
        json.dump(label_mappings, f, indent=4)
    
    print(f"Processed data saved to {output_dir}")

def main():
    # Define paths
    csv_path = 'data/your_data.csv'  # Replace with your actual CSV file path
    output_dir = 'processed_data'
    
    # Load data
    df = load_data(csv_path)
    print("Data Loaded Successfully")
    
    # Map labels
    df, sentiment_mapping, category_mapping = map_labels(df)
    print("Labels Mapped to Integers")
    print(f"Sentiment Mapping: {sentiment_mapping}")
    print(f"Category Mapping: {category_mapping}")
    
    # Split data
    train_df, test_df = split_data(df)
    print("Data Split into Training and Testing Sets")
    print(f"Training Set Size: {train_df.shape[0]}")
    print(f"Testing Set Size: {test_df.shape[0]}")
    
    # Save processed data with label mappings
    save_data(train_df, test_df, output_dir, sentiment_mapping, category_mapping)

if __name__ == "__main__":
    main()
