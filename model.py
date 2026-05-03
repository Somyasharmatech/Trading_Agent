from sklearn.ensemble import RandomForestClassifier
import logging

def split_data(df, train_ratio=0.7):
    """
    Time-series aware train-test split without random shuffling.
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

def train_model(train_df, feature_cols):
    """
    Trains a RandomForestClassifier.
    
    Args:
        train_df (pd.DataFrame): Training data.
        feature_cols (list): List of feature column names.
        
    Returns:
        RandomForestClassifier: Trained model.
    """
    logging.info("Training RandomForestClassifier...")
    X_train = train_df[feature_cols]
    y_train = train_df['Target']
    
    # Simple and stable Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model

def predict(model, df, feature_cols):
    """
    Generates predictions and confidences.
    
    Args:
        model: Trained model.
        df: Data to predict on.
        feature_cols: List of features.
        
    Returns:
        pd.Series: Predictions (0 or 1).
        pd.Series: Confidence probabilities for class 1.
    """
    X = df[feature_cols]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] # Probability of class 1 (BUY)
    
    return preds, probs

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    # Dummy test
    print("Model module ready.")
