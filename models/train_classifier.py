import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

def train_and_evaluate_model(dataset_path, model_output_path, results_output_path):    
    # Load the dataset.
    df = pd.read_csv(dataset_path)
    
    # Separate features (X) and labels (y).
    X = df.drop('label', axis=1)
    y = df['label']

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the Random Forest model.
    print("Training the Random Forest model on landmark data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

    # Save the trained model.
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_output_path}")

    # --- Visualization of Results ---
    # Generate the confusion matrix.
    conf_matrix = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    
    # Plot and save the new confusion matrix.
    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title('Confusion Matrix (Landmark Model)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Ensure the results directory exists.
    if not os.path.exists(results_output_path):
        os.makedirs(results_output_path)
    
    # Use a specific name for the new plot.
    plot_path = os.path.join(results_output_path, 'confusion_matrix_landmark.png')
    plt.savefig(plot_path)
    print(f"New confusion matrix saved to: {plot_path}")

if __name__ == '__main__':
    # Define relative paths, keeping script location in mind.
    dataset_csv_path = os.path.join('..', 'dataset', 'asl_landmarks.csv')
    model_save_path = os.path.join('..', 'checkpoints', 'asl_landmark_model.pkl')
    results_save_path = os.path.join('..', 'results')

    # Run the main function.
    train_and_evaluate_model(dataset_csv_path, model_save_path, results_save_path)
