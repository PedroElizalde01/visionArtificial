import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from labels import int_to_label
import huFile

EXTRA_HU_MOMENTS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'extra_huMoments.csv')
HU_MOMENTS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'huMoments.csv')

def load_hu_moments_data():
    csv_file1 = EXTRA_HU_MOMENTS_PATH
    csv_file2 = HU_MOMENTS_PATH

    hu_moments_df1 = pd.read_csv(csv_file1)
    hu_moments_df2 = pd.read_csv(csv_file2)

    hu_moments_df = pd.concat([hu_moments_df1, hu_moments_df2], ignore_index=True)
    
    X = hu_moments_df['hu_moments'].apply(eval).tolist()
    y = hu_moments_df['label']

    return X, y

def train_knn_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)

    y_pred_knn = knn_classifier.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"k-Nearest Neighbors Accuracy: {accuracy_knn}")

    return knn_classifier

def save_knn_classifier(knn_classifier, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(knn_classifier, output_path)

def main():
    X, y = load_hu_moments_data()
    knn_classifier = train_knn_classifier(X, y)
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'knn_super_model.joblib')
    save_knn_classifier(knn_classifier, output_path)

if __name__ == "__main__":
    main()