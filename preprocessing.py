import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """ Load data from a CSV file """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """ Preprocess the data: encode, scale, and split """
    # Encode categorical data
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['Geographical Location'] = label_encoder.fit_transform(data['Geographical Location'])

    # Scaling numerical features
    scaler = StandardScaler()
    numerical_features = ['Time Since Last Purchase', 'Total Number of Orders', 'Purchase Frequency',
                          'Number of Website Visits', 'Number of Items Viewed', 'Age', 'Total Spend']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Splitting the dataset
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load the data from the specified path
    data = load_data('data/merged_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
