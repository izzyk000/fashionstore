from preprocessing import load_data, preprocess_data
import numpy as np
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def build_model(optimizer='adam', activation='relu', layer1_units=128, layer2_units=64, dropout_rate=0.5, input_shape=(10,)):  # Adjust as needed
    """Builds and compiles a neural network model"""
    model = Sequential([
        Input(shape=input_shape),
        Dense(layer1_units, activation=activation),
        Dropout(dropout_rate),
        Dense(layer2_units, activation=activation),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data('data/merged_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Define the input shape based on the features of X_train
    input_shape = (X_train.shape[1],)

    # Wrap the model with KerasClassifier
    model = KerasClassifier(model=build_model, verbose=0, model__input_shape=input_shape)

    # Define the grid search parameters
    param_grid = {
        'model__optimizer': ['adam', 'rmsprop'],
        'model__activation': ['relu', 'tanh'],
        'model__layer1_units': [128, 256],
        'model__layer2_units': [64, 128],
        'model__dropout_rate': [0.5, 0.3],
        'batch_size': [10, 20],
        'epochs': [10, 50]
    }

    # Setup GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    # Output the best parameters and their respective scores
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Evaluate the best model found by the grid search on the test set
    y_pred = grid_result.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the best model
    best_model = grid_result.best_estimator_.model_
    best_model.save('churn_predictor_model.keras')
