from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score



def train_model(X_train, X_test, y_train, y_test):
    
    """
    Trains and evaluates a Logistic Regression and a Random Forest model on the provided data.

    Args:
    X_train (array-like): The feature vectors of the training data.
    X_test (array-like): The feature vectors of the test data.
    y_train (array-like): The target labels of the training data.
    y_test (array-like): The target labels of the test data.

    Returns:
    None
    """
    
    output_filename = f"results/output_model_training.txt"
    outfile = open(output_filename, 'w', encoding='utf-8')
    
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the training data and transform it
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Initialize Logistic Regression model
    print("***********************Creating a Logistic Regression Model***********************\n", file = outfile)
    model = LogisticRegression()
    
    model.fit(X_train_vectorized, y_train)
    
    y_pred =  model.predict(X_test_vectorized)
    
    # Evaluate the model
    accuracy = model.score(X_test_vectorized, y_test)
    print("\nAccuracy:", accuracy, file = outfile)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix\n", file = outfile)
    print(cm, file = outfile)
    
    report = classification_report(y_test, y_pred)
    print("\nClassification Report\n", file = outfile)
    print(report, file = outfile)
    
    mse = mean_squared_error(y_test, y_pred)
    print("\nMean Squared Error: ", mse, file = outfile)
    
    print("\nCoefficient of determination: ", r2_score(y_test, y_pred), file = outfile)
    
    # Initialize Random Forest model
    print("\n***********************Creating a Random Forest Model***********************\n", file = outfile)
    new_model = RandomForestClassifier(n_estimators=50, random_state=42)
    new_model.fit(X_train_vectorized, y_train)
    
    # Predict ratings for test data
    y_pred = new_model.predict(X_test_vectorized)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy, file = outfile)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix\n", file = outfile)
    print(cm, file = outfile)
    
    report = classification_report(y_test, y_pred)
    print("\nClassification Report\n", file = outfile)
    print(report, file = outfile)
    
    mse = mean_squared_error(y_test, y_pred)
    print("\nMean Squared Error: ", mse, file = outfile)
    
    print("\nCoefficient of determination: ", r2_score(y_test, y_pred), file = outfile)
