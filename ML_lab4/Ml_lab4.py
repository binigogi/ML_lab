import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score

# Load dataset
filePath = r"C:\Users\binit\OneDrive\Documents\ML_lab2\Lab Session Data.xlsx"
xls = pd.ExcelFile(filePath)
marketingDf = pd.read_excel(xls, sheet_name="marketing_campaign")

# Data Preprocessing
marketingDf.dropna(inplace=True)
X = marketingDf[['Income', 'NumWebPurchases']]
y = marketingDf['Response']

# Train-test split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled = scaler.transform(XTest)

# Train kNN classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(XTrainScaled, yTrain)

# Predictions
yTrainPred = knn.predict(XTrainScaled)
yTestPred = knn.predict(XTestScaled)

# Function to evaluate model performance
def evaluateModel(yTrue, yPred, dataset):
    cm = confusion_matrix(yTrue, yPred)
    print(f"Confusion Matrix ({dataset}):\n", cm)
    print(f"Classification Report ({dataset}):\n", classification_report(yTrue, yPred))

evaluateModel(yTrain, yTrainPred, "Training")
evaluateModel(yTest, yTestPred, "Test")

# Regression Metrics (Simulated Price Prediction)
yActual = marketingDf['Income'].values[:len(XTest)]  # Using Income as a proxy for price
yPred = knn.predict(XTestScaled)

mse = mean_squared_error(yActual, yPred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((yActual - yPred) / yActual)) * 100
r2 = r2_score(yActual, yPred)

print(f"MSE: {mse}, RMSE: {rmse}, MAPE: {mape}, R2: {r2}")

# Scatter Plot of Training Data
np.random.seed(42)
XTrainSample = np.random.uniform(1, 10, (20, 2))
yTrainSample = np.random.choice([0, 1], size=20)
colors = ['blue' if label == 0 else 'red' for label in yTrainSample]
plt.scatter(XTrainSample[:, 0], XTrainSample[:, 1], c=colors)
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Training Data Scatter Plot")
plt.show()

# Generate Test Data and kNN Classification
testX = np.array([(x, y) for x in np.arange(0, 10, 0.1) for y in np.arange(0, 10, 0.1)])
testYPred = knn.predict(testX)
testColors = ['blue' if label == 0 else 'red' for label in testYPred]
plt.scatter(testX[:, 0], testX[:, 1], c=testColors, s=1)  # Removed cmap='coolwarm'
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("kNN Classification (k=3)")
plt.show()

# Test for Various k values and display in order
kValues = [1, 5, 10]
for k in sorted(kValues):  # Ensure k values are displayed in order
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(XTrainScaled, yTrain)
    testYPred = knn.predict(testX)
    testColors = ['blue' if label == 0 else 'red' for label in testYPred]
    plt.scatter(testX[:, 0], testX[:, 1], c=testColors, s=1)  # Removed cmap='coolwarm'
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title(f"kNN Classification (k={k})")
    plt.show()

# Hyperparameter Tuning with GridSearchCV
paramGrid = {'n_neighbors': range(1, 20)}
gridSearch = GridSearchCV(KNeighborsClassifier(), paramGrid, cv=5)
gridSearch.fit(XTrainScaled, yTrain)
print("Best k value:", gridSearch.best_params_)
