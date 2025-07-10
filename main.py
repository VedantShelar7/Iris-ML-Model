import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Show the first 5 rows
print(df.head())
 
import seaborn as sns
import matplotlib.pyplot as plt

# Replace numbers with actual species names (0 -> Setosa, etc.)
df['species'] = df['species'].replace({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Create pairplot
sns.pairplot(df, hue='species')
plt.show()

from sklearn.model_selection import train_test_split

# Separate inputs and outputs
X = df.drop('species', axis=1)  # features (all columns except species)
y = df['species']               # labels (just the species column)

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print sizes to confirm
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create the model
model = SVC()  # SVC = Support Vector Classifier

# Train the model on training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate: compare predictions with actual labels
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Predict a new flower
sample = [[6.1, 2.8, 4.7, 1.2]]  # sepal length, sepal width, petal length, petal width
prediction = model.predict(sample)
print("Predicted Species:", prediction[0])
