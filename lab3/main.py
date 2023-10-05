import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('Credit_Screening.dat', delimiter=';')

X = data.iloc[:, :-2]
y = data['desired1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_k = None
best_accuracy = 0

for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Best k value: {best_k}")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Contingency table:")
print(conf_matrix)

error_rate = 100 * (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(y_test)
print(f"Error rate in %: {error_rate}%")
