gnb = GaussNB(X_train, y_train)
y_pred = gnb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("GaussianNB %):", metrics.accuracy_score(y_test, y_pred)*100)
