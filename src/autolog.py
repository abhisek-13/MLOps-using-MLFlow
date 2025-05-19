import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner = "abhisek-13",repo_name = "MLOps-using-MLFlow", mlflow = True)

mlflow.set_tracking_uri("https://dagshub.com/abhisek-13/MLOps-using-MLFlow.mlflow")

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 7
n_estimators = 15

mlflow.autolog()
mlflow.set_experiment("experiment1")

with mlflow.start_run():
  rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
  rf.fit(X_train, y_train)
  
  y_pred = rf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)
  
  # mlflow.log_metric("accuracy", accuracy)
  # mlflow.log_param("max_depth", max_depth)
  # mlflow.log_param("n_estimators", n_estimators)
  
  # creating a confusion matrix plot
  plt.figure(figsize=(10, 7))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
  plt.xlabel('actual')
  plt.ylabel('predicted')
  plt.title('Confusion Matrix')
  
  # saving the plot
  plt.savefig("confusion_matrix.png")
  
  # mlflow.log_artifact("confusion_matrix.png")
  mlflow.log_artifact(__file__)
  
  # tags
  mlflow.set_tags({"author":"Abhisek","project":"Wine Classification"})
  
  # log the model
  # mlflow.sklearn.log_model(rf, "RandomForest model")
  
  print(f"Accuracy: {accuracy}")