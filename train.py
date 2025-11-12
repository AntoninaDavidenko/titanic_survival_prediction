import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from model import NeuralNetwork

df = pd.read_csv('Titanic-Dataset.csv')
print(f"Стовпці: {list(df.columns)}")
print(df.head(3))

# заповнення пропусків
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)

fare_median = df['Fare'].median()
df['Fare'].fillna(fare_median, inplace=True)


print("\nСтовпці до видалення", list(df.columns))
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
print("Стовбці після видалення:", list(df.columns))

# One-Hot для розділення стовпців
# One-Hot для Sex та Embarked
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# заміняємо false/true на 0/1
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

X = df.drop('Survived', axis=1) # дані
y = df['Survived'] # правильний результат


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nРазмер тренировочной выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

# масштабування
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)


num_features = X_train_scaled.shape[1]
print(f"Количество признаков: {num_features}")


model = NeuralNetwork(input_size=num_features)

# параметри
learning_rate = 0.001
epochs = 100

# Loss и optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()

    # Forward
    pred = model(X_train_tensor)
    loss = loss_fn(pred, y_train_tensor)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # тести
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = loss_fn(test_pred, y_test_tensor)
            predicted = (test_pred > 0.5).float()
            accuracy = (predicted == y_test_tensor).float().mean()
            print(
                f"Epoch {epoch}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}, Accuracy={accuracy.item() * 100:.1f}%")

print("\nFinished")
torch.save(model.state_dict(), 'titanic_model.pth')

from minio import Minio
from minio.error import S3Error

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # False -> without HTTPS
)

if not client.bucket_exists("titanic"):
    client.make_bucket("titanic")
    print(f"Created bucket titanic")

try:
    client.fput_object("titanic", "titanic-model", "D:/PyCharmProjects/rabbitmq_test/titanic_model.pth")
    client.fput_object("titanic", "titanic-scaler", "D:/PyCharmProjects/rabbitmq_test/scaler.pkl")
    print("Uploaded successfully.")
except S3Error as e:
    print(f"Error uploading file: {e}")
