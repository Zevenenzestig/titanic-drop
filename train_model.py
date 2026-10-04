import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Загружаем датасет
df = pd.read_csv("train.csv")

# 2. Выбираем признаки
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
target = "Survived"

X = df[features]
y = df[target]

# 3. Делим данные на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Подготовка числовых признаков
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# 5. Подготовка категориального признака
categorical_features = ["Sex"]
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 6. Объединяем обработку данных
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 7. Создаём модель
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=6
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# 8. Обучение
pipeline.fit(X_train, y_train)

# 9. Проверка точности
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Точность модели: {accuracy * 100:.2f}%")
print("\nОтчёт классификации:")
print(classification_report(y_test, predictions))

# 10. Сохраняем модель
joblib.dump(pipeline, "titanic_model.pkl")
print("\nМодель сохранена в titanic_model.pkl")
