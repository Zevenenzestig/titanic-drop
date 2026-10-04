import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# Загружаем обученную модель
model = joblib.load("titanic_model.pkl")


def predict_survival():
    try:
        pclass = int(class_var.get())
        sex = sex_var.get()
        age = float(age_entry.get())
        sibsp = int(sibsp_entry.get())
        parch = int(parch_entry.get())
        fare = float(fare_entry.get())

        if age < 0 or sibsp < 0 or parch < 0 or fare < 0:
            raise ValueError

        passenger = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare
        }])

        prediction = model.predict(passenger)[0]
        probabilities = model.predict_proba(passenger)[0]
        survival_probability = probabilities[1] * 100

        if prediction == 1:
            result_label.config(
                text=f"ВЫЖИВЕТ\nВероятность: {survival_probability:.1f}%"
            )
        else:
            result_label.config(
                text=f"НЕ ВЫЖИВЕТ\nВероятность: {survival_probability:.1f}%"
            )

    except ValueError:
        messagebox.showerror(
            "Ошибка",
            "Проверьте введённые данные.\n"
            "Возраст и стоимость должны быть числами."
        )


# Главное окно
root = tk.Tk()
root.title("Titanic — прогноз выживания")
root.geometry("520x620")
root.resizable(False, False)

title = tk.Label(
    root,
    text="ПРОГНОЗ ВЫЖИВАНИЯ TITANIC",
    font=("Arial", 20, "bold")
)
title.pack(pady=20)

frame = tk.Frame(root)
frame.pack(pady=5)


def add_label(row, text):
    tk.Label(
        frame,
        text=text,
        font=("Arial", 12)
    ).grid(row=row, column=0, sticky="w", padx=10, pady=8)


add_label(0, "Класс билета:")
class_var = tk.StringVar(value="3")
class_box = ttk.Combobox(
    frame,
    textvariable=class_var,
    values=["1", "2", "3"],
    state="readonly",
    width=20
)
class_box.grid(row=0, column=1, padx=10, pady=8)

add_label(1, "Пол:")
sex_var = tk.StringVar(value="male")
sex_box = ttk.Combobox(
    frame,
    textvariable=sex_var,
    values=["male", "female"],
    state="readonly",
    width=20
)
sex_box.grid(row=1, column=1, padx=10, pady=8)

add_label(2, "Возраст:")
age_entry = tk.Entry(frame, width=23)
age_entry.insert(0, "25")
age_entry.grid(row=2, column=1, padx=10, pady=8)

add_label(3, "Братья/сёстры/супруги:")
sibsp_entry = tk.Entry(frame, width=23)
sibsp_entry.insert(0, "0")
sibsp_entry.grid(row=3, column=1, padx=10, pady=8)

add_label(4, "Родители/дети:")
parch_entry = tk.Entry(frame, width=23)
parch_entry.insert(0, "0")
parch_entry.grid(row=4, column=1, padx=10, pady=8)

add_label(5, "Стоимость билета:")
fare_entry = tk.Entry(frame, width=23)
fare_entry.insert(0, "20")
fare_entry.grid(row=5, column=1, padx=10, pady=8)

predict_button = tk.Button(
    root,
    text="ПРОГНОЗИРОВАТЬ",
    command=predict_survival,
    font=("Arial", 14, "bold"),
    padx=20,
    pady=10
)
predict_button.pack(pady=25)

result_title = tk.Label(
    root,
    text="Результат:",
    font=("Arial", 14, "bold")
)
result_title.pack()

result_label = tk.Label(
    root,
    text="Введите данные и нажмите кнопку",
    font=("Arial", 16, "bold"),
    justify="center"
)
result_label.pack(pady=15)

info = tk.Label(
    root,
    text="Модель: Random Forest\n"
         "Данные: Titanic train.csv",
    font=("Arial", 10)
)
info.pack(side="bottom", pady=15)

root.mainloop()
