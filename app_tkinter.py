import math
import random
import time
import tkinter as tk
from tkinter import ttk

import joblib
import pandas as pd

MODEL_PATH = "titanic_model.pkl"
DATA_PATH = "train.csv"
MAX_TRIES = 3

BG, PANEL, PANEL2 = "#0b0f17", "#141b27", "#1d2736"
TEXT, MUTED = "#eaf0f8", "#8795a8"
GREEN, RED, GOLD = "#1fe38a", "#ff4560", "#ffc83d"

# ───────────────────────── Модель и данные ─────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    raise SystemExit(
        f"Не найден файл: {e.filename}\nСначала запустите: python train_model.py"
    )

# Цена билета для пассажира в Titanic Drop — медиана по классу из train.csv
FARE = {c: float(data[data.Pclass == c].Fare.median()) for c in (1, 2, 3)}


def predict(c, sex, age, sibsp, parch, fare):
    df = pd.DataFrame([{
        "Pclass": c, "Sex": sex, "Age": age,
        "SibSp": sibsp, "Parch": parch, "Fare": fare,
    }])
    return float(model.predict_proba(df)[0][1])


# ───────────────────────── Логика Titanic Drop ─────────────────────────
def new_passenger():
    return {"c": 3, "sex": "male", "age": 20,
            "t": {"c": 0, "s": 0, "a": 0}}


def passenger_prob(st):
    return predict(st["c"], st["sex"], st["age"], 0, 0, FARE[st["c"]])


def build_upgrades(st):
    """Список апгрейдов. ok/bad меняют пассажира при успехе/неудаче."""
    ups = []

    def set_(key, val):
        def f(s):
            s[key] = val
        return f

    def older(s):
        s["age"] = min(80, s["age"] + 8)

    if st["c"] == 3:
        ups.append(dict(k="c", name="Класс 3 → 2", ch=0.60, ok=set_("c", 2),
                        bad=lambda s: None, bad_text="класс остаётся 3"))
    elif st["c"] == 2:
        ups.append(dict(k="c", name="Класс 2 → 1", ch=0.45, ok=set_("c", 1),
                        bad=set_("c", 3), bad_text="класс падает до 3"))
    else:
        ups.append(dict(k="c", name="Класс 1 (максимум)", off="максимум"))

    if st["sex"] == "male":
        ups.append(dict(k="s", name="Пол → женский", ch=0.15,
                        ok=set_("sex", "female"), bad=lambda s: None,
                        bad_text="пол остаётся мужским"))
    else:
        ups.append(dict(k="s", name="Пол: женский", off="максимум"))

    if st["age"] > 12:
        ups.append(dict(k="a", name="Возраст → 12 лет", ch=0.50,
                        ok=set_("age", 12), bad=older, bad_text="возраст +8 лет"))
    elif st["age"] > 4:
        ups.append(dict(k="a", name="Возраст → 4 года", ch=0.30,
                        ok=set_("age", 4), bad=older, bad_text="возраст +8 лет"))
    else:
        ups.append(dict(k="a", name="Возраст 4 года", off="максимум"))

    for u in ups:
        if "off" not in u and st["t"][u["k"]] >= MAX_TRIES:
            u["off"] = "попыток нет"
        u["left"] = MAX_TRIES - st["t"][u["k"]]
    return ups


# ───────────────────────── Интерфейс ─────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        root.title("TITANIC DROP")
        root.geometry("1040x760")
        root.minsize(980, 720)
        root.configure(bg=BG)

        self.st = new_passenger()
        self.sel = None
        self.busy = False
        self.angle = 0
        self.log = []

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2,
                        foreground=TEXT, arrowcolor=TEXT, bordercolor=PANEL2)

        self.build_header()
        self.page1 = tk.Frame(root, bg=BG)
        self.page2 = tk.Frame(root, bg=BG)
        self.build_page1()
        self.build_page2()
        self.show(1)
        self.calc()
        self.render()

    # ───── общее ─────
    def build_header(self):
        h = tk.Frame(self.root, bg=PANEL, height=64)
        h.pack(fill="x")
        h.pack_propagate(False)
        tk.Label(h, text="TITANIC", font=("Arial", 20, "bold"),
                 bg=PANEL, fg=GOLD).pack(side="left", padx=(20, 6))
        tk.Label(h, text="DROP", font=("Arial", 20, "bold"),
                 bg=PANEL, fg=GREEN).pack(side="left")
        self.nav = {}
        for n, t in [(2, "Titanic Drop"), (1, "Прогнозирование")]:
            b = tk.Button(h, text=t, font=("Arial", 11, "bold"), bg=PANEL2,
                          fg=TEXT, relief="flat", padx=16, pady=8,
                          cursor="hand2", highlightthickness=2,
                          command=lambda n=n: self.show(n))
            b.pack(side="right", padx=(0, 12))
            self.nav[n] = b

    def show(self, n):
        self.page1.pack_forget()
        self.page2.pack_forget()
        (self.page1 if n == 1 else self.page2).pack(fill="both", expand=True)
        for k, b in self.nav.items():
            b.config(highlightbackground=GOLD if k == n else PANEL2,
                     highlightcolor=GOLD)

    def card(self, parent, title):
        f = tk.Frame(parent, bg=PANEL)
        tk.Label(f, text=title, font=("Arial", 11, "bold"), bg=PANEL,
                 fg=MUTED).pack(anchor="w", padx=16, pady=(14, 8))
        return f

    # ───── страница 1: прогнозирование ─────
    def build_page1(self):
        p = self.page1
        left = self.card(p, "ПАССАЖИР")
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        form = tk.Frame(left, bg=PANEL)
        form.pack(padx=16, pady=(0, 16))

        def lbl(r, t):
            tk.Label(form, text=t, bg=PANEL, fg=TEXT,
                     font=("Arial", 11)).grid(row=r, column=0, sticky="w", pady=8)

        self.v_class = tk.StringVar(value="3")
        self.v_sex = tk.StringVar(value="male")
        lbl(0, "Класс билета")
        c1 = ttk.Combobox(form, textvariable=self.v_class, values=["1", "2", "3"],
                          state="readonly", width=14)
        c1.grid(row=0, column=1, padx=(14, 0))
        lbl(1, "Пол")
        c2 = ttk.Combobox(form, textvariable=self.v_sex,
                          values=["male", "female"], state="readonly", width=14)
        c2.grid(row=1, column=1, padx=(14, 0))
        for c in (c1, c2):
            c.bind("<<ComboboxSelected>>", lambda e: self.calc())

        self.entries = {}
        for r, (key, t, d) in enumerate([
            ("age", "Возраст", "20"),
            ("fare", "Цена билета (£)", "8"),
            ("sib", "Братья/сёстры/супруги", "0"),
            ("par", "Родители/дети", "0"),
        ], start=2):
            lbl(r, t)
            e = tk.Entry(form, width=17, bg=PANEL2, fg=TEXT,
                         insertbackground=TEXT, relief="flat",
                         font=("Arial", 11))
            e.insert(0, d)
            e.grid(row=r, column=1, padx=(14, 0), ipady=3)
            e.bind("<KeyRelease>", lambda ev: self.calc())
            self.entries[key] = e

        right = self.card(p, "ШАНС ВЫЖИТЬ")
        right.pack(side="left", fill="both", expand=True, padx=(8, 16), pady=16)
        self.fp = tk.Label(right, text="—", font=("Arial", 64, "bold"),
                           bg=PANEL, fg=TEXT)
        self.fp.pack(expand=True, pady=(40, 0))
        self.fv = tk.Label(right, text="", font=("Arial", 18, "bold"),
                           bg=PANEL, fg=TEXT)
        self.fv.pack(expand=True, pady=(0, 40))

    def calc(self):
        try:
            age = float(self.entries["age"].get())
            fare = float(self.entries["fare"].get())
            sib = int(self.entries["sib"].get())
            par = int(self.entries["par"].get())
            if min(age, fare, sib, par) < 0:
                raise ValueError
        except ValueError:
            self.fp.config(text="—", fg=TEXT)
            self.fv.config(text="Проверьте введённые данные", fg=TEXT)
            return
        p = predict(int(self.v_class.get()), self.v_sex.get(),
                    age, sib, par, fare)
        col = GREEN if p >= 0.5 else RED
        self.fp.config(text=f"{p * 100:.0f}%", fg=col)
        self.fv.config(text="Скорее ВЫЖИВЕТ" if p >= 0.5
                       else "Скорее НЕ ВЫЖИВЕТ", fg=col)

    # ───── страница 2: Titanic Drop ─────
    def build_page2(self):
        p = self.page2
        tk.Label(p, text="TITANIC DROP", font=("Arial", 30, "bold"),
                 bg=BG, fg=GOLD).pack(pady=(12, 4))
        body = tk.Frame(p, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        # Пассажир
        left = self.card(body, "ТВОЙ ПАССАЖИР")
        left.pack(side="left", fill="y", padx=(0, 8))
        tk.Label(left, text="🧍", font=("Arial", 50), bg=PANEL,
                 fg=TEXT).pack(padx=40)
        self.stat = {}
        for key, t in [("c", "Класс билета"), ("sex", "Пол"), ("age", "Возраст"),
                       ("sib", "Братья/сёстры/супруги"), ("par", "Родители/дети")]:
            row = tk.Frame(left, bg=PANEL)
            row.pack(fill="x", padx=16, pady=5)
            tk.Label(row, text=t, bg=PANEL, fg=MUTED,
                     font=("Arial", 10)).pack(side="left")
            v = tk.Label(row, text="", bg=PANEL, fg=TEXT,
                         font=("Arial", 11, "bold"))
            v.pack(side="right")
            self.stat[key] = v
        self.pr = tk.Label(left, text="", font=("Arial", 40, "bold"), bg=PANEL)
        self.pr.pack(pady=(14, 0))
        tk.Label(left, text="шанс выжить", bg=PANEL, fg=MUTED,
                 font=("Arial", 10)).pack(pady=(0, 16))

        # Улучшения
        right = self.card(body, "УЛУЧШИ ПАССАЖИРА")
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        cards = tk.Frame(right, bg=PANEL)
        cards.pack(fill="x", padx=16)
        self.up_btns = []
        for i in range(3):
            b = tk.Button(cards, text="", justify="left", anchor="w",
                          font=("Arial", 10), bg=PANEL2, fg=TEXT, relief="flat",
                          padx=10, pady=10, cursor="hand2",
                          highlightthickness=2, highlightbackground=PANEL2,
                          disabledforeground=MUTED, wraplength=170)
            b.pack(side="left", fill="both", expand=True, padx=4)
            self.up_btns.append(b)

        mid = tk.Frame(right, bg=PANEL)
        mid.pack(pady=8)
        self.chance_lbl = self.side_box(mid, "Шанс успеха")
        self.chance_lbl[0].pack(side="left", padx=14)
        self.cv = tk.Canvas(mid, width=250, height=250, bg=PANEL,
                            highlightthickness=0)
        self.cv.pack(side="left")
        self.go = tk.Button(self.cv, text="UPGRADE", font=("Arial", 12, "bold"),
                            bg=GOLD, fg="#1a1200", activebackground="#e6b52f",
                            relief="flat", cursor="hand2", command=self.upgrade)
        self.cv.create_window(125, 125, window=self.go, width=100, height=100)
        self.gain_lbl = self.side_box(mid, "Если успех")
        self.gain_lbl[0].pack(side="left", padx=14)

        self.msg = tk.Label(right, text="", font=("Arial", 13, "bold"),
                            bg=PANEL, fg=TEXT, wraplength=520)
        self.msg.pack(pady=(4, 6))
        btns = tk.Frame(right, bg=PANEL)
        btns.pack()
        tk.Button(btns, text="🚢  Отправить в рейс", font=("Arial", 12, "bold"),
                  bg=GREEN, fg="#06281a", relief="flat", padx=16, pady=8,
                  cursor="hand2", command=self.voyage).pack(side="left", padx=6)
        tk.Button(btns, text="Новый пассажир", font=("Arial", 11), bg=PANEL2,
                  fg=TEXT, relief="flat", padx=14, pady=8, cursor="hand2",
                  command=self.reset).pack(side="left", padx=6)
        self.log_lbl = tk.Label(right, text="", bg=PANEL, fg=MUTED,
                                font=("Arial", 10), justify="left")
        self.log_lbl.pack(anchor="w", padx=16, pady=(8, 2))
        tk.Label(right, text="По 3 попытки на класс, пол и возраст. Неудача "
                 "ухудшает пассажира.\nЕсли пол мужской и смена не удалась, "
                 "он остаётся мужским.", bg=PANEL, fg=MUTED,
                 font=("Arial", 9), justify="left").pack(anchor="w", padx=16,
                                                         pady=(0, 12))

    def side_box(self, parent, title):
        f = tk.Frame(parent, bg=PANEL2, padx=14, pady=10)
        tk.Label(f, text=title, bg=PANEL2, fg=MUTED,
                 font=("Arial", 9)).pack()
        v = tk.Label(f, text="—", bg=PANEL2, fg=TEXT, font=("Arial", 16, "bold"))
        v.pack()
        return f, v

    def render(self):
        st = self.st
        self.stat["c"].config(text=str(st["c"]))
        self.stat["sex"].config(text="Женский" if st["sex"] == "female" else "Мужской")
        self.stat["age"].config(text=str(st["age"]))
        self.stat["sib"].config(text="0")
        self.stat["par"].config(text="0")
        p = passenger_prob(st)
        self.pr.config(text=f"{p * 100:.0f}%", fg=GREEN if p >= 0.5 else RED)

        ups = build_upgrades(st)
        if self.sel is None or not any(u["k"] == self.sel and "off" not in u
                                       for u in ups):
            self.sel = next((u["k"] for u in ups if "off" not in u), None)

        for b, u in zip(self.up_btns, ups):
            if "off" in u:
                b.config(text=f"{u['name']}\n{u['off']}", state="disabled",
                         highlightbackground=PANEL2, command=lambda: None)
            else:
                b.config(text=f"{u['name']}\nшанс {u['ch'] * 100:.0f}% · "
                              f"попыток {u['left']}\nпровал: {u['bad_text']}",
                         state="normal",
                         highlightbackground=GOLD if u["k"] == self.sel else PANEL2,
                         command=lambda k=u["k"]: self.select(k))

        cur = next((u for u in ups if u["k"] == self.sel), None)
        ch = cur["ch"] if cur else 0
        self.draw_wheel(ch)
        if cur:
            self.chance_lbl[1].config(text=f"{ch * 100:.0f}%")
            n = dict(st)
            cur["ok"](n)
            q = passenger_prob(n)
            d = (q - p) * 100
            self.gain_lbl[1].config(
                text=f"{q * 100:.0f}% ({d:+.0f})", fg=GREEN if d >= 0 else RED)
            self.go.config(text="UPGRADE", state="disabled" if self.busy else "normal")
        else:
            self.chance_lbl[1].config(text="—")
            self.gain_lbl[1].config(text="—", fg=TEXT)
            self.go.config(text="НЕТ\nПОПЫТОК", state="disabled")
        self.log_lbl.config(text="\n".join(self.log[:5]))

    def select(self, k):
        if not self.busy:
            self.sel = k
            self.render()

    # ───── колесо ─────
    def draw_wheel(self, ch):
        c = self.cv
        c.delete("ring")
        box = (25, 25, 225, 225)
        c.create_oval(*box, outline=RED, width=18, tags="ring")
        if ch > 0:
            c.create_arc(*box, start=90, extent=-ch * 360, style="arc",
                         outline=GREEN, width=18, tags="ring")
        self.draw_marker(self.angle)

    def draw_marker(self, ang):
        self.angle = ang
        c = self.cv
        c.delete("marker")
        a = math.radians(ang)
        x, y = 125 + 100 * math.sin(a), 125 - 100 * math.cos(a)
        c.create_oval(x - 10, y - 10, x + 10, y + 10, fill="white",
                      outline=BG, width=2, tags="marker")

    def upgrade(self):
        if self.busy:
            return
        ups = build_upgrades(self.st)
        cur = next((u for u in ups if u["k"] == self.sel and "off" not in u), None)
        if not cur:
            return
        self.busy = True
        self.go.config(state="disabled")
        self.msg.config(text="", fg=TEXT)
        r = random.random()
        ok = r < cur["ch"]
        start_angle = self.angle % 360
        target = 360 * 5 + r * 360 - start_angle
        t0, dur = time.time(), 3.6

        def step():
            t = min(1.0, (time.time() - t0) / dur)
            e = 1 - (1 - t) ** 3
            self.draw_marker(start_angle + target * e)
            if t < 1:
                self.root.after(16, step)
            else:
                self.finish(cur, ok)
        step()

    def finish(self, cur, ok):
        self.st["t"][cur["k"]] += 1
        if ok:
            cur["ok"](self.st)
            text, col = f"Успех! {cur['name']}", GREEN
        else:
            cur["bad"](self.st)
            text, col = f"Неудача: {cur['bad_text']}", RED
        self.log.insert(0, f"{'✅' if ok else '❌'} {cur['name']}")
        self.busy = False
        if not any("off" not in u for u in build_upgrades(self.st)):
            text += "\nПопытки закончились — отправляй в рейс!"
        self.render()
        self.msg.config(text=text, fg=col)

    def voyage(self):
        if self.busy:
            return
        p = passenger_prob(self.st)
        ok = random.random() < p
        self.log.insert(0, f" {'выжил' if ok else 'погиб'} ({p * 100:.0f}%)")
        self.st = new_passenger()
        self.sel = None
        self.render()
        self.msg.config(
            text=f"Пассажир {'ВЫЖИЛ' if ok else 'НЕ ВЫЖИЛ'} "
                 f"(шанс был {p * 100:.0f}%)",
            fg=GREEN if ok else RED)

    def reset(self):
        if self.busy:
            return
        self.st = new_passenger()
        self.sel = None
        self.render()
        self.msg.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
