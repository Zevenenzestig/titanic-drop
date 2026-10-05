"""TITANIC DROP — десктопное окно с тем же интерфейсом, что и на сайте.

Запуск:  python app.py
Нужен пакет pywebview (pip install -r requirements.txt).
Если его нет, страница откроется в браузере.
"""
import os
import webbrowser
from pathlib import Path

BASE = Path(__file__).resolve().parent
HTML = BASE / "titanic_drop.html"


def main():
    if not HTML.exists():
        raise SystemExit(f"Не найден файл: {HTML.name}")
    try:
        import webview
    except ImportError:
        print("pywebview не установлен (pip install pywebview) — "
              "открываю в браузере.")
        webbrowser.open(HTML.as_uri())
        return

    webview.create_window(
        "TITANIC DROP", HTML.as_uri(),
        width=1200, height=820, min_size=(900, 650),
    )
    # Сохраняем состояние игры между запусками
    webview.start(private_mode=False,
                  storage_path=str(BASE / ".webview_data"))


if __name__ == "__main__":
    main()
