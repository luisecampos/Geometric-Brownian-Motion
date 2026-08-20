"""
Market Watch Widget
--------------------
A tiny always-on-top desktop widget showing live prices for:
DOW, NASDAQ, S&P 500, 10-Year Treasury Yield, and BTC.
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import yfinance as yf
from datetime import datetime

# ---- Configuration ----
TICKERS = {
    "DOW":    "^DJI",
    "NASDAQ": "^IXIC",
    "S&P 500": "^GSPC",
    "10Y":    "^TNX",     # yfinance reports this *10, we divide by 10 below
    "BTC":    "BTC-USD",
}

BG_COLOR = "#111318"
FG_COLOR = "#E8E8E8"
UP_COLOR = "#3ddc84"
DOWN_COLOR = "#ff5c5c"
ACCENT = "#4d9fff"


class MarketWidget:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)       # no title bar -> looks like a sticker
        self.root.attributes("-topmost", True)  # always on top
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("240x260+80+80")     # width x height + x_offset + y_offset

        self.label_font = tkfont.Font(family="Segoe UI", size=10)
        self.price_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.title_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        self._build_ui()
        self._make_draggable()
        self.refresh()  # initial load

    def _build_ui(self):
        # Title bar (also drag handle) with close button
        top = tk.Frame(self.root, bg=BG_COLOR)
        top.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(top, text="Market Watch", bg=BG_COLOR, fg=ACCENT,
                  font=self.title_font).pack(side="left")

        close_btn = tk.Label(top, text="✕", bg=BG_COLOR, fg=FG_COLOR,
                               font=self.label_font, cursor="hand2")
        close_btn.pack(side="right")
        close_btn.bind("<Button-1>", lambda e: self.root.destroy())

        # Rows for each ticker
        self.rows = {}
        body = tk.Frame(self.root, bg=BG_COLOR)
        body.pack(fill="both", expand=True, padx=10)

        for name in TICKERS:
            row = tk.Frame(body, bg=BG_COLOR)
            row.pack(fill="x", pady=4)

            tk.Label(row, text=name, bg=BG_COLOR, fg=FG_COLOR,
                      font=self.label_font, width=8, anchor="w").pack(side="left")

            price_lbl = tk.Label(row, text="--", bg=BG_COLOR, fg=FG_COLOR,
                                   font=self.price_font, anchor="e")
            price_lbl.pack(side="right")

            self.rows[name] = price_lbl

        # Footer: last updated + refresh button
        footer = tk.Frame(self.root, bg=BG_COLOR)
        footer.pack(fill="x", padx=10, pady=(6, 8))

        self.updated_lbl = tk.Label(footer, text="", bg=BG_COLOR, fg="#888888",
                                      font=("Segoe UI", 8))
        self.updated_lbl.pack(side="left")

        self.refresh_btn = tk.Label(footer, text="↻ Refresh", bg=BG_COLOR, fg=ACCENT,
                                      font=self.label_font, cursor="hand2")
        self.refresh_btn.pack(side="right")
        self.refresh_btn.bind("<Button-1>", lambda e: self.refresh())

    def _make_draggable(self):
        self._drag_data = {"x": 0, "y": 0}

        def start_drag(event):
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y

        def do_drag(event):
            x = self.root.winfo_x() + (event.x - self._drag_data["x"])
            y = self.root.winfo_y() + (event.y - self._drag_data["y"])
            self.root.geometry(f"+{x}+{y}")

        self.root.bind("<ButtonPress-1>", start_drag)
        self.root.bind("<B1-Motion>", do_drag)

    def refresh(self):
        self.refresh_btn.config(text="…")
        threading.Thread(target=self._fetch_and_update, daemon=True).start()

    def _fetch_and_update(self):
        results = {}
        for name, symbol in TICKERS.items():
            try:
                t = yf.Ticker(symbol)
                hist = t.history(period="2d")
                if hist.empty:
                    results[name] = ("N/A", 0)
                    continue
                last_close = hist["Close"].iloc[-1]
                prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else last_close

                if name == "10Y":
                    change = (last_close - prev_close) * 100  # bps
                else:
                    change = ((last_close - prev_close) / prev_close) * 100 if prev_close else 0  # %

                results[name] = (last_close, change)
            except Exception:
                results[name] = ("Err", 0)

        self.root.after(0, self._update_ui, results)

    def _update_ui(self, results):
        for name, (value, pct) in results.items():
            lbl = self.rows[name]
            if value in ("N/A", "Err"):
                lbl.config(text=value, fg=FG_COLOR)
                continue

            if name == "10Y":
                text = f"{value:.2f}%"
            elif name == "BTC":
                text = f"${value:,.0f}"
            else:
                text = f"{value:,.2f}"

            suffix = "bps" if name == "10Y" else "%"
            decimals = 0 if name == "10Y" else 2

            arrow = "▲" if pct >= 0 else "▼"
            color = UP_COLOR if pct >= 0 else DOWN_COLOR
            lbl.config(text=f"{text}  {arrow}{abs(pct):.{decimals}f}{suffix}", fg=color)

        self.updated_lbl.config(text=f"Updated {datetime.now().strftime('%I:%M %p')}")
        self.refresh_btn.config(text="↻ Refresh")
if __name__ == "__main__":
    root = tk.Tk()
    app = MarketWidget(root)
    root.mainloop()