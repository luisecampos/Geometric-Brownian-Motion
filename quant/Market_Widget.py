"""
Market Watch Widget
--------------------
A tiny always-on-top desktop widget showing live prices for:
DOW, NASDAQ, S&P 500, 10-Year Treasury Yield, BTC, and GBP/USD, EUR/USD.

Extras:
- Auto-refreshes on a timer and plots a rolling sparkline per row (fills in
  as the widget stays open -- no extra network calls beyond the normal poll).
- Right-click a row to set a price-cross alert (blank input clears it). When
  the price crosses the threshold in either direction, the row flashes and
  the system beeps.
"""

import tkinter as tk
from tkinter import font as tkfont
from tkinter import simpledialog
import threading
from collections import deque
import yfinance as yf
from datetime import datetime

try:
    import winsound
    def _beep():
        winsound.Beep(1000, 150)
except ImportError:  # non-Windows fallback
    def _beep():
        pass

# ---- Configuration ----
TICKERS = {
    "DOW":     "^DJI",
    "NASDAQ":  "^IXIC",
    "S&P 500": "^GSPC",
    "10Y":     "^TNX",     # yfinance reports this *10, we divide by 10 below
    "BTC":     "BTC-USD",
    "GBP/USD": "GBPUSD=X",
    "EUR/USD": "EURUSD=X",
}

FX_PAIRS = {"GBP/USD", "EUR/USD"}

HISTORY_LEN = 30            # sparkline points kept per ticker
REFRESH_INTERVAL_MS = 60_000  # auto-refresh cadence

BG_COLOR = "#111318"
FG_COLOR = "#E8E8E8"
UP_COLOR = "#3ddc84"
DOWN_COLOR = "#ff5c5c"
ACCENT = "#4d9fff"
ALERT_COLOR = "#ffd23d"
SPARK_NEUTRAL = "#666666"


class MarketWidget:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)       # no title bar -> looks like a sticker
        self.root.attributes("-topmost", True)  # always on top
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("300x330+80+80")     # width x height + x_offset + y_offset

        self.label_font = tkfont.Font(family="Segoe UI", size=10)
        self.price_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.title_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        self.history = {name: deque(maxlen=HISTORY_LEN) for name in TICKERS}
        self.alerts = {}        # name -> threshold price
        self.last_values = {}   # name -> most recent numeric value (for crossing checks)

        self._build_ui()
        self._make_draggable()
        self.refresh()  # initial load
        self._schedule_auto_refresh()

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
        self.sparklines = {}
        body = tk.Frame(self.root, bg=BG_COLOR)
        body.pack(fill="both", expand=True, padx=10)

        for name in TICKERS:
            row = tk.Frame(body, bg=BG_COLOR)
            row.pack(fill="x", pady=4)

            name_lbl = tk.Label(row, text=name, bg=BG_COLOR, fg=FG_COLOR,
                                  font=self.label_font, width=8, anchor="w")
            name_lbl.pack(side="left")

            price_lbl = tk.Label(row, text="--", bg=BG_COLOR, fg=FG_COLOR,
                                   font=self.price_font, anchor="e")
            price_lbl.pack(side="right")

            spark = tk.Canvas(row, width=46, height=16, bg=BG_COLOR,
                                highlightthickness=0)
            spark.pack(side="right", padx=6)

            # Right-click anywhere on the row to set/clear a price-cross alert
            for widget in (row, name_lbl, price_lbl, spark):
                widget.bind("<Button-3>", lambda e, n=name: self._prompt_alert(n))
            for widget in (name_lbl, price_lbl, spark):
                widget.config(cursor="hand2")

            self.rows[name] = price_lbl
            self.sparklines[name] = spark

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

    def _schedule_auto_refresh(self):
        self.root.after(REFRESH_INTERVAL_MS, self._auto_refresh)

    def _auto_refresh(self):
        if not self.root.winfo_exists():
            return  # widget was closed; stop the after() chain
        self.refresh()
        self._schedule_auto_refresh()

    def _prompt_alert(self, name):
        current = self.alerts.get(name)
        prompt = f"Alert price for {name} (blank to clear):"
        if current is not None:
            prompt = f"Alert price for {name}, currently {current:g} (blank to clear):"
        answer = simpledialog.askstring("Set Alert", prompt, parent=self.root)
        if answer is None:
            return  # dialog cancelled, leave alert untouched
        answer = answer.strip()
        if not answer:
            self.alerts.pop(name, None)
            return
        try:
            self.alerts[name] = float(answer)
        except ValueError:
            pass  # ignore unparseable input rather than crash the widget

    def _check_alert(self, name, value):
        threshold = self.alerts.get(name)
        prev = self.last_values.get(name)
        self.last_values[name] = value
        if threshold is None or prev is None:
            return
        crossed = (prev < threshold <= value) or (prev > threshold >= value)
        if crossed:
            self._flash_row(name)
            _beep()

    def _flash_row(self, name, _n=0):
        if not self.root.winfo_exists():
            return  # widget was closed; stop the after() chain
        lbl = self.rows[name]
        color = ALERT_COLOR if _n % 2 == 0 else BG_COLOR
        lbl.config(bg=color)
        if _n < 5:
            self.root.after(150, lambda: self._flash_row(name, _n + 1))
        else:
            lbl.config(bg=BG_COLOR)

    def _draw_sparkline(self, name):
        canvas = self.sparklines[name]
        canvas.delete("all")
        points = list(self.history[name])
        if len(points) < 2:
            return

        w, h = 46, 16
        lo, hi = min(points), max(points)
        span = hi - lo or 1  # avoid divide-by-zero on a flat line
        step = w / (len(points) - 1)

        coords = []
        for i, p in enumerate(points):
            x = i * step
            y = h - ((p - lo) / span) * h
            coords.extend([x, y])

        color = UP_COLOR if points[-1] >= points[0] else DOWN_COLOR
        if points[-1] == points[0]:
            color = SPARK_NEUTRAL
        canvas.create_line(*coords, fill=color, width=1.5, smooth=True)

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

            self.history[name].append(value)
            self._draw_sparkline(name)
            self._check_alert(name, value)

            if name == "10Y":
                text = f"{value:.2f}%"
            elif name == "BTC":
                text = f"${value:,.0f}"
            elif name in FX_PAIRS:
                text = f"{value:,.4f}"
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