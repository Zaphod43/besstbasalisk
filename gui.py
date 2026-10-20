"""
gui.py
======
Tkinter-Oberfläche (deutsch) für den Batteriespeicher-Profit-Optimierer.

Führt standardmäßig die REALISTISCHE Day-Ahead-Simulation aus
(optimizer.simulate_day_ahead): pro Kalendertag wird ein LP gelöst, das nur
die an diesem Tag bekannten Preise verwendet - genau wie ein echter
Day-Ahead-Händler. Optional wird zusätzlich optimizer.optimize() über den
gesamten Zeitraum berechnet, als THEORETISCHE OBERGRENZE (perfekte
Voraussicht) - nur als Vergleichswert, keine erreichbare Strategie.

Die eigentliche Mathematik steckt in optimizer.py; diese Datei kümmert sich
nur um Eingaben, Dateiauswahl und Diagramme.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from optimizer import load_timeseries_csv, BatteryParams, optimize, simulate_day_ahead


def to_float(text: str) -> float:
    """Zahl einlesen; Dezimalkomma und Dezimalpunkt werden akzeptiert."""
    return float(text.strip().replace(",", "."))


def de(x: float, decimals: int = 2) -> str:
    """Zahl im deutschen Format ausgeben: 1.234,56"""
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


class BessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batteriespeicher-Profit-Optimierer - Day-Ahead-Händlersimulation")
        self.geometry("1200x840")
        self.minsize(1000, 700)

        self.price_path = tk.StringVar()
        self.gridlimit_path = tk.StringVar()
        self.date_from = tk.StringVar()
        self.date_to = tk.StringVar()
        self.status_var = tk.StringVar(value="Bereit.")

        self.last_dayahead = None
        self.last_clairvoyant = None

        self._build_layout()

    # ------------------------------------------------------------------
    def _build_layout(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer, padding=(0, 0, 10, 0))
        left.pack(side="left", fill="y")

        right = ttk.Frame(outer)
        right.pack(side="left", fill="both", expand=True)

        self._build_input_panel(left)
        self._build_results_panel(right)

    def _build_input_panel(self, parent):
        row = 0

        def section(label):
            nonlocal row
            ttk.Label(parent, text=label, font=("Segoe UI", 10, "bold")).grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(10, 2))
            row += 1

        def field(label, default, width=10):
            nonlocal row
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(default))
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=row, column=1, sticky="w", padx=(6, 0))
            row += 1
            return var

        section("Datendateien")
        ttk.Label(parent, text="Preis-CSV *").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.price_path, width=22).grid(row=row, column=1, sticky="w", padx=(6, 0))
        ttk.Button(parent, text="Durchsuchen", command=self._pick_price_file).grid(row=row, column=2, padx=(4, 0))
        row += 1

        ttk.Label(parent, text="Verfügbare Leistung\n(CSV, optional)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.gridlimit_path, width=22).grid(row=row, column=1, sticky="w", padx=(6, 0))
        ttk.Button(parent, text="Durchsuchen", command=self._pick_gridlimit_file).grid(row=row, column=2, padx=(4, 0))
        row += 1

        section("Zeitraum (leer = gesamte Datei)")
        ttk.Label(parent, text="Von (TT.MM.JJJJ)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.date_from, width=14).grid(row=row, column=1, sticky="w", padx=(6, 0))
        row += 1
        ttk.Label(parent, text="Bis (TT.MM.JJJJ)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.date_to, width=14).grid(row=row, column=1, sticky="w", padx=(6, 0))
        row += 1

        section("Batterieparameter")
        self.f_capacity = field("Kapazität (kWh)", 1000)
        self.f_dod = field("Entladetiefe DoD (0-1)", "0,9")
        self.f_maxc = field("Max. Ladeleistung (kW)", 250)
        self.f_maxd = field("Max. Entladeleistung (kW)", 250)
        self.f_effc = field("Ladewirkungsgrad (0-1)", "0,95")
        self.f_effd = field("Entladewirkungsgrad (0-1)", "0,95")
        self.f_ramp = field("Rampe (%/min, z. B. 10 = 10 %)", 10)
        self.f_initsoc = field("Start-Ladezustand (kWh,\nleer = Minimum)", "")

        section("Kosten")
        self.f_fee = field("Gebühr pro geladener kWh", "0,50")
        self.f_annual = field("Fixe Jahreskosten", 5000)
        self.f_throughput = field("Max. Jahresdurchsatz kWh\n(leer = keine Grenze)", "")

        section("Vergleichswert")
        self.show_upper_bound = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent, text="Zusätzlich Obergrenze bei perfekter\nVoraussicht berechnen (zum Vergleich)",
            variable=self.show_upper_bound
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        row += 1
        ttk.Button(parent, text="Day-Ahead-Simulation starten", command=self._run).grid(
            row=row, column=0, columnspan=3, sticky="we", pady=(15, 4))
        row += 1
        ttk.Button(parent, text="Fahrplan als CSV exportieren", command=self._export_schedule).grid(
            row=row, column=0, columnspan=3, sticky="we", pady=(0, 2))
        row += 1
        ttk.Button(parent, text="Tagesübersicht als CSV exportieren", command=self._export_daily).grid(
            row=row, column=0, columnspan=3, sticky="we", pady=(0, 4))
        row += 1
        ttk.Label(parent, textvariable=self.status_var, wraplength=250, foreground="#444").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _build_results_panel(self, parent):
        summary_frame = ttk.LabelFrame(parent, text="Ergebnisübersicht", padding=8)
        summary_frame.pack(fill="x", pady=(0, 8))

        self.summary_text = tk.Text(summary_frame, height=14, width=100, state="disabled",
                                     font=("Consolas", 10))
        self.summary_text.pack(fill="x")

        chart_frame = ttk.LabelFrame(parent, text="Realisierter Day-Ahead-Fahrplan", padding=4)
        chart_frame.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(7.5, 5.0), dpi=100)
        self.ax_price = self.fig.add_subplot(211)
        self.ax_power = self.fig.add_subplot(212, sharex=self.ax_price)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    def _pick_price_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV-Dateien", "*.csv")])
        if path:
            self.price_path.set(path)

    def _pick_gridlimit_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV-Dateien", "*.csv")])
        if path:
            self.gridlimit_path.set(path)

    def _set_status(self, text):
        self.status_var.set(text)
        self.update_idletasks()

    # ------------------------------------------------------------------
    def _collect_params(self) -> BatteryParams:
        throughput = self.f_throughput.get().strip()
        init_soc = self.f_initsoc.get().strip()
        return BatteryParams(
            capacity_kwh=to_float(self.f_capacity.get()),
            dod_pct=to_float(self.f_dod.get()),
            max_charge_kw=to_float(self.f_maxc.get()),
            max_discharge_kw=to_float(self.f_maxd.get()),
            charge_eff=to_float(self.f_effc.get()),
            discharge_eff=to_float(self.f_effd.get()),
            ramp_pct_per_min=to_float(self.f_ramp.get()) / 100.0,
            fee_per_kwh_charged=to_float(self.f_fee.get()),
            fixed_annual_cost=to_float(self.f_annual.get()),
            max_annual_throughput_kwh=to_float(throughput) if throughput else None,
            initial_soc_kwh=to_float(init_soc) if init_soc else None,
        )

    def _load_and_filter_prices(self):
        prices = load_timeseries_csv(self.price_path.get())
        if self.date_from.get().strip():
            start = pd.to_datetime(self.date_from.get().strip(), dayfirst=True)
            prices = prices[prices["timestamp"] >= start]
        if self.date_to.get().strip():
            end = pd.to_datetime(self.date_to.get().strip(), dayfirst=True) + pd.Timedelta(days=1)
            prices = prices[prices["timestamp"] < end]
        return prices.reset_index(drop=True)

    def _run(self):
        try:
            if not self.price_path.get():
                messagebox.showerror("Datei fehlt", "Bitte eine Preis-CSV-Datei auswählen.")
                return

            self._set_status("Daten werden geladen ...")
            prices = self._load_and_filter_prices()
            if len(prices) < 2:
                messagebox.showerror("Keine Daten", "Der gewählte Zeitraum enthält weniger als 2 Preispunkte.")
                return

            grid_limit = None
            if self.gridlimit_path.get():
                grid_limit = load_timeseries_csv(self.gridlimit_path.get())

            params = self._collect_params()

            self._set_status(f"Day-Ahead-Simulation läuft ({len(prices)} Zeitschritte, "
                              f"ein LP pro Tag) ...")
            dayahead = simulate_day_ahead(prices, params, grid_limit=grid_limit)

            clairvoyant = None
            if self.show_upper_bound.get():
                self._set_status("Obergrenze bei perfekter Voraussicht wird berechnet ...")
                clairvoyant = optimize(prices, params, grid_limit=grid_limit)

            self.last_dayahead = dayahead
            self.last_clairvoyant = clairvoyant
            self._show_summary(dayahead, clairvoyant, params)
            self._plot(dayahead)
            self._set_status(f"Fertig. {dayahead.n_days} Handelstag(e) simuliert.")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Fehler", f"{type(e).__name__}: {e}")
            self._set_status("Fehler - Details in der Konsole.")

    def _show_summary(self, dayahead, clairvoyant, params: BatteryParams):
        cur = params.price_currency
        lines = []
        lines.append("REALISTISCHE DAY-AHEAD-SIMULATION")
        lines.append("(pro Tag ein LP, nur mit den an diesem Tag bekannten Preisen -")
        lines.append(" kein Blick in zukünftige Tage; dies ist das erreichbare Ergebnis)")
        lines.append(f"  Simulierte Tage:              {dayahead.n_days}")
        lines.append(f"  Gesamtgewinn:                 {de(dayahead.total_profit)} {cur}")
        lines.append(f"    Erlöse (Verkauf/Entladung): {de(dayahead.total_revenue)} {cur}")
        lines.append(f"    Ladekosten (Energie):      -{de(dayahead.total_charge_cost)} {cur}")
        lines.append(f"    Ladegebühren:              -{de(dayahead.total_fees)} {cur}")
        lines.append(f"    Fixkosten (anteilig):      -{de(dayahead.fixed_cost_allocated)} {cur}")
        lines.append(f"  Geladene Energie:             {de(dayahead.total_energy_charged_kwh, 1)} kWh")
        lines.append(f"  Entladene Energie:            {de(dayahead.total_energy_discharged_kwh, 1)} kWh")
        if dayahead.messages:
            lines.append(f"  Warnungen: {len(dayahead.messages)} Tag(e) übersprungen - siehe Konsole.")
            for m in dayahead.messages[:5]:
                print("WARNUNG:", m)

        if clairvoyant is not None and clairvoyant.status == "OPTIMAL":
            lines.append("")
            lines.append("OBERGRENZE BEI PERFEKTER VORAUSSICHT (nur zum Vergleich, nicht erreichbar)")
            lines.append("(ein einziges LP über den GESAMTEN Zeitraum, als wären alle künftigen")
            lines.append(" Preise heute schon bekannt - theoretische Grenze, keine echte Strategie)")
            lines.append(f"  Gesamtgewinn:                 {de(clairvoyant.total_profit)} {cur}")
            gap = clairvoyant.total_profit - dayahead.total_profit
            pct = (gap / clairvoyant.total_profit * 100) if clairvoyant.total_profit else 0
            lines.append(f"  Abstand zur Day-Ahead-Simulation: {de(gap)} {cur}  ({de(pct, 1)} % höher)")

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state="disabled")

    def _plot(self, dayahead):
        sched = dayahead.schedule
        self.ax_price.clear()
        self.ax_power.clear()

        self.ax_price.plot(sched["timestamp"], sched["price"], color="tab:blue", linewidth=0.9)
        self.ax_price.set_ylabel("Preis / kWh")
        self.ax_price.set_title("Marktpreis (Day-Ahead, tageweise bekannt)")

        if hasattr(self, "ax_soc") and self.ax_soc is not None:
            try:
                self.ax_soc.remove()
            except Exception:
                pass
        self.ax_soc = self.ax_price.twinx()
        self.ax_soc.plot(sched["timestamp"], sched["soc_kwh"], color="tab:green", linewidth=0.9, alpha=0.7)
        self.ax_soc.set_ylabel("Ladezustand (kWh)", color="tab:green")

        self.ax_power.fill_between(sched["timestamp"], 0, sched["charge_kw"],
                                    color="tab:red", alpha=0.6, label="Laden (kW)", step="mid")
        self.ax_power.fill_between(sched["timestamp"], 0, -sched["discharge_kw"],
                                    color="tab:orange", alpha=0.6, label="Entladen (kW)", step="mid")
        self.ax_power.set_ylabel("Leistung (kW)")
        self.ax_power.axhline(0, color="black", linewidth=0.5)
        self.ax_power.legend(loc="upper right", fontsize=8)
        self.ax_power.set_title("Realisierter Lade-/Entlade-Fahrplan")

        self.fig.autofmt_xdate()
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _export_schedule(self):
        if self.last_dayahead is None:
            messagebox.showinfo("Nichts zu exportieren", "Bitte zuerst eine Simulation starten.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV-Dateien", "*.csv")])
        if path:
            self.last_dayahead.schedule.to_csv(path, index=False, sep=";", decimal=",")
            self._set_status(f"Fahrplan exportiert nach {path}")

    def _export_daily(self):
        if self.last_dayahead is None:
            messagebox.showinfo("Nichts zu exportieren", "Bitte zuerst eine Simulation starten.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV-Dateien", "*.csv")])
        if path:
            self.last_dayahead.daily_breakdown.to_csv(path, index=False, sep=";", decimal=",")
            self._set_status(f"Tagesübersicht exportiert nach {path}")


if __name__ == "__main__":
    app = BessApp()
    app.mainloop()
