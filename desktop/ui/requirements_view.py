"""Pantalla BLOQUEADA, aviso de requisitos e "Información del sistema" (HU-DS-20, E-02, E-12).

Solo dibujan un ``requirements_check.Report``; la lógica y los textos están en ``desktop/requirements_check.py``.
"""

import tkinter as tk
from tkinter import messagebox

from desktop import requirements_check as rc
from desktop.messages import message_for

BG, FG, DIM = "#181818", "#f0f0f0", "#a0a0a0"
LEVEL_COLORS = {"ok": "#4caf50", "warn": "#ffb300", "fail": "#ff5252"}
LEVEL_MARKS = {"ok": "✓", "warn": "!", "fail": "✗"}


def _table(parent: tk.Misc, report: rc.Report) -> tk.Frame:
    frame = tk.Frame(parent, bg=BG)
    for row, item in enumerate(report.checks):
        color = LEVEL_COLORS[item.level]
        tk.Label(frame, text=LEVEL_MARKS[item.level], bg=BG, fg=color, font=("Segoe UI", 11, "bold")).grid(
            row=row, column=0, sticky="w", padx=(0, 8)
        )
        tk.Label(frame, text=item.label, bg=BG, fg=FG, anchor="w").grid(row=row, column=1, sticky="w", padx=(0, 12))
        tk.Label(frame, text=item.value, bg=BG, fg=color, anchor="w").grid(row=row, column=2, sticky="w")
        if item.message:
            tk.Label(frame, text=item.message, bg=BG, fg=DIM, anchor="w", wraplength=520, justify="left").grid(
                row=row, column=3, sticky="w", padx=(12, 0)
            )
    versions = " · ".join(f"{k} {v}" for k, v in report.info.versions.items())
    tk.Label(frame, text=versions, bg=BG, fg=DIM, anchor="w").grid(
        row=len(report.checks), column=0, columnspan=4, sticky="w", pady=(10, 0)
    )
    return frame


def show_system_info(parent: tk.Misc, report: rc.Report) -> tk.Toplevel:
    """Ayuda → Información del sistema (y el botón de la pantalla BLOQUEADA)."""
    win = tk.Toplevel(parent)
    win.title("Información del sistema")
    win.configure(bg=BG, padx=20, pady=16)
    _table(win, report).pack(anchor="w")
    buttons = tk.Frame(win, bg=BG)
    buttons.pack(anchor="e", pady=(14, 0))

    def copy():
        win.clipboard_clear()
        win.clipboard_append(rc.system_info_text(report))

    tk.Button(buttons, text="Copiar", command=copy, padx=10).pack(side="left", padx=4)
    tk.Button(buttons, text="Cerrar", command=win.destroy, padx=10).pack(side="left", padx=4)
    win.bind("<Escape>", lambda e: win.destroy())
    return win


class BlockedWindow(tk.Tk):
    """BLOQUEADA (CA-2): la app no carga los modelos; solo se puede ver la información o salir."""

    def __init__(self, report: rc.Report):
        super().__init__()
        self.report = report
        self.title("WWM Clasificador — equipo no compatible")
        self.configure(bg=BG, padx=24, pady=20)
        self.resizable(False, False)
        tk.Label(
            self, text="Tu equipo no es compatible", bg=BG, fg=LEVEL_COLORS["fail"], font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")
        tk.Label(
            self,
            text=message_for("REQUIREMENTS_NOT_MET", motivo=report.reason),
            bg=BG,
            fg=FG,
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(8, 14))
        buttons = tk.Frame(self, bg=BG)
        buttons.pack(anchor="e")
        tk.Button(buttons, text="Información del sistema", command=self.show_info, padx=10).pack(side="left", padx=4)
        tk.Button(buttons, text="Salir", command=self.destroy, padx=10).pack(side="left", padx=4)
        self.bind("<Escape>", lambda e: self.destroy())

    def show_info(self) -> tk.Toplevel:
        return show_system_info(self, self.report)


def show_blocked(report: rc.Report) -> None:
    BlockedWindow(report).mainloop()


def show_warning_once(report: rc.Report, warned_version: str, version: str) -> bool:
    """CA-3: el aviso no bloqueante, una vez por versión. Devuelve ``True`` si se mostró."""
    if not report.warnings or warned_version == version:
        return False
    root = tk.Tk()
    root.withdraw()
    try:
        messagebox.showwarning("Requisitos del equipo", rc.warning_text(report), parent=root)
    finally:
        root.destroy()
    return True
