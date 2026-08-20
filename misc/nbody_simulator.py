"""
N-Body Gravitational Simulator
================================
Dependencies: pip install matplotlib numpy

Run: python nbody_simulator.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation
import math

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
G          = 500.0
SOFTENING  = 15.0
DT         = 0.016
MAX_TRAIL  = 180

# ─────────────────────────────────────────────
#  BODY CLASS
# ─────────────────────────────────────────────
class Body:
    def __init__(self, name, mass, x, y, vx, vy, color, body_type="planet"):
        self.name      = name
        self.mass      = float(mass)
        self.x         = float(x)
        self.y         = float(y)
        self.vx        = float(vx)
        self.vy        = float(vy)
        self.color     = color
        self.body_type = body_type
        self.ax        = 0.0
        self.ay        = 0.0
        self.trail_x   = []
        self.trail_y   = []
        base = {"star": 16, "planet": 7, "moon": 4, "asteroid": 2}.get(body_type, 6)
        self.radius = max(3, int(base * (mass ** 0.30)))

    def update_trail(self):
        self.trail_x.append(self.x)
        self.trail_y.append(self.y)
        if len(self.trail_x) > MAX_TRAIL:
            self.trail_x.pop(0)
            self.trail_y.pop(0)

    def kinetic_energy(self):
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)


# ─────────────────────────────────────────────
#  PHYSICS
# ─────────────────────────────────────────────
def compute_accelerations(bodies):
    for b in bodies:
        b.ax = 0.0
        b.ay = 0.0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            a, b = bodies[i], bodies[j]
            dx = b.x - a.x
            dy = b.y - a.y
            dist2 = dx*dx + dy*dy + SOFTENING**2
            dist  = math.sqrt(dist2)
            f     = G / dist2
            fx    = f * dx / dist
            fy    = f * dy / dist
            a.ax += b.mass * fx
            a.ay += b.mass * fy
            b.ax -= a.mass * fx
            b.ay -= a.mass * fy


def integrate(bodies, dt):
    for b in bodies:
        b.vx += b.ax * dt
        b.vy += b.ay * dt
        b.x  += b.vx * dt
        b.y  += b.vy * dt
        b.update_trail()


def check_collisions(bodies):
    to_remove = set()
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if i in to_remove or j in to_remove:
                continue
            a, b = bodies[i], bodies[j]
            dx   = a.x - b.x
            dy   = a.y - b.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < (a.radius + b.radius) * 0.18:
                total  = a.mass + b.mass
                b.vx   = (b.vx * b.mass + a.vx * a.mass) / total
                b.vy   = (b.vy * b.mass + a.vy * a.mass) / total
                b.x    = (b.x  * b.mass + a.x  * a.mass) / total
                b.y    = (b.y  * b.mass + a.y  * a.mass) / total
                b.mass = total
                b.radius = max(3, int(b.radius * (total ** 0.08)))
                to_remove.add(i)
    return [b for i, b in enumerate(bodies) if i not in to_remove]


# ─────────────────────────────────────────────
#  PRESETS
# ─────────────────────────────────────────────
def preset_solar_system():
    return [
        Body("Sun",      1000,   0,   0,   0,    0,   "#FFE050", "star"),
        Body("Mercury",     1, 120,   0,   0,   64,   "#B4A090", "planet"),
        Body("Venus",       3, 200,   0,   0,   50,   "#E6BE64", "planet"),
        Body("Earth",       4, 290,   0,   0,   43,   "#50A0FF", "planet"),
        Body("Moon",      0.1, 310,   0,   0,   51,   "#C8C8C8", "moon"),
        Body("Mars",        2, 400,   0,   0,   36,   "#DC6432", "planet"),
        Body("Jupiter",    80, 600,   0,   0,   26,   "#D2AA78", "planet"),
        Body("Saturn",     50, 800,   0,   0,   22,   "#E6C88C", "planet"),
    ]

def preset_binary_star():
    return [
        Body("Alpha",   500, -120,   0,   0,  18,  "#FFC850", "star"),
        Body("Beta",    500,  120,   0,   0, -18,  "#64B4FF", "star"),
        Body("Planet",    2,    0, 280, -22,   0,  "#50DC8C", "planet"),
        Body("Planet2",   1,    0,-260,  20,   0,  "#FF8250", "planet"),
    ]

def preset_figure_eight():
    v = 93.24
    return [
        Body("A", 100,  97.0, -24.3,  v*0.306893,  v*0.125507, "#FF7850", "star"),
        Body("B", 100, -97.0,  24.3,  v*0.306893,  v*0.125507, "#50C8FF", "star"),
        Body("C", 100,   0.0,   0.0, -v*0.613786, -v*0.251014, "#78FF8C", "star"),
    ]

def preset_three_body_chaos():
    return [
        Body("Chaos-1", 200, -150,  80,  12,  -8, "#FF5078", "star"),
        Body("Chaos-2", 200,  160, -90,  -9,  15, "#50A0FF", "star"),
        Body("Chaos-3", 200,   10,  20,  -3,  -7, "#FFC832", "star"),
    ]

def preset_mini_galaxy():
    bodies = [Body("Core", 2000, 0, 0, 0, 0, "#FFF0B4", "star")]
    colors = ["#64B4FF","#50DC8C","#FFA050","#DC64FF","#FF6464",
              "#64FFD2","#FFE064","#B464FF","#64FFA0","#FF8C8C",
              "#8CC8FF","#C8FF8C"]
    for i in range(12):
        angle  = i * (2 * math.pi / 12) + i * 0.3
        radius = 150 + i * 28
        x  = math.cos(angle) * radius
        y  = math.sin(angle) * radius
        spd = math.sqrt(G * 2000 / radius) * 0.9
        vx = -math.sin(angle) * spd
        vy =  math.cos(angle) * spd
        bodies.append(Body(f"S{i+1}", 1+(i%3), x, y, vx, vy, colors[i%len(colors)], "planet"))
    return bodies

def preset_rogue_planet():
    bodies = preset_solar_system()
    bodies.append(Body("Rogue", 30, -900, 200, 55, -5, "#B450FF", "planet"))
    return bodies

PRESETS = {
    "Solar System":     preset_solar_system,
    "Binary Star":      preset_binary_star,
    "Figure Eight":     preset_figure_eight,
    "Three Body Chaos": preset_three_body_chaos,
    "Mini Galaxy":      preset_mini_galaxy,
    "Rogue Planet":     preset_rogue_planet,
}

# ─────────────────────────────────────────────
#  SIMULATOR STATE
# ─────────────────────────────────────────────
class Sim:
    def __init__(self):
        self.bodies     = preset_solar_system()
        self.paused     = False
        self.speed      = 1.0
        self.sim_time   = 0.0
        self.place_mode = False
        self.new_name   = "Planet"
        self.new_mass   = 5.0
        self.new_vx     = 0.0
        self.new_vy     = 20.0
        self.new_type   = "planet"
        self.color_pool = ["#64B4FF","#50DC8C","#FFA050","#DC64FF",
                           "#FF6464","#64FFD2","#FFE064","#FF8C8C"]
        self.color_idx  = 0

    def next_color(self):
        c = self.color_pool[self.color_idx % len(self.color_pool)]
        self.color_idx += 1
        return c

    def load_preset(self, name):
        self.bodies   = PRESETS[name]()
        self.sim_time = 0.0

    def step(self):
        if self.paused or len(self.bodies) < 1:
            return
        steps = max(1, int(self.speed))
        dt    = DT * (self.speed / steps)
        for _ in range(steps):
            if len(self.bodies) > 1:
                compute_accelerations(self.bodies)
            integrate(self.bodies, dt)
        if len(self.bodies) > 1:
            self.bodies = check_collisions(self.bodies)
        self.sim_time += DT * self.speed

    def total_ke(self):
        return sum(b.kinetic_energy() for b in self.bodies)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    sim = Sim()

    # ── Figure layout ──────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor="#030610")
    fig.canvas.manager.set_window_title("N-Body Gravitational Simulator")

    # Main sim axis
    ax_sim = fig.add_axes([0.0, 0.0, 0.72, 1.0], facecolor="#030610")
    ax_sim.set_xlim(-950, 950)
    ax_sim.set_ylim(-600, 600)
    ax_sim.set_aspect("equal")
    ax_sim.axis("off")

    # Stars background
    rng = np.random.default_rng(42)
    star_x = rng.uniform(-950, 950, 280)
    star_y = rng.uniform(-600, 600, 280)
    star_b = rng.uniform(0.15, 0.55, 280)
    ax_sim.scatter(star_x, star_y, s=0.4, c=[[v,v,v] for v in star_b], zorder=0)

    # Title
    ax_sim.text(-930, 565, "N-BODY GRAVITATIONAL SIMULATOR",
                color="#64C8FF", fontsize=11, fontweight="bold",
                fontfamily="monospace", va="top", zorder=10)
    ax_sim.text(-930, 540, "scroll=zoom  drag=pan  click=place body",
                color="#3C5070", fontsize=8, fontfamily="monospace", va="top", zorder=10)

    # HUD text
    hud_time  = ax_sim.text(-930, -555, "", color="#64C8FF",
                             fontsize=9, fontfamily="monospace", va="bottom", zorder=10)
    hud_ke    = ax_sim.text(-930, -575, "", color="#64C8FF",
                             fontsize=9, fontfamily="monospace", va="bottom", zorder=10)
    hud_n     = ax_sim.text(-930, -535, "", color="#64C8FF",
                             fontsize=9, fontfamily="monospace", va="bottom", zorder=10)
    hud_mode  = ax_sim.text(0, 0, "", color="#50FF90",
                             fontsize=11, fontfamily="monospace",
                             ha="center", va="center", zorder=20,
                             bbox=dict(boxstyle="round,pad=0.4", fc="#030610", ec="#50FF90", lw=1))

    # Draw objects (trails + bodies + labels)
    trail_lines  = {}
    body_dots    = {}
    body_labels  = {}
    glow_patches = {}

    def init_artists():
        for key in list(trail_lines.keys()):
            trail_lines[key].remove()
            body_dots[key].remove()
            body_labels[key].remove()
            if key in glow_patches:
                glow_patches[key].remove()
        trail_lines.clear()
        body_dots.clear()
        body_labels.clear()
        glow_patches.clear()

        for b in sim.bodies:
            tl, = ax_sim.plot([], [], color=b.color, lw=0.8, alpha=0.5, zorder=2)
            dot, = ax_sim.plot([], [], "o", color=b.color,
                               ms=b.radius * 0.85, zorder=5,
                               markeredgecolor="white", markeredgewidth=0.4)
            lbl = ax_sim.text(0, 0, b.name, color="#6080A0",
                              fontsize=7, fontfamily="monospace",
                              va="center", zorder=6)
            trail_lines[id(b)]  = tl
            body_dots[id(b)]    = dot
            body_labels[id(b)]  = lbl

            if b.body_type == "star":
                glow = plt.Circle((b.x, b.y), b.radius * 2.5,
                                  color=b.color, alpha=0.12, zorder=3)
                ax_sim.add_patch(glow)
                glow_patches[id(b)] = glow

    init_artists()

    # ── Right panel ────────────────────────────
    ax_panel = fig.add_axes([0.72, 0.0, 0.28, 1.0], facecolor="#080E20")

    def panel_text(x, y, s, color="#8090B0", size=8, bold=False):
        ax_panel.text(x, y, s, color=color, fontsize=size,
                      fontfamily="monospace", va="top",
                      fontweight="bold" if bold else "normal",
                      transform=ax_panel.transAxes)

    ax_panel.axis("off")
    panel_text(0.06, 0.98, "CONTROLS", "#64C8FF", 9, bold=True)
    panel_text(0.06, 0.94, "─────────────────────", "#1E2840")

    # ── Preset buttons ──────────────────────────
    preset_names = list(PRESETS.keys())
    preset_axes  = []
    preset_btns  = []
    for i, name in enumerate(preset_names):
        row = i // 2
        col = i % 2
        left   = 0.73 + col * 0.135
        bottom = 0.78 - row * 0.062
        pax = fig.add_axes([left, bottom, 0.125, 0.048])
        btn = widgets.Button(pax, name, color="#0A1428", hovercolor="#142040")
        btn.label.set_fontsize(7)
        btn.label.set_fontfamily("monospace")
        btn.label.set_color("#64C8FF")
        preset_axes.append(pax)
        preset_btns.append(btn)

        def make_cb(n):
            def cb(event):
                sim.load_preset(n)
                init_artists()
            return cb
        btn.on_clicked(make_cb(name))

    # ── Sliders ────────────────────────────────
    ax_speed = fig.add_axes([0.755, 0.585, 0.205, 0.025])
    sl_speed = widgets.Slider(ax_speed, "Speed", 0.1, 8.0, valinit=1.0,
                               color="#2050A0", track_color="#0A1428")
    sl_speed.label.set_color("#8090B0")
    sl_speed.label.set_fontsize(8)
    sl_speed.label.set_fontfamily("monospace")
    sl_speed.valtext.set_color("#64C8FF")
    sl_speed.valtext.set_fontfamily("monospace")

    ax_trail = fig.add_axes([0.755, 0.545, 0.205, 0.025])
    sl_trail = widgets.Slider(ax_trail, "Trails", 0, 400, valinit=180,
                               color="#2050A0", track_color="#0A1428")
    sl_trail.label.set_color("#8090B0")
    sl_trail.label.set_fontsize(8)
    sl_trail.label.set_fontfamily("monospace")
    sl_trail.valtext.set_color("#64C8FF")
    sl_trail.valtext.set_fontfamily("monospace")

    def update_speed(val):
        sim.speed = sl_speed.val
    def update_trail(val):
        global MAX_TRAIL
        MAX_TRAIL = int(sl_trail.val)
    sl_speed.on_changed(update_speed)
    sl_trail.on_changed(update_trail)

    # ── Pause / Clear buttons ──────────────────
    ax_pause = fig.add_axes([0.735, 0.490, 0.12, 0.042])
    ax_clear = fig.add_axes([0.865, 0.490, 0.10, 0.042])
    btn_pause = widgets.Button(ax_pause, "⏸  Pause", color="#0A1428", hovercolor="#142040")
    btn_clear = widgets.Button(ax_clear, "✕ Clear",  color="#0A1428", hovercolor="#200A0A")
    for btn in (btn_pause, btn_clear):
        btn.label.set_fontsize(8)
        btn.label.set_fontfamily("monospace")
    btn_pause.label.set_color("#64C8FF")
    btn_clear.label.set_color("#FF6464")

    def on_pause(event):
        sim.paused = not sim.paused
        btn_pause.label.set_text("▶  Play " if sim.paused else "⏸  Pause")
        fig.canvas.draw_idle()
    def on_clear(event):
        sim.bodies = []
        sim.sim_time = 0.0
        init_artists()
    btn_pause.on_clicked(on_pause)
    btn_clear.on_clicked(on_clear)

    # ── Add body inputs ────────────────────────
    panel_y_base = 0.425
    input_label_props = dict(color="#8090B0", fontsize=7.5,
                              fontfamily="monospace", transform=fig.transFigure)

    fig.text(0.735, panel_y_base + 0.005, "ADD BODY",
             color="#64C8FF", fontsize=8, fontweight="bold",
             fontfamily="monospace", transform=fig.transFigure)

    ax_iname = fig.add_axes([0.735, panel_y_base - 0.045, 0.235, 0.032])
    ax_imass = fig.add_axes([0.735, panel_y_base - 0.090, 0.110, 0.032])
    ax_itype = fig.add_axes([0.855, panel_y_base - 0.090, 0.110, 0.032])
    ax_ivx   = fig.add_axes([0.735, panel_y_base - 0.135, 0.110, 0.032])
    ax_ivy   = fig.add_axes([0.855, panel_y_base - 0.135, 0.110, 0.032])

    tb_name = widgets.TextBox(ax_iname, "Name  ", initial="Planet",
                               color="#080E20", hovercolor="#0D1830")
    tb_mass = widgets.TextBox(ax_imass, "Mass  ", initial="5",
                               color="#080E20", hovercolor="#0D1830")
    tb_type = widgets.TextBox(ax_itype, "Type  ", initial="planet",
                               color="#080E20", hovercolor="#0D1830")
    tb_vx   = widgets.TextBox(ax_ivx,   "Vx    ", initial="0",
                               color="#080E20", hovercolor="#0D1830")
    tb_vy   = widgets.TextBox(ax_ivy,   "Vy    ", initial="20",
                               color="#080E20", hovercolor="#0D1830")

    for tb in (tb_name, tb_mass, tb_type, tb_vx, tb_vy):
        tb.label.set_color("#607090")
        tb.label.set_fontsize(7.5)
        tb.label.set_fontfamily("monospace")
        tb.text_disp.set_color("#C8D8F0")
        tb.text_disp.set_fontfamily("monospace")

    ax_place = fig.add_axes([0.735, panel_y_base - 0.185, 0.235, 0.038])
    btn_place = widgets.Button(ax_place, "PLACE ON CANVAS  →  click sim",
                                color="#061428", hovercolor="#0A2040")
    btn_place.label.set_color("#50FF90")
    btn_place.label.set_fontsize(7.5)
    btn_place.label.set_fontfamily("monospace")

    def on_place(event):
        sim.place_mode = True
        hud_mode.set_text("CLICK on the simulation\nto place the body")
        hud_mode.set_visible(True)
        fig.canvas.draw_idle()
    btn_place.on_clicked(on_place)

    # ── Info panel text ────────────────────────
    info_texts = []
    info_y_start = 0.195
    info_lines = [
        ("HOTKEYS", "#64C8FF", 8,  True),
        ("space  →  pause / resume",    "#607090", 7.5, False),
        ("scroll →  zoom in / out",     "#607090", 7.5, False),
        ("drag   →  pan camera",        "#607090", 7.5, False),
        ("1–6    →  load preset",       "#607090", 7.5, False),
        ("",                            "#607090", 7.5, False),
        ("BODY TYPES",                  "#64C8FF", 8,   True),
        ("star  planet  moon  asteroid","#607090", 7.5, False),
    ]
    for i, (txt, col, sz, bold) in enumerate(info_lines):
        t = fig.text(0.738, info_y_start - i * 0.022, txt,
                     color=col, fontsize=sz, fontfamily="monospace",
                     fontweight="bold" if bold else "normal",
                     transform=fig.transFigure)
        info_texts.append(t)

    # Live stats
    stat_bodies = fig.text(0.738, 0.03, "", color="#64C8FF",
                            fontsize=8, fontfamily="monospace", transform=fig.transFigure)
    stat_time   = fig.text(0.738, 0.015, "", color="#64C8FF",
                            fontsize=8, fontfamily="monospace", transform=fig.transFigure)
    stat_ke     = fig.text(0.738, 0.00, "", color="#64C8FF",
                            fontsize=8, fontfamily="monospace", transform=fig.transFigure)

    # ── Camera state ───────────────────────────
    cam = {"x": 0.0, "y": 0.0, "zoom": 1.0,
           "dragging": False, "drag_start": None, "cam_start": None}

    def world_to_sim(mx, my):
        xmin, xmax = ax_sim.get_xlim()
        ymin, ymax = ax_sim.get_ylim()
        wx = xmin + mx * (xmax - xmin)
        wy = ymin + my * (ymax - ymin)
        return wx, wy

    def on_press(event):
        if event.inaxes != ax_sim:
            return
        if sim.place_mode:
            try:
                name  = tb_name.text.strip() or "Body"
                mass  = float(tb_mass.text)
                vx    = float(tb_vx.text)
                vy    = float(tb_vy.text)
                btype = tb_type.text.strip()
                if btype not in ("star","planet","moon","asteroid"):
                    btype = "planet"
                color = sim.next_color()
                sim.bodies.append(Body(name, mass, event.xdata, event.ydata, vx, vy, color, btype))
                init_artists()
            except Exception:
                pass
            sim.place_mode = False
            hud_mode.set_visible(False)
            fig.canvas.draw_idle()
        else:
            cam["dragging"]  = True
            cam["drag_start"]= (event.xdata, event.ydata)
            cam["cam_start"] = (ax_sim.get_xlim(), ax_sim.get_ylim())

    def on_release(event):
        cam["dragging"] = False

    def on_motion(event):
        if not cam["dragging"] or event.inaxes != ax_sim:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - cam["drag_start"][0]
        dy = event.ydata - cam["drag_start"][1]
        xl, yl = cam["cam_start"]
        ax_sim.set_xlim(xl[0] - dx, xl[1] - dx)
        ax_sim.set_ylim(yl[0] - dy, yl[1] - dy)
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes != ax_sim:
            return
        factor = 0.88 if event.button == "up" else 1.14
        xl = ax_sim.get_xlim()
        yl = ax_sim.get_ylim()
        cx = event.xdata if event.xdata else sum(xl)/2
        cy = event.ydata if event.ydata else sum(yl)/2
        ax_sim.set_xlim(cx + (xl[0]-cx)*factor, cx + (xl[1]-cx)*factor)
        ax_sim.set_ylim(cy + (yl[0]-cy)*factor, cy + (yl[1]-cy)*factor)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            sim.paused = not sim.paused
            btn_pause.label.set_text("▶  Play " if sim.paused else "⏸  Pause")
            fig.canvas.draw_idle()
        keys = {"1":"Solar System","2":"Binary Star","3":"Figure Eight",
                "4":"Three Body Chaos","5":"Mini Galaxy","6":"Rogue Planet"}
        if event.key in keys:
            sim.load_preset(keys[event.key])
            init_artists()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event",  on_motion)
    fig.canvas.mpl_connect("scroll_event",         on_scroll)
    fig.canvas.mpl_connect("key_press_event",      on_key)

    hud_mode.set_visible(False)

    # ── Animation loop ─────────────────────────
    def animate(frame):
        sim.step()

        # Sync new bodies that may have appeared
        for b in sim.bodies:
            bid = id(b)
            if bid not in trail_lines:
                tl, = ax_sim.plot([], [], color=b.color, lw=0.8, alpha=0.5, zorder=2)
                dot, = ax_sim.plot([], [], "o", color=b.color,
                                   ms=b.radius * 0.85, zorder=5,
                                   markeredgecolor="white", markeredgewidth=0.4)
                lbl  = ax_sim.text(0, 0, b.name, color="#6080A0",
                                   fontsize=7, fontfamily="monospace",
                                   va="center", zorder=6)
                trail_lines[bid] = tl
                body_dots[bid]   = dot
                body_labels[bid] = lbl

        living = {id(b) for b in sim.bodies}

        # Remove dead artists
        for bid in list(trail_lines.keys()):
            if bid not in living:
                trail_lines[bid].remove()
                body_dots[bid].remove()
                body_labels[bid].remove()
                if bid in glow_patches:
                    glow_patches[bid].remove()
                    del glow_patches[bid]
                del trail_lines[bid]
                del body_dots[bid]
                del body_labels[bid]

        # Update living bodies
        for b in sim.bodies:
            bid = id(b)
            n   = len(b.trail_x)
            if n > 1:
                alphas = np.linspace(0.05, 0.55, n)
                trail_lines[bid].set_data(b.trail_x, b.trail_y)
                trail_lines[bid].set_alpha(0.45)
            body_dots[bid].set_data([b.x], [b.y])
            body_labels[bid].set_position((b.x + b.radius * 1.8, b.y))
            if bid in glow_patches:
                glow_patches[bid].center = (b.x, b.y)

        # HUD
        hud_n.set_text(f"bodies: {len(sim.bodies)}")
        hud_time.set_text(f"time:   {sim.sim_time:,.1f} units")
        hud_ke.set_text(f"KE:     {sim.total_ke():,.0f}")

        stat_bodies.set_text(f"bodies: {len(sim.bodies)}")
        stat_time.set_text(f"time: {sim.sim_time:,.1f}")
        stat_ke.set_text(f"KE: {sim.total_ke():,.0f}")

        return list(trail_lines.values()) + list(body_dots.values()) + \
               list(body_labels.values()) + list(glow_patches.values()) + \
               [hud_n, hud_time, hud_ke, stat_bodies, stat_time, stat_ke]

    ani = FuncAnimation(fig, animate, interval=16, blit=False, cache_frame_data=False)

    plt.show()


if __name__ == "__main__":
    main()
