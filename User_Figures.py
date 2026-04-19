import os

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

def fgn_sim(n=1000, H=0.7):
    mean = 0
    std = 1

    z = np.random.normal(size=2 * n)
    zr = z[:n]
    zi = z[n:]
    zic = -zi
    zi[0] = 0
    zr[0] = zr[0] * np.sqrt(2)
    zi[n - 1] = 0
    zr[n - 1] = zr[n - 1] * np.sqrt(2)
    zr = np.concatenate([zr[:n], zr[n - 2::-1]])
    zi = np.concatenate([zi[:n], zic[n - 2::-1]])
    z = zr + 1j * zi

    k = np.arange(n)
    gammak = (
        np.abs(k - 1) ** (2 * H)
        - 2 * np.abs(k) ** (2 * H)
        + np.abs(k + 1) ** (2 * H)
    ) / 2
    ind = np.concatenate([np.arange(n - 1), [n - 1], np.arange(n - 2, 0, -1)])
    gammak = gammak[ind]
    gkFGN0 = np.fft.ifft(gammak)
    gksqrt = np.real(gkFGN0)

    if np.all(gksqrt > 0):
        gksqrt = np.sqrt(gksqrt)
        z = z[:len(gksqrt)] * gksqrt
        z = np.fft.ifft(z)
        z = 0.5 * (n - 1) ** (-0.5) * z
        z = np.real(z[:n])
    else:
        raise ValueError("Re(gk)-vector not positive")

    return std * z + mean


def make_plot(x, y, filename, show_scatter=True, show_line=True):
    fig, ax = plt.subplots(figsize=(8, 4))

    if show_line:
        ax.plot(x, y, color='red', alpha=0.5, linewidth=2)

    if show_scatter:
        ax.scatter(x, y, s=50, c='red')

    ax.axis('off')
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    # plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)


# =========================
# DATA
# =========================
n_points = 30
x = np.arange(n_points)

y_random = np.random.uniform(0, 10, n_points)
y_pink = fgn_sim(n=n_points, H=0.99)
y_sine = np.sin(np.linspace(0, 8 * np.pi, n_points))


# =========================
# RANDOM
# =========================
# scatter + line
# make_plot(x, y_random, "random_scatter_line.png", show_scatter=True, show_line=True)
#
# # scatter ONLY
# make_plot(x, y_random, "random_scatter.png", show_scatter=True, show_line=False)
#
# # line ONLY
# make_plot(x, y_random, "random_scatter_line.png", show_scatter=False, show_line=True)
#
# # =========================
# # PINK
# # =========================
# # scatter + line
# make_plot(x, y_pink, "pink_scatter_line.png", show_scatter=True, show_line=True)
#
# # line only
# make_plot(x, y_pink, "pink_line.png", show_scatter=False, show_line=True)


# # =========================
# # SINE
# # =========================
# # scatter + line
# make_plot(x, y_sine, "sine_scatter_line.png", show_scatter=True, show_line=True)
#
# # line only
# make_plot(x, y_sine, "sine_line.png", show_scatter=False, show_line=True)





# =========================================================
# SETTINGS
# =========================================================
# directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Presentations\2026 Human Movement Variability and Great Plains Biomechanics Conferences 2026\Figures'
# os.chdir(directory)
# image_path = "Sine blocks.png"   # change if needed
#
# img = Image.open(image_path).convert("RGBA")
# img_np = np.array(img)
#
# # Separate channels
# R = img_np[:, :, 0]
# G = img_np[:, :, 1]
# B = img_np[:, :, 2]
# A = img_np[:, :, 3]
#
# # =========================================================
# # DETECT RED PIXELS
# # =========================================================
# red_mask = (R > 200) & (G < 80) & (B < 80) & (A > 0)
#
# # =========================================================
# # FIND BAR TOPS
# # =========================================================
# height, width = red_mask.shape
#
# bar_tops = []
# for x in range(width):
#     column = red_mask[:, x]
#     if np.any(column):
#         y_top = np.argmax(column)   # first red pixel from top
#         bar_tops.append((x, y_top))
#
# x_pts = np.array([p[0] for p in bar_tops])
# y_pts = np.array([p[1] for p in bar_tops])
#
# # =========================================================
# # GROUP COLUMNS INTO BARS
# # =========================================================
# groups = []
# current_group = [0]
#
# for i in range(1, len(x_pts)):
#     if x_pts[i] - x_pts[i - 1] < 5:
#         current_group.append(i)
#     else:
#         groups.append(current_group)
#         current_group = [i]
#
# groups.append(current_group)
#
# # One point per bar: center x, highest point y
# x_bar = []
# y_bar = []
#
# for g in groups:
#     x_bar.append(np.mean(x_pts[g]))
#     y_bar.append(np.min(y_pts[g]))
#
# x_bar = np.array(x_bar)
# y_bar = np.array(y_bar)
#
# # =========================================================
# # SMOOTH CURVE THROUGH BAR TOPS
# # =========================================================
# x_dense = np.linspace(x_bar.min(), x_bar.max(), 30)
# spline = make_interp_spline(x_bar, y_bar, k=3)
# y_dense = spline(x_dense)
#
# # =========================================================
# # PLOT
# # Press "t" to toggle image on/off
# # =========================================================
# fig, ax = plt.subplots(figsize=(8, 5))
#
# # Background image artist
# img_artist = ax.imshow(img_np)
#
# # Red line + many red scatter points
# line_artist, = ax.plot(x_dense, y_dense, color='red', linewidth=2, alpha=0.5)
# scatter_artist = ax.scatter(x_dense, y_dense, c='red', s=50)
#
# # Keep same coordinate system as image
# ax.set_xlim(0, img_np.shape[1])
# ax.set_ylim(img_np.shape[0], 0)
#
# # Remove axes
# ax.axis('off')
#
# # Optional transparent figure background
# fig.patch.set_alpha(0)
# ax.set_facecolor('none')
#
# # State variable
# show_background = True
#
# def on_key(event):
#     global show_background
#
#     if event.key == 't':
#         show_background = not show_background
#         img_artist.set_visible(show_background)
#         fig.canvas.draw_idle()
#
# fig.canvas.mpl_connect('key_press_event', on_key)
#
# print("Press 't' to hide/show the image background.")
# plt.show()


def make_pink_line_only(x, y, filename=None):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, y, color='#FF4FA3', linewidth=10)

    ax.axis('off')
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    if filename is not None:
        plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)

    plt.show()
    plt.close(fig)


n_points_new = 60
x_new = np.arange(n_points_new)

# white noise
y_white = np.random.uniform(0, 10, n_points_new)

# pink noise
y_pink = fgn_sim(n=n_points_new, H=0.99)

# sine wave (12 cycles)
y_sine = np.sin(np.linspace(0, 24 * np.pi, 200))
x_sine = np.arange(200)

make_pink_line_only(x_new, y_white)
make_pink_line_only(x_new, y_pink)
make_pink_line_only(x_sine, y_sine)