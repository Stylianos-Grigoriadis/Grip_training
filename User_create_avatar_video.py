import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
import os
import Lib_grip as lb

# =========================================================
# INTERPOLATION
# =========================================================
def signal_interpolation(signal, step, plot=False):

    signal = np.array(signal)
    x_original = np.linspace(0, len(signal), len(signal))
    y_original = signal
    total_data_points = step * len(signal)
    x_new = np.linspace(x_original[0], x_original[-1], total_data_points)
    y_new = np.interp(x_new, x_original, y_original)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.scatter(x_original, y_original, label='Original', s=100)
        plt.scatter(x_new, y_new, label='Upsampled', s=1)
        plt.legend()
        plt.show()

    return y_new


# =========================================================
# PINK NOISE
# =========================================================
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


# =========================================================
# MAIN FUNCTION
# =========================================================
def plot_movable_pink_line_with_bg_and_avatar(
    y_data,
    bg_path,
    avatar_path,
    avatar_y_data=None,   # 👈 NEW
    bg_alpha=1.0,
    color='#FF4FA3',
    linewidth=4,          # 👈 already there, now emphasized
    figsize=(10, 5),
    avatar_x=0,
    avatar_scale=0.18,
    avatar_width=None,
    avatar_height=None,
    avatar_rescale_with_zoom=True,
    autoplay_speed=1,
    autoplay_interval=40
):

    y_data = np.asarray(y_data, dtype=float)
    x_data = np.arange(len(y_data), dtype=float)

    # 👇 Avatar data handling
    if avatar_y_data is None:
        avatar_y_data = y_data
    else:
        avatar_y_data = np.asarray(avatar_y_data, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.22)

    bg_img = np.array(Image.open(bg_path).convert("RGBA"))
    avatar_img = np.array(Image.open(avatar_path).convert("RGBA"))

    x_margin = max(1, int(0.05 * len(x_data)))
    y_range = np.max(y_data) - np.min(y_data)
    y_margin = 0.15 * y_range if y_range > 0 else 1

    xlim_initial = (np.min(x_data) - x_margin, np.max(x_data) + x_margin)
    ylim_initial = (np.min(y_data) - y_margin, np.max(y_data) + y_margin)

    ax.set_xlim(*xlim_initial)
    ax.set_ylim(*ylim_initial)

    bg_artist = ax.imshow(
        bg_img,
        extent=[*ax.get_xlim(), *ax.get_ylim()],
        aspect='auto',
        alpha=bg_alpha,
        zorder=0
    )

    # 👇 LINE uses y_data
    line_artist, = ax.plot(x_data, y_data, color=color, linewidth=linewidth, zorder=1)

    current_shift = 0
    is_playing = False

    # 👇 Avatar uses avatar_y_data
    def get_avatar_y(shift):
        sample_x = avatar_x - shift
        return np.interp(sample_x, x_data, avatar_y_data, left=avatar_y_data[0], right=avatar_y_data[-1])

    def get_avatar_size():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]

        if avatar_rescale_with_zoom:
            width = avatar_width if avatar_width is not None else avatar_scale * x_span
            height = avatar_height if avatar_height is not None else avatar_scale * y_span
        else:
            x_span0 = xlim_initial[1] - xlim_initial[0]
            y_span0 = ylim_initial[1] - ylim_initial[0]
            width = avatar_width if avatar_width is not None else avatar_scale * x_span0
            height = avatar_height if avatar_height is not None else avatar_scale * y_span0

        return width, height

    def update_avatar_extent():
        avatar_y = get_avatar_y(current_shift)
        width, height = get_avatar_size()

        avatar_artist.set_extent([
            avatar_x - width / 2,
            avatar_x + width / 2,
            avatar_y - height / 2,
            avatar_y + height / 2
        ])

    initial_avatar_y = get_avatar_y(current_shift)
    initial_width, initial_height = get_avatar_size()

    avatar_artist = ax.imshow(
        avatar_img,
        extent=[
            avatar_x - initial_width / 2,
            avatar_x + initial_width / 2,
            initial_avatar_y - initial_height / 2,
            initial_avatar_y + initial_height / 2
        ],
        aspect='auto',
        zorder=2
    )

    slider_ax = plt.axes([0.15, 0.08, 0.58, 0.05])
    shift_slider = Slider(
        ax=slider_ax,
        label='X shift',
        valmin=-len(x_data),
        valmax=len(x_data),
        valinit=0,
        valstep=1
    )

    def update_slider(val):
        nonlocal current_shift
        current_shift = shift_slider.val

        line_artist.set_xdata(x_data + current_shift)
        update_avatar_extent()

        fig.canvas.draw_idle()

    shift_slider.on_changed(update_slider)

    button_ax = plt.axes([0.77, 0.08, 0.12, 0.05])
    play_button = Button(button_ax, 'Play')

    timer = fig.canvas.new_timer(interval=autoplay_interval)

    def timer_update():
        if not is_playing:
            return

        new_shift = shift_slider.val - autoplay_speed
        if new_shift < shift_slider.valmin:
            new_shift = shift_slider.valmax

        shift_slider.set_val(new_shift)

    timer.add_callback(timer_update)

    def toggle_play(event):
        nonlocal is_playing
        is_playing = not is_playing

        if is_playing:
            play_button.label.set_text('Stop')
            timer.start()
        else:
            play_button.label.set_text('Play')
            timer.stop()

        fig.canvas.draw_idle()

    play_button.on_clicked(toggle_play)

    def update_on_zoom_pan(event_ax):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        bg_artist.set_extent([*xlim, *ylim])
        update_avatar_extent()

        fig.canvas.draw_idle()

    ax.callbacks.connect('xlim_changed', update_on_zoom_pan)
    ax.callbacks.connect('ylim_changed', update_on_zoom_pan)

    plt.show()

# =========================================================
# DATA
# =========================================================
n_points = 50
sine_points = 250
average = 20
sd = 5
plus_sd = 2
plus_average = 0
autoplay_speed=1
autoplay_interval=50


y_random = np.random.uniform(0, 10, n_points)
y_pink = fgn_sim(n=n_points, H=0.99)
y_sine = np.sin(np.linspace(0, 24 * np.pi, sine_points))

y_random = signal_interpolation(y_random, 10)
y_pink = signal_interpolation(y_pink, 10)
y_sine = signal_interpolation(y_sine, 2)

y_random = lb.z_transform(y_random, sd, average)
y_pink = lb.z_transform(y_pink, sd, average)
y_sine = lb.z_transform(y_sine, sd, average)

plus_random = lb.z_transform(np.random.uniform(0, 10, n_points), plus_sd, plus_average)
plus_random = signal_interpolation(plus_random, 10)
avatar_random = y_random + plus_random

plus_pink = lb.z_transform(np.random.uniform(0, 10, n_points), plus_sd, plus_average)
plus_pink = signal_interpolation(plus_pink, 10)
avatar_pink = y_pink + plus_pink

plus_sine = lb.z_transform(np.random.uniform(0, 10, n_points), plus_sd, plus_average)
plus_sine = signal_interpolation(plus_sine, 10)
avatar_sine = y_sine + plus_sine

step_points = 500
y_step = np.concatenate([np.full(step_points // 2, 30), np.full(step_points - step_points // 2, 50)])
plus_step = lb.z_transform(np.random.uniform(0, 10, n_points), plus_sd, plus_average)
plus_step = signal_interpolation(plus_step, int(np.ceil(len(y_step) / n_points)))
plus_step = plus_step[:len(y_step)]
avatar_step = y_step + plus_step

# =========================================================
# PATHS
# =========================================================
background_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Presentations\2026 Human Movement Variability and Great Plains Biomechanics Conferences 2026\Figures'
os.chdir(background_directory)

background_path = "background.jpg"
avatar_path = "avatar.png"


# =========================================================
# RUN
# =========================================================
plot_movable_pink_line_with_bg_and_avatar(y_step, background_path, avatar_path, avatar_y_data=avatar_step, bg_alpha=0.8, avatar_x=0, avatar_scale=0.18, avatar_rescale_with_zoom=True, figsize=(8, 7), autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval)
plot_movable_pink_line_with_bg_and_avatar(y_random, background_path, avatar_path,avatar_y_data=avatar_random, bg_alpha=0.8, avatar_x=0, avatar_scale=0.18, avatar_rescale_with_zoom=True, figsize=(8, 7), autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval)
plot_movable_pink_line_with_bg_and_avatar(y_sine, background_path, avatar_path,avatar_y_data=avatar_sine, bg_alpha=0.8, avatar_x=0, avatar_scale=0.18, avatar_rescale_with_zoom=True, figsize=(8, 7), autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval)
plot_movable_pink_line_with_bg_and_avatar(y_pink, background_path, avatar_path,avatar_y_data=avatar_pink, bg_alpha=0.8, avatar_x=0, avatar_scale=0.18, avatar_rescale_with_zoom=True, figsize=(8, 7), autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval)

