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
# VARIABLE ERROR ENVELOPE
# =========================================================
def make_variable_error_signal(
    length_of_signal,
    initial_sd_pre_perturbation,
    ending_sd_pre_perturbation,
    perturbation_start_index,
    perturbation_end_index,
    initial_sd_post_perturbation,
    ending_sd_post_perturbation,
    seed=None,
    smooth_noise=True,
    smoothing_kernel_size=11
):
    """
    Create a time-varying error signal whose SD changes linearly before and after perturbation.

    Parameters
    ----------
    length_of_signal : int
        Total number of points in the signal.

    initial_sd_pre_perturbation : float
        SD at the beginning of the pre-perturbation phase.

    ending_sd_pre_perturbation : float
        SD at the end of the pre-perturbation phase
        (just before perturbation_start_index).

    perturbation_start_index : int
        Index where the perturbation starts.

    perturbation_end_index : int
        Index where the perturbation ends.

    initial_sd_post_perturbation : float
        SD at the beginning of the post-perturbation phase
        (starting at perturbation_end_index).

    ending_sd_post_perturbation : float
        SD at the end of the post-perturbation phase.

    seed : int or None
        Random seed.

    smooth_noise : bool
        Whether to smooth the random noise slightly.

    smoothing_kernel_size : int
        Kernel size for smoothing the random noise.

    Returns
    -------
    error_signal : np.ndarray
        Random error signal with time-varying SD.
    """

    if seed is not None:
        np.random.seed(seed)

    length_of_signal = int(length_of_signal)
    perturbation_start_index = int(perturbation_start_index)
    perturbation_end_index = int(perturbation_end_index)

    if length_of_signal <= 0:
        raise ValueError("length_of_signal must be > 0")

    if not (0 <= perturbation_start_index <= perturbation_end_index <= length_of_signal):
        raise ValueError(
            "Indices must satisfy: 0 <= perturbation_start_index <= perturbation_end_index <= length_of_signal"
        )

    # -----------------------------------------------------
    # 1. Create base random noise
    # -----------------------------------------------------
    noise = np.random.normal(0, 1, length_of_signal)

    if smooth_noise:
        smoothing_kernel_size = int(smoothing_kernel_size)
        if smoothing_kernel_size < 1:
            smoothing_kernel_size = 1
        if smoothing_kernel_size % 2 == 0:
            smoothing_kernel_size += 1

        kernel = np.ones(smoothing_kernel_size) / smoothing_kernel_size
        noise = np.convolve(noise, kernel, mode='same')

    # Re-standardize to mean=0, std=1
    noise = noise - np.mean(noise)
    noise_std = np.std(noise)
    if noise_std > 0:
        noise = noise / noise_std

    # -----------------------------------------------------
    # 2. Create SD profile
    # -----------------------------------------------------
    sd_profile = np.zeros(length_of_signal, dtype=float)

    # Pre-perturbation phase: linear change
    if perturbation_start_index > 0:
        sd_profile[:perturbation_start_index] = np.linspace(
            initial_sd_pre_perturbation,
            ending_sd_pre_perturbation,
            perturbation_start_index
        )

    # Perturbation phase: keep constant at initial post-perturbation SD
    if perturbation_end_index > perturbation_start_index:
        sd_profile[perturbation_start_index:perturbation_end_index] = initial_sd_post_perturbation

    # Post-perturbation phase: linear change
    post_length = length_of_signal - perturbation_end_index
    if post_length > 0:
        sd_profile[perturbation_end_index:] = np.linspace(
            initial_sd_post_perturbation,
            ending_sd_post_perturbation,
            post_length
        )

    # -----------------------------------------------------
    # 3. Multiply noise by SD profile
    # -----------------------------------------------------
    error_signal = noise * sd_profile

    return error_signal


# =========================================================
# MAIN FUNCTION
# =========================================================
# =========================================================
# MAIN FUNCTION
# =========================================================
def plot_movable_pink_line_with_bg_and_avatar(
    y_data,
    bg_path,
    avatar_path,
    avatar_y_data=None,
    bg_alpha=1.0,
    color='#FF4FA3',
    linewidth=4,
    figsize=(10, 5),
    avatar_x=0,
    avatar_scale=0.18,
    avatar_width=None,
    avatar_height=None,
    avatar_rescale_with_zoom=True,
    autoplay_speed=1,
    autoplay_interval=40,
    xlim=None,
    ylim=None,
    show_ticks=False,
    show_background=True  # <-- NEW ARGUMENT
):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from PIL import Image

    y_data = np.asarray(y_data, dtype=float)
    x_data = np.arange(len(y_data), dtype=float)

    if avatar_y_data is None:
        avatar_y_data = y_data
    else:
        avatar_y_data = np.asarray(avatar_y_data, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.22)

    if show_background:
        bg_img = np.array(Image.open(bg_path).convert("RGBA"))

    avatar_img = np.array(Image.open(avatar_path).convert("RGBA"))

    x_margin = max(1, int(0.05 * len(x_data)))
    y_range = np.max(y_data) - np.min(y_data)
    y_margin = 0.15 * y_range if y_range > 0 else 1

    xlim_initial = (np.min(x_data) - x_margin, np.max(x_data) + x_margin)
    ylim_initial = (np.min(y_data) - y_margin, np.max(y_data) + y_margin)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(*xlim_initial)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(*ylim_initial)

    # ---- REMOVE TICKS IF REQUESTED ----
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)

    if show_background:
        bg_artist = ax.imshow(
            bg_img,
            extent=[*ax.get_xlim(), *ax.get_ylim()],
            aspect='auto',
            alpha=bg_alpha,
            zorder=0
        )
    else:
        bg_artist = None
        ax.set_facecolor("white")

    line_artist, = ax.plot(x_data, y_data, color=color, linewidth=linewidth, zorder=1)

    current_shift = 0
    is_playing = False

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

        if show_background:
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
average = 30
sd = 10
autoplay_speed = 1
autoplay_interval = 50
perturbation_points = 500
perturb_random = n_points // 2
preperturbation_duration = 250
perturbtion_duration = 20
postperturbation_duration = perturbation_points - preperturbation_duration - perturbtion_duration
interpolation_step = 10
error_sd = 4
error_average = 0

y_random = np.random.uniform(0, 10, n_points)
y_pink = fgn_sim(n=n_points, H=0.99)
y_sine = np.sin(np.linspace(0, 24 * np.pi, sine_points))
y_isometric_high = np.full(perturbation_points, 50)
y_isometric_low = np.full(perturbation_points, 10)

y_random = signal_interpolation(y_random, 10)
y_pink = signal_interpolation(y_pink, 10)
y_sine = signal_interpolation(y_sine, 2)

y_random = lb.z_transform(y_random, sd, average)
y_pink = lb.z_transform(y_pink, sd, average)
y_sine = lb.z_transform(y_sine, sd, average)

y_step_high = np.concatenate([np.full(perturbation_points // 2, 30), np.full(perturbation_points - perturbation_points // 2, 50)])
y_step_low = np.concatenate([np.full(perturbation_points // 2, 30), np.full(perturbation_points - perturbation_points // 2, 10)])

# Create the avatar's movement
perturbation_end_index = preperturbation_duration + perturbtion_duration
y_step_avatar_high = np.concatenate([np.full(preperturbation_duration, 30), np.linspace(30, 50, perturbtion_duration), np.full(postperturbation_duration, 50)])
y_step_avatar_low = np.concatenate([np.full(preperturbation_duration, 30), np.linspace(30, 10, perturbtion_duration), np.full(postperturbation_duration, 10)])


# Variable error signals
error_white = lb.signal_interpolation(lb.z_transform(np.random.uniform(0, 10, len(y_random)//interpolation_step), error_sd, error_average),interpolation_step)
error_pink = lb.signal_interpolation(lb.z_transform(np.random.uniform(0, 10, len(y_pink)//interpolation_step), error_sd, error_average),interpolation_step)
error_sine = lb.signal_interpolation(lb.z_transform(np.random.uniform(0, 10, len(y_sine)//interpolation_step), error_sd, error_average),interpolation_step)
error_step = make_variable_error_signal(len(y_step_high), 2.0, 0.2, preperturbation_duration, perturbation_end_index, 5.0, 0.2)
error_isometric = lb.signal_interpolation(lb.z_transform(np.random.uniform(0, 10, len(y_isometric_high)//interpolation_step), error_sd, error_average),interpolation_step)


avatar_random = y_random + error_white
avatar_pink = y_pink + error_pink
avatar_sine = y_sine + error_sine
avatar_step_high = y_step_avatar_high + error_step
avatar_step_low = y_step_avatar_low + error_step
avatar_isometric_high = y_isometric_high + error_isometric
avatar_isometric_low = y_isometric_low + error_isometric

# =========================================================
# PATHS
# =========================================================
# background_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Presentations\2026 Human Movement Variability and Great Plains Biomechanics Conferences 2026\Figures'
# os.chdir(background_directory)

background_path = "background.jpg"
avatar_path = "avatar.png"


# =========================================================
# RUN
# =========================================================
figsize=(12, 12)

# plot_movable_pink_line_with_bg_and_avatar(y_isometric_high, background_path, avatar_path, avatar_y_data=avatar_isometric_high, bg_alpha=0.8, avatar_x=0, avatar_scale=0.3, avatar_rescale_with_zoom=True, figsize=figsize, autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval, linewidth=15, xlim=(-0.5,4), ylim=(-2,62))
# plot_movable_pink_line_with_bg_and_avatar(y_isometric_low, background_path, avatar_path, avatar_y_data=avatar_isometric_low, bg_alpha=0.8, avatar_x=0, avatar_scale=0.3, avatar_rescale_with_zoom=True, figsize=figsize, autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval, linewidth=15, xlim=(-0.5,4), ylim=(-2,62))
# plot_movable_pink_line_with_bg_and_avatar(y_step_high, background_path, avatar_path, avatar_y_data=avatar_step_high, bg_alpha=0.8, avatar_x=0, avatar_scale=0.3, avatar_rescale_with_zoom=True, figsize=figsize, autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval, linewidth=15, xlim=(-0.5,4), ylim=(-2,62))
# plot_movable_pink_line_with_bg_and_avatar(y_step_low, background_path, avatar_path, avatar_y_data=avatar_step_low, bg_alpha=0.8, avatar_x=0, avatar_scale=0.3, avatar_rescale_with_zoom=True, figsize=figsize, autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval, linewidth=15, xlim=(-0.5,4), ylim=(-2,62))
plot_movable_pink_line_with_bg_and_avatar(y_random, background_path, avatar_path, avatar_y_data=avatar_random, bg_alpha=0.8, avatar_x=0, avatar_scale=0.3, avatar_rescale_with_zoom=True, figsize=figsize, autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval, linewidth=15, xlim=(-0.5,4), ylim=(-2,62))
plot_movable_pink_line_with_bg_and_avatar(y_sine, background_path, avatar_path, avatar_y_data=avatar_sine, bg_alpha=0.8, avatar_x=0, avatar_scale=0.3, avatar_rescale_with_zoom=True, figsize=figsize, autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval, linewidth=15, xlim=(-0.5,4), ylim=(-2,62))
plot_movable_pink_line_with_bg_and_avatar(y_pink, background_path, avatar_path, avatar_y_data=avatar_pink, bg_alpha=0.8, avatar_x=0, avatar_scale=0.3, avatar_rescale_with_zoom=True, figsize=figsize, autoplay_speed=autoplay_speed, autoplay_interval=autoplay_interval, linewidth=15, xlim=(-0.5,4), ylim=(-2,62))