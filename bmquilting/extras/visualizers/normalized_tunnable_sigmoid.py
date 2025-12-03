import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from abc import ABC, abstractmethod
from bmquilting.misc.functions import NormalizedTunableSigmoid, TwoNTS


# ----------------------------------------------------------------------
# DUMMY PACKAGE: funcs (Replace this with your actual funcs package import)
# ----------------------------------------------------------------------

class __FuncWrapper(ABC):
    def func(self, x: np.ndarray) -> np.ndarray:
        x_copy = x.copy()
        self.inplace_func(x_copy)
        return x_copy

    @abstractmethod
    def inplace_func(self, x: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_label(self) -> str:
        pass


class __NTSWithSoftDeadzone:
    """
    Normalized Tunable Sigmoid (NTS) with soft deadzone.

    This is a simplified, non-Numba, vectorized implementation of the NTS
    using a smooth tanh-like function for the deadzone region blending,
    designed to work in-place.
    """
    k: float = 0.5  # NTS shape parameter (k=0 linear, k>0 stronger S-shape).
    deadzone: float = 0.1  # Region |x| < center_deadzone is softened toward zero.
    beta: float = 50.0  # Sharpness of deadzone boundary (large = sharp).
    top: float = 1.0

    def __init__(self, k: float = 0.5, deadzone: float = 0.1, beta: float = 50.0):
        self.k = k
        self.deadzone = deadzone
        self.beta = beta
        self.top = 1.0  # Hardcoded top limit

    def inplace_func(self, x: np.ndarray) -> None:
        D = self.deadzone
        T = self.top
        k = self.k
        beta = self.beta

        T_prime = T - D  # Effective range for scaling

        # 1. Base NTS calculation: S_rational(x)
        # S(x) = (x - kx) / (k - 2k|x| + 1)
        # This function operates on the range [-1, 1] and outputs [-1, 1].

        # Save sign
        x_sign = np.sign(x)
        x_abs = np.abs(x)

        # Normalize the input to the full range [-1, 1] for the NTS formula
        # x_norm = x / T
        x_norm = x_abs / T

        # S_rational = (x_norm - k * x_norm) / (k - 2 * k * x_norm + 1)
        # We perform the calculation using x_norm as the working array (S_raw)

        # Numerator: x_norm * (1 - k)
        S_raw = x_norm * (1.0 - k)

        # Denominator: k - 2 * k * x_norm + 1
        denominator = k - 2.0 * k * x_norm + 1.0

        # S_raw = Numerator / Denominator (Range [0, 1])
        S_raw /= denominator

        # 2. Soft Deadzone Blending (The key difference)

        # The blending factor (alpha) controls the mix between S_raw (full scaling)
        # and 0 (the value inside the deadzone).

        # alpha uses a sigmoid/tanh-like transition:
        # alpha is 0 when |x| < D, and 1 when |x| > D, with smoothness defined by beta.
        # We use a simple exponential curve for smooth, fast transition:
        # alpha = 1.0 / (1.0 + exp(-beta * (|x| - D)))

        # Argument for blending function: beta * (|x| - D)
        blend_arg = beta * (x_abs - D)

        # Smoothness transition: S = 1 / (1 + exp(-arg))
        blend_factor = 1.0 / (1.0 + np.exp(-blend_arg))

        # Apply the blend: S_final = S_raw * blend_factor
        S_final = S_raw * blend_factor

        # 3. Final Sigmoid Transform: y = 0.5 + 0.5 * sign * S_final
        x[:] = 0.5 + 0.5 * x_sign * S_final

        np.clip(x, 0.0, 1.0, out=x)

    def get_label(self) -> str:
        """Returns the specific label for the NTSWithSoftDeadzone."""
        return f'NTS Soft Deadzone (k={self.k:.2f}, DZ={self.deadzone:.2f}, β={self.beta:.1f})'


# ----------------------------------------------------------------------
# MATPLOTLIB INTERACTIVE VISUALIZATION SCRIPT
# ----------------------------------------------------------------------

# 1. Setup figure and axes for plot and sliders
fig = plt.figure(figsize=(10, 8))
ax = fig.add_axes([0.1, 0.35, 0.8, 0.6])  # Main plot axes (top part of the figure)
fig.subplots_adjust(bottom=0.3)

# 2. Create the function instance
# Initial parameters for the NTSWithSoftDeadzone function
initial_k = -0.5
initial_deadzone = 0.0
initial_beta = .5


nts_func = NormalizedTunableSigmoid(initial_k, initial_deadzone, initial_beta)
two_nts_func = TwoNTS(initial_k)
target_func = nts_func
nts_label = 'Normalized Tunable Sigmoid (NTS) with Soft Deadzone'
two_nts_label = 'Two NTS functions composite'

# 3. Define the data range
X_MAX = 1.2
X_AXIS = np.linspace(-X_MAX, X_MAX, 500)

# 4. Plot the initial function curve
Y_AXIS = target_func.func(X_AXIS)
line, = ax.plot(X_AXIS, Y_AXIS, lw=3, color='#3498db')
ax.set_title(nts_label, fontsize=14, fontweight='bold')
ax.set_xlabel('Input Magnitude (normalized)', fontsize=12)
ax.set_ylabel('Output Factor (Blur Weight)', fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, linestyle='--', alpha=0.6)

# Add reference lines
ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6)
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6)
deadzone_span = ax.axvspan(-initial_deadzone, initial_deadzone, alpha=0.1, color='blue', label='Deadzone')
ax.legend([line], [target_func.get_label()], loc='upper left')

# 5. Create axes for the sliders
slider_color = '#d1d1d1'
k_ax = fig.add_axes([0.1, 0.20, 0.8, 0.03], facecolor=slider_color)
deadzone_ax = fig.add_axes([0.1, 0.15, 0.8, 0.03], facecolor=slider_color)
beta_ax = fig.add_axes([0.1, 0.10, 0.8, 0.03], facecolor=slider_color)

# 6. Create the sliders
k_slider = Slider(
    ax=k_ax,
    label='Tuning Factor k (Curviness)',
    valmin=-.99,
    valmax=.99,
    valinit=initial_k,
    valstep=0.01
)

deadzone_slider = Slider(
    ax=deadzone_ax,
    label='Deadzone',
    valmin=0.0,
    valmax=0.8,
    valinit=initial_deadzone,
    valstep=0.01
)

beta_slider = Slider(
    ax=beta_ax,
    label='Beta β (Deadzone Sharpness)',
    valmin=0.0,
    valmax=1.0,
    valinit=initial_beta,
    valstep=.01
)


# 7. Define the update function
def update(val):
    """Recalculates the function and updates the plot based on slider values."""

    # Update function parameters
    new_k = k_slider.val
    new_deadzone = deadzone_slider.val
    new_beta = beta_slider.val

    target_func.k = new_k
    target_func.deadzone = new_deadzone
    target_func.beta = new_beta

    # Recalculate Y data
    new_y = target_func.func(X_AXIS)
    line.set_ydata(new_y)

    deadzone_span.set_bounds([-new_deadzone, 0, new_deadzone*2, 1])

    # Update the label
    line.set_label(target_func.get_label())
    ax.legend([line], [target_func.get_label()], loc='upper left')

    fig.canvas.draw_idle()


def swap_func_to_plot(val):
    global target_func

    # Check current state and switch
    if target_func == nts_func:
        target_func = two_nts_func
        ax.set_title(two_nts_label)
    else:
        target_func = nts_func
        ax.set_title(nts_label)

    update(0)

# 8. Attach the update function to the slider events
k_slider.on_changed(update)
deadzone_slider.on_changed(update)
beta_slider.on_changed(update)

# 9. Add a button to change the plotted func
button_ax = plt.axes([0.7, 0.05, 0.2, 0.035]) # Position the button
switch_button = Button(button_ax, 'Change Function')
switch_button.on_clicked(swap_func_to_plot)

# 9. Display the plot
plt.show()