from dataclasses import dataclass, field
from collections.abc import Callable
from numpy import ndarray
from enum import Enum
import cv2

from .misc.functions import FuncWrapper, LogScalingFunc, TwoNTS

type NumPixels = int
"""the number of pixels (integer)"""

type Percentage = float
"""a float value in the range [0,1]"""

type PatchIdx = tuple[int, int, int]
"""patches indexes (texture index, y coord, x coord), where coords are relative to the top-left corner."""

type _2D_Slice = tuple[slice, slice]

type MinCutMethod = Callable[[ndarray, ndarray, int, int, BlendConfig], ndarray]


class Orientation(Enum):
    H = "H"
    V = "V"
    H_AND_V = "H & V"


@dataclass
class BlendConfig:
    sobel_kernel_size: NumPixels = 3
    min_blur_diameter: NumPixels = 3
    max_blur_diameter: NumPixels = 11

    use_vignette: bool = True  # TODO? could add vignette params here, but might be overkill

    blur_size_func: FuncWrapper = field(default_factory=LogScalingFunc)
    """
    Function used to remap the Normalized Gradient Differences (NGDs) computed around the seam.

    The remapped values remain within the interval [0, 1], as they are used to 
    interpolate between the minimum and maximum blur diameters. This remapping 
    introduces a non-linear relationship between the blur size and the NGDs.

    Note:
        NGDs are normalized with respect to the theoretical maximum possible value 
        for the given kernel size used to compute the gradients.

    Default Behavior:
        The `LogScalingFunc` is used with `gain=100` and `top=0.5`.

        - Setting `top=0.5` ensures the function reaches 1 (the maximum blur radius) 
        when the NGD value is half of its theoretical maximum.
        - Setting `gain=100` makes the function concave, steep at the start, meaning 
        the blur radius increases quickly for small gradients and more slowly as 
        it approaches the top.
    """

    blur_shape_func: FuncWrapper = field(default_factory=TwoNTS)
    """
    Function used to shape the transition curve when blending two patches (e.g., in a multi-patch surface).

    This function maps the **signed distance to the seam** to an **interpolation weight** 
    (ranging from 0 to 1) used for blending. It determines how the two patches transition 
    within the specified blur area, influencing the resulting smoothness or sharp change.

    Default Behavior: 
        The `TwoNTS` (Two-NormalizedTunableSigmoid) is used with `k=-0.5`.

        The resulting curve shape is composed of **two scaled sigmoid-like curves** that meet 
        smoothly near the seam (where the distance is zero).
        Visual Analogy: The transition resembles two gentle "hills" meeting at their bases, 
        providing a very smooth, Gaussian-kernel-like transition rather than a simple linear ramp. 
        This makes the blending effect strongest *at* the seam and rapidly decreases away from it.
    """

    adaptive_maximum_filter_number_of_levels: int = 3
    """
    This parameter sets the **number of iterations (or levels)** with different kernel sizes 
        that the adaptive maximum filter executes when computing the blur diameters.

    Context:
    1.  Blur Diameter Calculation: When blending a seam, the initial blur diameter is determined 
        based on the **gradient differences** computed near the seam.
    2.  Diameter Propagation: These initial diameters need to be **propagated** across the texture to 
        determine the correct blend amount for every pixel (i.e., if a pixel's distance to the seam is 
        less than the propagated radius, it needs blending).
    3.  Adaptive Filter: An **adaptive maximum filter** is used for this propagation/expansion. 
        This filter runs with different kernel sizes across multiple iterations to ensure that locally 
        lower blur diameters aren't overshadowed by higher values during expansion.

    Note on Quantization:
        The final quantization interval for the diameters is determined by the *min_blur_diameter* 
        and the *maximum diameter found* in the propagation process (it is **not** based on the 
        theoretical possible maximum defined by `max_blur_diameter`).
    """

    use_blur_radii_limiter: bool = True
    """
    Attempts to mitigate seam blurring artifacts AFTER seam computation.
    This is done by limiting the seam blur radius with respect to its proximity to the overlapping area edges.
    This helps prevent seam artifacts near the edges that could visually give away the generation grid layout.
    """

    def __post_init__(self):
        if self.sobel_kernel_size % 2 == 0:
            raise ValueError(f"{self.sobel_kernel_size = } is invalid, kernel size should be an odd number.")

        if self.max_blur_diameter < self.min_blur_diameter:
            raise ValueError(f"Invalid range: {self.min_blur_diameter = }"
                             f" must be less or equal to {self.max_blur_diameter = }")


@dataclass
class SquarePatchingBlendConfig(BlendConfig):
    use_blur_radii_guess_pathfind_limiter: bool = True
    """
    Attempts to mitigate seam blurring artifacts BEFORE seam computation.
    Prior to computing the seam, make an educated guess of the potential max blur radius.
    When computing the seam the overlapping area is further constrained with respect to this guess to avoid having 
    the seam go near the edges of the overlapping area.

    Only applicable when using pyastar2d to compute the seam.
    """


@dataclass
class GenParams:
    """
    Data used across multiple quilting subroutines.
    Used in quilting.py and make_seamless.py
    """
    block_size: NumPixels
    overlap: NumPixels
    tolerance: Percentage
    blend_config: SquarePatchingBlendConfig | None
    vignette_on_match_template: bool
    """whether to use the blending vignette as a mask when searching for a matching patch"""

    min_cut_search_method: MinCutMethod
    """
    From synthesis_subroutines, the following methods can be used:
        - get_min_cut_patch_mask_horizontal_astar: A* grid based solution. Seams can backtrack.
        - get_min_cut_patch_mask_horizontal_jena2020: purist solution, seams can not backtrack.
    
    The A* solution should be slightly faster due to the C backend; however, there is no direct way to customize the 
    the errors computation, this can only be done indirectly via a proxy texture.    
    
    For a more performant solution --- faster or with a different error computation --- a custom function can be used. 
    """

    match_template_method: int = cv2.TM_SQDIFF
    """
    Match template method used when searching for a patch with a matching overlap section.
    Only TM_SQDIFF and TM_CCOEFF_NORMED are supported.
    """


    def __post_init__(self):
        if not (0.0 <= self.tolerance <= 1.0):
            raise ValueError(f"{self.tolerance = } tolerance should be in the [0,1] range.")

        # Sets a method to adjust match template computed errors so that
        #  the minimum selection behavior is kept unchanged independently of the selected method
        if self.match_template_method == cv2.TM_SQDIFF:
            self._mt_error_adjust = lambda e: e
        elif self.match_template_method == cv2.TM_CCOEFF_NORMED:
            self._mt_error_adjust = lambda e: 1 - e  # values from 0 to 2
        else:
            raise ValueError(f"{self.match_template_method = } is invalid.\n"
                             f"Only TM_SQDIFF and TM_CCOEFF_NORMED are supported.")


    def _compute_min_cut(self, source: ndarray, patch: ndarray) -> ndarray:
        return self.min_cut_search_method(source, patch, self.block_size, self.overlap, self.blend_config)

    @property
    def blend_into_patch(self) -> bool:
        return self.blend_config is not None

    @property
    def bo(self) -> tuple[NumPixels, NumPixels]:
        return self.block_size, self.overlap

    @property
    def bot(self) -> tuple[NumPixels, NumPixels, Percentage]:
        return self.block_size, self.overlap, self.tolerance
