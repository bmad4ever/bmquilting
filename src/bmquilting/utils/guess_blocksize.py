from .._internal.blocksize_heuristics import analyze_freq_spectrum, analyze_keypoint_scales, NumPixels, SizeWeightPairs
from math import ceil
import numpy as np
import heapq


def _find_sync_wavelens(pairs, lower, upper, n):
    best_values = []  # stores the top n values as (distance_sum, number) tuples

    for num in range(lower, upper + 1):
        distance_sum = 0
        for d, w in pairs:
            if num < d:
                distance = d - num
            else:
                remainder = num % d
                distance = min(remainder, d - remainder)

            distance_sum += distance * w

        heapq.heappush(best_values, (-distance_sum, num))
        if len(best_values) > n:
            heapq.heappop(best_values)

    best_values = [(-ds, num) for ds, num in best_values]  # convert back to positive distance sums
    distances = [val[0] for val in best_values]

    if len(distances) == 0:  # edge case
        return []

    threshold = np.mean(distances) + np.std(distances)
    relevant = [val[1] for val in best_values if val[0] <= threshold]
    return relevant


def _make_guess(pairs: SizeWeightPairs, min_dim: NumPixels, max_block_size: NumPixels | None = None) -> NumPixels:
    """
    :param min_dim: shortest edge length in the source image/latent
    """
    default_value = round(min_dim / 3)  # returned in edge cases

    if len(pairs) == 0:  # an edge case; maybe a blank image is sent...
        return default_value

    # dev note:
    #  it is important to keep at least one block of addressable space for good multiples (where patterns meet).
    #
    #    let "m" be the best multiple, and "b" the potential block size
    #       this implementation tries to ensure that:   min_dim - b >= m
    #    (using a multiple of "m" could help increasing diversity:  min_dim - b >= km, where k is int)
    #
    #  the size of b should fit the size of the biggest repeating pattern,
    #  so that it is not lost due to a small block size.
    #  to select this size is, however, tricky; and it must also not compromise the above condition.
    #
    #    grossly simplifying, this implementation considers that the block size "b" should be equal to "m"
    #    thus, the search upper bound can be obtained by replacing "m" w/ "b":   b < min_dim - b
    #    this allows for a "simple" implementation
    #
    #  unlike previous implementation, pairs with big sizes are not discarded.
    #  these pairs will add weight to higher multiples, skewing the block size toward a high multiple.
    #  since max size is already limited to half min_dim the obtained block size won't be "too big".

    pairs.sort()
    block_size_lower_bound = pairs[0][0]  # the smallest possible distance between a pattern
    block_size_upper_bound = min_dim // 2  # b <= min_dim - b <-> b <= min_dim/2
    if max_block_size is not None and max_block_size < block_size_upper_bound:
        block_size_upper_bound = max_block_size

    rel = _find_sync_wavelens(pairs, block_size_lower_bound, block_size_upper_bound,
                              ceil((block_size_upper_bound - block_size_lower_bound) / 10))

    if len(rel) == 0:  # edge case
        return default_value

    return max(rel)  # can afford to go for the max since it won't go over more than half of min_dim


def _filter_pairs_by_weight(pairs: SizeWeightPairs, weight_percentage_threshold):
    total_weight = sum(weight for _, weight in pairs)
    threshold = total_weight * (weight_percentage_threshold / 100)
    filtered_pairs = [(divisor, weight) for divisor, weight in pairs if weight >= threshold]
    return filtered_pairs


def guess_nice_block_size(
        src: np.ndarray,
        heuristic: str | bool = "fft",
        max_block_size: NumPixels | None = None
) -> NumPixels:
    """
    Block size guess heuristic selector.

    :param src: must be single channel if "fft" is used.
    :param heuristic:
        - "fft": FFT-based analysis only (default)
        - "sift": SIFT keypoint-based analysis only
        - "both": FFT + SIFT combined analysis
        - bool: backward-compatible flag for old `freq_analysis_only` behavior
            * True  => FFT only
            * False => both, FFT + SIFT
    :param max_block_size: further restricts the upper bound for the guess.
                           the actually used upper bound might be lower than max_block_size.
    """

    # Backward compatibility with previous freq_analysis_only boolean interface
    if isinstance(heuristic, bool):
        heuristic = "fft" if heuristic else "both"

    heuristic = str(heuristic).strip().lower()
    if heuristic not in {"fft", "sift", "both"}:
        raise ValueError("heuristic must be 'fft', 'sift', 'both' or boolean (legacy), got %r" % heuristic)

    freq_analysis_only = heuristic == "fft"

    def normalize_weights(pairs: SizeWeightPairs):
        if not pairs:
            return []
        weights = [weight for index, weight in pairs]
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_pairs = [(index, weight / total_weight) for index, weight in pairs]
        else:
            normalized_pairs = [(index, 0) for index, weight in pairs]
        return normalized_pairs

    # src should come with normalized float values already for FFT
    if heuristic in {"fft", "both"}:
        src_float32 = np.float32(src)/255.0 if src.dtype != np.float32 else src
        freq_analysis_pairs = analyze_freq_spectrum(src_float32)
    else:
        freq_analysis_pairs = []

    # here the image needs to go with integer, 0 to 255, values for SIFT
    if heuristic in {"sift", "both"}:
        src_uint8 = (src * 255.0).astype(np.uint8) if src.dtype != np.uint8 else src
        desc_analysis_pairs = analyze_keypoint_scales(src_uint8)
    else:
        desc_analysis_pairs = []

    # all pairs should come already sorted in descending order w/ respect to weight


    # filter out 'very' small or 'very' big block sizes ( with respect to the image dimensions )
    min_dim = min(src.shape[:2])
    block_size_lower_bound = ceil(min_dim ** (1 / 4))
    block_size_upper_bound = round(min_dim / 1.2) if max_block_size is None else max_block_size
    freq_analysis_pairs = [(dst, w) for dst, w in freq_analysis_pairs if
                           block_size_lower_bound <= dst < block_size_upper_bound]
    desc_analysis_pairs = [(dst, w) for dst, w in desc_analysis_pairs if
                           block_size_lower_bound <= dst < block_size_upper_bound]

    # filter distances whose weight is comparatively low
    freq_analysis_pairs = _filter_pairs_by_weight(freq_analysis_pairs[:6], 12 if freq_analysis_only else 20)
    desc_analysis_pairs = _filter_pairs_by_weight(desc_analysis_pairs[:6], 20)

    final_pairs = [
        *normalize_weights(freq_analysis_pairs),
        *normalize_weights(desc_analysis_pairs)
    ]  # may contain duplicates or multiples, that is expected

    return _make_guess(final_pairs, min_dim, max_block_size)
