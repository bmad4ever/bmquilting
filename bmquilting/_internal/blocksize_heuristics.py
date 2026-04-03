from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import cv2

from .common import NumPixels
type Weight = float
type SizeWeightPairs = list[tuple[NumPixels, Weight]]


# region ==== SIFT DESCRIPTORS DISTRIBUTION BASED ====

type Area = int
type Label = int


def _find_optimal_clusters(data: list, max_k: int = 6) -> tuple[list[Label], list[...], int, float]:
    if len(np.unique(data)) == 1:  # edge case
        return [0] * len(data), [data[0]], 1, 0

    iters = range(2, max_k + 1)
    best_k = 2
    best_score = -1.0
    best_labels = []
    best_centers = []
    data = np.array(data).reshape(-1, 1)

    for k in iters:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        score = silhouette_score(data, labels, random_state=0)

        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers.reshape(-1), best_k, best_score


def _min_distance_same_label(positions: list[tuple[int, int]], labels: list[Label]) -> \
        tuple[dict[Label, NumPixels], dict[Label, Area]]:
    """
    :return: two dictionaries where the keys are the unique labels:
        the 1st contains the minimum distances found for each label;
        the 2nd contains the area between the points from which the minimum distance was obtained.
    """
    positions = np.array(positions)
    labels = np.array(labels)

    unique_labels = np.unique(labels)
    min_distances = {}
    min_dist_areas = {}

    def max_distance(u, v):
        return max(abs(u[0] - v[0]), abs(u[1] - v[1]))

    def area(u, v):
        return abs(u[0] - v[0]) * abs(u[1] - v[1])

    for _label in unique_labels:
        label_indices = np.nonzero(labels == _label)[0]
        if len(label_indices) < 2:
            min_distances[_label] = 0.0  # not enough points to compute distance
            min_dist_areas[_label] = 0.0
            continue

        label_descriptors = positions[label_indices]  # same label descriptors
        distances = pairwise_distances(label_descriptors, label_descriptors, metric=max_distance)
        areas = pairwise_distances(label_descriptors, label_descriptors, metric=area)
        distances[distances < 1] = np.inf  # remove diagonals and less than 1 pixel away pairs

        # find the minimum distance & its area
        min_distance = np.min(distances)
        min_distances[_label] = min_distance
        min_dist_areas[_label] = np.median(areas[distances == min_distance])

    return min_distances, min_dist_areas


def _inner_square_area(circle_diameter: float) -> float:
    return 2 * (circle_diameter / 2) ** 2


def analyze_keypoint_scales(image: np.ndarray) -> SizeWeightPairs:
    """
    :param image: uint8 image with arbitrary number of channels.
    """
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    if len(keypoints) == 0:  # edge case
        return []
    kp_sizes = [kp.size for kp in keypoints]  # keypoints' diameters, in pixels
    kp_pts = [kp.pt for kp in keypoints]  # keypoints' (y, x) positions

    # cluster keypoints by size.
    # then, consider their size & distance in the analysis, and weight them w/ respect to area covered
    labels, kp_pt_cluster_centers, _, _ = _find_optimal_clusters(kp_sizes)

    # weight clusters with respect to the area coverage on the image
    label_counts = Counter(labels)  # the number of keypoints' of a given label. should be sorted
    labels_coverage = [_inner_square_area(diam) * label_counts[i] for i, diam in enumerate(kp_pt_cluster_centers)]
    dist_weight_pairs = [(round(kp_pt_cluster_centers[i]), w) for i, w in enumerate(labels_coverage)]

    # get the minimum distance between keypoints belonging to the same cluster
    # suppose that each keypoints has at least one area adjacent, so that the min. area is given by min_area * num_kps
    distance_pairs, pairs_areas = _min_distance_same_label(kp_pts, labels)
    dist_weight_pairs.extend([(round(distance_pairs[i]), pairs_areas[i] * label_counts[i])
                              for i, _ in enumerate(labels_coverage)])
    dist_weight_pairs.sort(key=lambda i: i[1], reverse=True)
    return dist_weight_pairs

# endregion ==== SIFT DESCRIPTORS DISTRIBUTION BASED ====


# region ==== FOURIER TRANSFORM BASED ====

def _compute_fft(image):
    dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    return magnitude_spectrum


def _compute_wavelens_of_interest(spectrum: np.ndarray, max_to_fetch: int = 16) -> SizeWeightPairs:
    h, w = spectrum.shape[:2]
    unique_wavelen = set()
    wavelen_magnitude_pairs = {}

    # flatten the spectrum and get the indices of the sorted magnitudes
    flat_indices = np.argsort(spectrum, axis=None)[::-1]  # sort in descending order
    flat_spectrum = spectrum.flatten()

    unique_count = 0

    start_index = 1  # skip the first maximum magnitude
    for flat_index in flat_indices[start_index:]:
        if unique_count >= max_to_fetch:
            break

        # convert flat index to 2D indices
        y, x = np.unravel_index(flat_index, spectrum.shape)
        magnitude = round(flat_spectrum[flat_index])

        # calculate the frequency as the maximum absolute distance from the center
        freq_y = abs(y - h / 2) / h
        freq_x = abs(x - w / 2) / w
        # compute wavelen
        wavelen_y = 1 / freq_y if freq_y > 0 else 0  # don't return infinity when selecting max
        wavelen_x = 1 / freq_x if freq_x > 0 else 0
        wavelen = int(max(wavelen_y, wavelen_x))

        if wavelen not in unique_wavelen:
            unique_wavelen.add(wavelen)
            wavelen_magnitude_pairs[wavelen] = magnitude
            unique_count += 1
        else:
            if magnitude > wavelen_magnitude_pairs[wavelen]:
                wavelen_magnitude_pairs[wavelen] = magnitude

    return list(wavelen_magnitude_pairs.items())


def analyze_freq_spectrum(image: np.ndarray, max_items: int = 16) -> SizeWeightPairs:
    """
    :param image: single channel float32 image
    """
    magnitude_spectrum = _compute_fft(image)
    wlen_mag_pairs = _compute_wavelens_of_interest(magnitude_spectrum, max_items)
    return wlen_mag_pairs

# endregion ==== FOURIER TRANSFORM BASED ====
