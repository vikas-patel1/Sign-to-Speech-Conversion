import numpy as np


def compute_distance_matrix(hand_landmarks):
    hand = np.array(hand_landmarks, dtype=float)
    n = len(hand)

    dist_matrix = np.zeros((n, n), dtype=float)
    palm_size = np.linalg.norm(hand[0] - hand[9])

    if palm_size == 0:
        return dist_matrix

    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(hand[i] - hand[j]) / palm_size

    return dist_matrix
