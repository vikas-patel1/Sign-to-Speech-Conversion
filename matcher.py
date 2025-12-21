def compute_error(known, unknown, keypoints):
    error = 0.0
    for i in keypoints:
        for j in keypoints:
            error += abs(known[i][j] - unknown[i][j])
    return error


def match_gesture(unknown, known_gestures, names, keypoints, tolerance):
    min_error = float("inf")
    best_index = -1

    for i, known in enumerate(known_gestures):
        err = compute_error(known, unknown, keypoints)
        if err < min_error:
            min_error = err
            best_index = i

    if min_error < tolerance:
        return names[best_index]
    return "Unknown"
