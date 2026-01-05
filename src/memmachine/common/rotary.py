import numpy as np
from datetime import datetime


def datetime_rope(embedding: list[float], dt: datetime) -> list[float]:
    dimensions = len(embedding)
    if dimensions < 100:
        return embedding

    num_pairs_per_cycle = dimensions // 100
    total_pairs = num_pairs_per_cycle * 5

    year = dt.year
    month = dt.month
    day = dt.day
    weekday = dt.weekday()
    hour = dt.hour

    angles = []
    angles += [2 * np.pi * year / 10] * num_pairs_per_cycle
    angles += [2 * np.pi * month / 12] * num_pairs_per_cycle
    angles += [2 * np.pi * day / 31] * num_pairs_per_cycle
    angles += [2 * np.pi * weekday / 7] * num_pairs_per_cycle
    angles += [2 * np.pi * hour / 24] * num_pairs_per_cycle

    angles = np.array(angles)

    embedding_even = embedding[0:dimensions:2]
    embedding_odd = embedding[1:dimensions:2]

    embedding_rot_even = embedding_even.copy()
    embedding_rot_odd  = embedding_odd.copy()

    embedding_rot_even[:total_pairs] = embedding_even[:total_pairs] * np.cos(angles) - embedding_odd[:total_pairs] * np.sin(angles)
    embedding_rot_odd[:total_pairs]  = embedding_even[:total_pairs] * np.sin(angles) + embedding_odd[:total_pairs] * np.cos(angles)

    embedding_rot = np.empty_like(embedding)
    embedding_rot[0:dimensions:2] = embedding_rot_even
    embedding_rot[1:dimensions:2] = embedding_rot_odd
    return embedding_rot.astype(float).tolist()

def datetime_rotary_decay(embedding: list[float], dt: datetime) -> list[float]:
    dimensions = len(embedding)
    if dimensions < 20:
        return embedding

    total_pairs = dimensions // 20

    seconds_since_epoch = dt.timestamp()

    angles = []
    angles += [2 * np.pi * seconds_since_epoch / 8_000_000_000] * total_pairs

    angles = np.array(angles)

    embedding_even = embedding[0:dimensions:2]
    embedding_odd = embedding[1:dimensions:2]

    embedding_rot_even = embedding_even.copy()
    embedding_rot_odd  = embedding_odd.copy()

    embedding_rot_even[:total_pairs] = embedding_even[:total_pairs] * np.cos(angles) - embedding_odd[:total_pairs] * np.sin(angles)
    embedding_rot_odd[:total_pairs]  = embedding_even[:total_pairs] * np.sin(angles) + embedding_odd[:total_pairs] * np.cos(angles)

    embedding_rot = np.empty_like(embedding)
    embedding_rot[0:dimensions:2] = embedding_rot_even
    embedding_rot[1:dimensions:2] = embedding_rot_odd
    return embedding_rot.astype(float).tolist()
