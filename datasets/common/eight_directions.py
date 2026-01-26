import numpy as np


def eight_dirs_by_heading(
    ego_pos,
    ego_theta,
    sv_pos,
    alongside_lat=2.5,
    preceding_long=0.0,
    eps=1e-6,
):
    """
    Use ego heading instead of Frenet mapping for direction bins.
    Return: one of dir_names
    """
    ego_pos_arr = np.asarray(ego_pos, dtype=float)
    sv_pos_arr = np.asarray(sv_pos, dtype=float)
    if ego_pos_arr.shape[-1] != 2 or sv_pos_arr.shape[-1] != 2:
        raise ValueError("ego_pos and sv_pos must both be length-2 vectors")

    rel = sv_pos_arr - ego_pos_arr
    cos_theta = float(np.cos(ego_theta))
    sin_theta = float(np.sin(ego_theta))

    x = cos_theta * rel[0] + sin_theta * rel[1]  # front(+)/rear(-)
    y = -sin_theta * rel[0] + cos_theta * rel[1]  # left(+)/right(-)

    if abs(y) <= alongside_lat + eps:
        return "leftAlongside" if y >= 0 else "rightAlongside"

    if x >= preceding_long - eps:
        return "leftPreceding" if y >= 0 else "rightPreceding"

    return "leftFollowing" if y >= 0 else "rightFollowing"


def eight_dirs(ds_pos_ev, ds_pos_sv, ego_state):

    s_diff = ds_pos_sv[1] - ds_pos_ev[1]  # sv longitudinal offset relative to ego
    d_diff = ds_pos_sv[0] - ds_pos_ev[0]  # sv lateral offset relative to ego (right positive)

    width_half = ego_state.height / 2.0
    length_half = ego_state.width / 2.0

    # Approximate thresholds (tunable)
    s_thresh = length_half
    d_thresh = width_half

    if abs(d_diff) <= d_thresh:
        if s_diff > s_thresh:
            return "preceding"
        elif s_diff < -s_thresh:
            return "following"
    elif abs(s_diff) <= s_thresh:
        if d_diff > d_thresh:
            return "rightAlongside"
        elif d_diff < -d_thresh:
            return "leftAlongside"
    else:
        if s_diff > 0 and d_diff > 0:
            return "rightPreceding"
        elif s_diff > 0 and d_diff < 0:
            return "leftPreceding"
        elif s_diff < 0 and d_diff > 0:
            return "rightFollowing"
        elif s_diff < 0 and d_diff < 0:
            return "leftFollowing"

    return "Unknown"


__all__ = ["eight_dirs_by_heading", "eight_dirs"]
