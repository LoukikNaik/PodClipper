"""Duration-based safety limits shared by the CLI and hosted worker."""

import math


def reel_cap(duration, cfg):
    settings = cfg.analyze
    maximum = getattr(settings, 'max_reel_cap', 30)
    override = settings.target_clips
    if override is not None:
        return min(override, maximum)
    minimum = getattr(settings, 'min_reel_cap', 3)
    interval = getattr(settings, 'seconds_per_reel', 300.0)
    return min(maximum, max(minimum, math.ceil(duration / interval)))
