import os
from pathlib import Path


def get_first_found_file(root, opts):
    for opt in opts:
        path = os.path.join(root, opt)
        if os.path.isfile(path):
            return path

    return None


def checkpoint_by_name(checkpoints, name):
    if len(checkpoints) == 0:
        raise ValueError("no available checkpoints")

    if len(checkpoints) == 1 and name == "":
        return checkpoints[0]

    if len(checkpoints) > 1 and name == "":
        raise ValueError(
            f"multiple checkpoints, no specified checkpoint (available: {', '.join(checkpoints)})"
        )

    n = []
    for cp in checkpoints:
        fname = Path(cp).name
        if fname.startswith(name):
            return cp

        n.append(name.split("-epoch")[0])

    raise ValueError(
        f"checkpoint not found: {name}. Available: {', '.join(n)}"
    )
