import hashlib
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    return device


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rng_state():
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    name, values, position, gaussian, cached = np.random.get_state()
    state["numpy"] = {
        "name": name,
        "values": values.tolist(),
        "position": position,
        "gaussian": gaussian,
        "cached": cached,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if "numpy" in state:
        saved = state["numpy"]
        np.random.set_state(
            (
                saved["name"],
                np.asarray(saved["values"], dtype=np.uint32),
                saved["position"],
                saved["gaussian"],
                saved["cached"],
            )
        )
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def load_checkpoint(path):
    # Old Lightning checkpoints contain NumPy scalar metadata. Allow only these
    # numerical types while retaining the restricted weights-only unpickler.
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # NumPy 1.26 keeps the old module name.
        from numpy.core.multiarray import scalar

    def legacy_scalar(*args):
        return scalar(*args)

    # PyTorch 2.5 does not accept (function, import-name) aliases in safe_globals.
    # A narrowly scoped numeric wrapper supplies the historical pickle import name.
    legacy_scalar.__module__ = "numpy.core.multiarray"
    legacy_scalar.__name__ = "scalar"
    numerical = [scalar, legacy_scalar, np.dtype]
    numerical.extend(
        type(np.dtype(t))
        for t in [
            np.float16,
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.bool_,
        ]
    )
    with torch.serialization.safe_globals(numerical):
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as error:
            if "mmap can only be used" not in str(error):
                raise
            return torch.load(path, map_location="cpu", weights_only=True)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
