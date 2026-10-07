import numpy as np
import torch
from pvaug.utils import load_checkpoint


def test_checkpoint_numpy_scalar_metadata_and_classic_torch_format(tmp_path):
    path = tmp_path / "legacy.pt"
    value = {"state_dict": {"weight": torch.ones(3)}, "score": np.float64(0.9)}
    torch.save(value, path)
    restored = load_checkpoint(path)
    assert restored["score"] == 0.9
    assert torch.equal(restored["state_dict"]["weight"], torch.ones(3))
    classic = tmp_path / "classic.pth"
    torch.save({"weight": torch.ones(3)}, classic, _use_new_zipfile_serialization=False)
    assert torch.equal(load_checkpoint(classic)["weight"], torch.ones(3))


def test_rng_snapshot_restores_numpy_python_and_torch():
    import random
    import numpy as np
    import torch
    from pvaug.utils import rng_state, restore_rng, seed_everything

    seed_everything(21)
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(state)
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
