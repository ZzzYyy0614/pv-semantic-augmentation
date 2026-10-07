from dataclasses import replace
import pytest
from PIL import Image
from pvaug.config import CLASSES
from pvaug.data import Sample, audit_samples, write_manifest, read_manifest, select_generated
from pvaug.demo import make_demo


def test_relative_paths_and_domain_validation(tmp_path):
    image = tmp_path / "images" / "sample.png"
    image.parent.mkdir()
    Image.new("RGB", (16, 16), "gray").save(image)
    sample = Sample(image, CLASSES[0], "train")
    path = tmp_path / "manifests" / "real.csv"
    write_manifest(path, [sample])
    assert read_manifest(path, CLASSES)[0].image == image.resolve()
    write_manifest(path, [replace(sample, domain="generated", split="test")])
    with pytest.raises(ValueError, match="only in train"):
        read_manifest(path, CLASSES)


def test_detects_same_pixels_saved_with_different_encoding(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    image = Image.new("RGB", (16, 16), "gray")
    image.save(first, compress_level=0)
    image.save(second, compress_level=9)
    assert first.read_bytes() != second.read_bytes()
    with pytest.raises(ValueError, match="Data leakage"):
        audit_samples([Sample(first, CLASSES[0], "train"), Sample(second, CLASSES[0], "test")])


def test_module_group_and_mask_alignment(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("RGB", (16, 16), "gray").save(first)
    Image.new("RGB", (16, 16), "black").save(second)
    with pytest.raises(ValueError, match="Module group"):
        audit_samples(
            [
                Sample(first, CLASSES[0], "train", group="module1"),
                Sample(second, CLASSES[0], "test", group="module1"),
            ]
        )
    with pytest.raises(ValueError, match="size mismatch"):
        mask = tmp_path / "mask.png"
        Image.new("L", (8, 8)).save(mask)
        audit_samples([Sample(first, CLASSES[0], "train", mask=mask)])


def test_generated_counts_exclude_normal_and_do_not_silently_resample(tmp_path):
    manifest = make_demo(tmp_path / "data")
    real = [s for s in read_manifest(manifest, CLASSES) if s.split == "train"]
    fake = [replace(s, domain="generated") for s in real]
    selected = select_generated(real, fake, 1.0, CLASSES, seed=42)
    assert len(selected) == 8 and all(s.label != "defect_free" for s in selected)
    with pytest.raises(ValueError, match="Need 4"):
        select_generated(real, fake, 2.0, CLASSES, seed=42)


def test_import_legacy_integer_order_and_path_remapping(tmp_path):
    from pvaug.data import import_legacy_lists

    root = tmp_path / "local"
    root.mkdir()
    Image.new("RGB", (16, 16), "gray").save(root / "cell.png")
    Image.new("L", (16, 16)).save(root / "mask.png")
    source = tmp_path / "original_train.txt"
    source.write_text("/disk2/EL/cell.png;/disk2/EL/mask.png;1\n", encoding="utf-8")
    output = tmp_path / "real.csv"
    order = ["black_core", "micro_crack"]
    import_legacy_lists(output, order, source, path_map={"/disk2/EL": str(root)})
    sample = read_manifest(output, order)[0]
    assert sample.label == "micro_crack" and sample.image == root / "cell.png"
    assert sample.mask == root / "mask.png" and sample.split == "train"
