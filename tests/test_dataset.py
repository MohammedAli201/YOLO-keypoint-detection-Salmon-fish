import tempfile
import unittest
from pathlib import Path
from scripts.validate_dataset import validate_row, validate_split


def valid_row():
    return "0 0.5 0.5 0.8 0.8 " + " ".join(["0.5 0.5 2"] * 20)


class DatasetTests(unittest.TestCase):
    def test_valid_row(self):
        self.assertIsNone(validate_row(valid_row()))

    def test_bad_counts_nonfinite_and_visibility(self):
        for row in ["0 0.5", valid_row().replace("0.8", "nan", 1), valid_row()[:-1]+"3"]:
            self.assertIsNotNone(validate_row(row))

    def test_missing_label_is_detected_from_image_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"images/val").mkdir(parents=True)
            (root/"labels/val").mkdir(parents=True)
            (root/"images/val/fish.png").write_bytes(b"placeholder")
            count, errors=validate_split(root,"val")
            self.assertEqual(count,1)
            self.assertTrue(any("missing label" in e for e in errors))
            (root/"labels/val/fish.txt").write_text(valid_row())
            self.assertEqual(validate_split(root,"val")[1],[])

    def test_orphan_label_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"images/val").mkdir(parents=True)
            (root/"labels/val").mkdir(parents=True)
            (root/"labels/val/orphan.txt").write_text(valid_row())
            self.assertTrue(any("no matching image" in e for e in validate_split(root,"val")[1]))


if __name__ == "__main__":
    unittest.main()
