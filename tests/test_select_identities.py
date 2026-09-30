import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import select_identities


class SelectIdentitiesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.pool_dir = self.root / "shared_pool"
        self.logs_dir = self.root / "logs"
        self.docs_dir = self.root / "docs"
        for identity_id, image_count in ((3, 23), (7, 24), (8335, 25), (1212, 23)):
            identity_dir = self.pool_dir / str(identity_id)
            identity_dir.mkdir(parents=True)
            for image_index in range(image_count):
                (identity_dir / f"{image_index}.jpg").touch()

    def run_selection(self, identity_ids):
        args = [
            "select_identities.py",
            "--identity-ids",
            *(str(identity_id) for identity_id in identity_ids),
            "--diversity-review",
            "Visual variety reviewed.",
        ]
        with (
            patch.object(select_identities, "SHARED_POOL_DIR", self.pool_dir),
            patch.object(select_identities, "LOGS_DIR", self.logs_dir),
            patch.object(select_identities, "DOCS_DIR", self.docs_dir),
            patch.object(sys, "argv", args),
        ):
            select_identities.main()

    def test_accepts_four_identities_at_23_to_25_images(self):
        self.run_selection([3, 7, 8335, 1212])

        report = json.loads((self.logs_dir / "selected_identities.json").read_text())
        self.assertTrue(report["counts_verified"])
        self.assertEqual(
            [row["num_images"] for row in report["selected"]],
            [23, 24, 25, 23],
        )

    def test_rejects_identity_outside_required_range(self):
        for image in (self.pool_dir / "3").glob("*.jpg"):
            image.unlink()
        (self.pool_dir / "3" / "replacement.jpg").touch()

        with self.assertRaises(SystemExit):
            self.run_selection([3, 7, 8335, 1212])
        self.assertFalse((self.logs_dir / "selected_identities.json").exists())

    def test_rejects_fewer_than_four_identities(self):
        with self.assertRaises(SystemExit):
            self.run_selection([3, 7, 8335])


if __name__ == "__main__":
    unittest.main()