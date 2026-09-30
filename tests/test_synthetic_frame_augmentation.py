import random
import unittest

from PIL import Image

from src.build_synthetic_frames import (
    augment_frame_and_boxes,
    make_frame_identity_sets,
    split_sources,
    transform_box,
)


class SyntheticFrameAugmentationTests(unittest.TestCase):
    def test_source_split_uses_largest_remainder_and_keeps_images_disjoint(self):
        expected_counts = {23: (16, 4, 3), 24: (17, 4, 3), 25: (17, 4, 4)}
        for source_count, expected in expected_counts.items():
            with self.subTest(source_count=source_count):
                splits = split_sources(list(range(source_count)), random.Random(42))
                actual = tuple(len(splits[name]) for name in ("train", "val", "test"))
                self.assertEqual(actual, expected)
                images = sum(splits.values(), [])
                self.assertEqual(len(images), len(set(images)))

    def test_frame_schedule_includes_every_identity_when_possible(self):
        identity_ids = [3, 7, 1212, 8335]
        for frame_count in (1, 2, 3, 4, 20):
            with self.subTest(frame_count=frame_count):
                frame_sets = make_frame_identity_sets(
                    identity_ids, frame_count, random.Random(frame_count)
                )
                self.assertEqual(set().union(*map(set, frame_sets)), set(identity_ids))
                self.assertTrue(all(2 <= len(frame_ids) <= 4 for frame_ids in frame_sets))

    def test_horizontal_flip_transforms_box_coordinates(self):
        self.assertEqual(
            transform_box((10, 20, 30, 40), 100, True, 1.0, 0.0),
            (70, 20, 90, 40),
        )

    def test_rotation_uses_frame_center_and_returns_axis_aligned_box(self):
        self.assertEqual(
            transform_box((10, 20, 30, 40), 100, False, 1.0, 90.0),
            (20, 70, 40, 90),
        )

    def test_random_augmentations_keep_face_boxes_within_frame(self):
        annotations = [
            {"class_id": 0, "identity_id": 3, "source_image": "a.jpg", "bbox": (70, 180, 180, 300)},
            {"class_id": 1, "identity_id": 7, "source_image": "b.jpg", "bbox": (430, 190, 540, 310)},
        ]
        frame = Image.new("RGB", (640, 640), "gray")

        for seed in range(100):
            with self.subTest(seed=seed):
                _, transformed, parameters = augment_frame_and_boxes(
                    frame.copy(), annotations, random.Random(seed)
                )
                self.assertEqual(len(transformed), len(annotations))
                self.assertTrue(0.92 <= parameters["scale"] <= 1.08)
                self.assertTrue(-6.0 <= parameters["rotation_deg"] <= 6.0)
                self.assertTrue(0.90 <= parameters["brightness"] <= 1.10)
                self.assertTrue(0.90 <= parameters["contrast"] <= 1.10)
                for annotation in transformed:
                    left, top, right, bottom = annotation["bbox"]
                    self.assertTrue(0 <= left < right <= 640)
                    self.assertTrue(0 <= top < bottom <= 640)


if __name__ == "__main__":
    unittest.main()
