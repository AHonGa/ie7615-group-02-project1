import unittest

from src import utils


class UtilsApiTests(unittest.TestCase):
    def test_training_helpers_are_available(self):
        self.assertTrue(hasattr(utils, "set_seed"))
        self.assertTrue(hasattr(utils, "get_device"))
        self.assertTrue(hasattr(utils, "train_model"))


if __name__ == "__main__":
    unittest.main()
