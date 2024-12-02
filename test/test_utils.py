import unittest

from text3d2video.util import ordered_sample


class TestUtils(unittest.TestCase):
    def test_ordered_sample(self):
        data = ["a", "b", "c", "d", "e"]
        sample = ordered_sample(data, 0)
        self.assertEqual(sample, [])

        sample = ordered_sample(data, 1)
        self.assertEqual(sample, [0])

        sample = ordered_sample(data, 3)
        self.assertEqual(sample, [0, 2, 4])

        data = ["a", "b", "c", "d", "e"]

        sample = ordered_sample(data, 100)
        self.assertEqual(sample, data)

        sample = ordered_sample(data, 5)
        self.assertEqual(sample, data)
