import os
import unittest

from model.instance import Instance
from model import utils


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


class TestUtils(unittest.TestCase):
    def test_make_inference_text__yield_title_with_separator_if_no_abstract_avail(self):
        instance = Instance(title="asdf", abstract=None)

        self.assertEqual(utils.make_inference_text(instance), "asdf ||||| ")

    def test_make_inference_text__yield_concatenated_text(self):
        instance = Instance(title="asdf", abstract="fdsa")

        self.assertEqual(utils.make_inference_text(instance), "asdf ||||| fdsa")
