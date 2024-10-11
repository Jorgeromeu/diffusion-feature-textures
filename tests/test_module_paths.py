import shutil
import tempfile
import unittest
from pathlib import Path

from diffusers.models import UNet2DConditionModel

from text3d2video.sd_feature_extraction import find_attn_modules


class TestModulePaths(unittest.TestCase):

    sd_repo = "runwayml/stable-diffusion-v1-5"

    def setUp(self):
        self.unet = UNet2DConditionModel.from_pretrained(self.sd_repo)

    def tearDown(self):
        pass

    def test_find_attn_modules(self):
        attn_modules = find_attn_modules(self.unet)

        print("asdf", len(attn_modules))
