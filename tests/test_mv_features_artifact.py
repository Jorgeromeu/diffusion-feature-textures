import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.multiview_features import extract_multiview_features
from text3d2video.artifacts.multiview_features_artifact import \
    MVFeaturesArtifact
from text3d2video.diffusion import depth2img_pipe
from text3d2video.obj_io import load_objs_as_meshes


class TestMvFeaturesArtifact(unittest.TestCase):

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.pipe = depth2img_pipe(device="cuda")
        self.mesh = load_objs_as_meshes(["data/mixamo-human.obj"], device="cuda")
        self.prompt = "Deadpool"
        self.n_views = 1

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_read_write(self):

        cams, features, ims = extract_multiview_features(
            self.pipe,
            self.mesh,
            self.prompt,
            n_views=self.n_views,
            resolution=512,
            num_inference_steps=2,
            device="cuda",
        )

        # save to folder
        MVFeaturesArtifact.write_to_path(
            self.folder, cameras=cams, features=features, images=ims
        )

        # read from folder
        artifact = MVFeaturesArtifact.from_path(self.folder)

        identifier = {"timestep": 0, "layer": "level_0"}
        view_features = [
            artifact.get_feature(v, identifier) for v in artifact.view_indices()
        ]
        self.assertEqual(len(view_features), self.n_views)

        for f in view_features:
            self.assertEqual(len(f.shape), 4)
            _, _, h, w = f.shape
            self.assertEqual(h, w)
