import unittest

from hydra import compose, initialize


class TestScripts(unittest.TestCase):

    common_overrides = ["run.wandb=False"]

    def test_generative_rendering_script(self):

        with initialize(config_path="../config"):

            layer_name = "down_blocks.0.attentions.0.transformer_blocks.0.attn1"

            cfg = compose(
                config_name="generative_rendering",
                overrides=self.common_overrides
                + [
                    "generative_rendering.num_inference_steps=1",
                    f"generative_rendering.module_paths=[{layer_name}]",
                    "inputs.animation_n_frames=2",
                    "generative_rendering.chunk_size=2",
                ],
            )

            # pylint: disable=import-outside-toplevel
            from scripts.run_generative_rendering import run

            run(cfg)
