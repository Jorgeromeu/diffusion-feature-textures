import unittest

from hydra import compose, initialize


class TestScripts(unittest.TestCase):
    def test_generative_rendering_script(self):
        module = "down_blocks.0.attentions.0.transformer_blocks.0.attn1"

        overrides = [
            'run.tags=["testing"]',
            f"generative_rendering.module_paths=['{module}']",
            "generative_rendering.num_inference_steps=1",
            "generative_rendering.num_keyframes=2",
            "animation.n_frames=2",
            "run.wandb=False",
        ]

        for do_uv_init in [True, False]:
            for do_pre_attn in [True, False]:
                for do_post_attn in [True, False]:
                    local_overrides = [
                        f"generative_rendering.do_uv_noise_init={do_uv_init}",
                        f"generative_rendering.do_pre_attn_injection={do_pre_attn}",
                        f"generative_rendering.do_post_attn_injection={do_post_attn}",
                    ]

                    with initialize(version_base="1.2", config_path="../config"):
                        cfg = compose(
                            config_name="generative_rendering",
                            overrides=overrides + local_overrides,
                        )

                        # pylint: disable=import-outside-toplevel
                        from scripts.run_generative_rendering import run

                        run(cfg)
