from codetiming import Timer
from einops import rearrange
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from torchvision.io import read_image
import rerun as rr
import rerun.blueprint as rrb
from text3d2video.rendering import make_feature_renderer
import text3d2video.rerun_util as ru
import torchvision.transforms.functional as TF
import time
from text3d2video.util import project_vertices_to_features
from pytorch3d.renderer import TexturesVertex


def test_projection(anim_tag: str, image_path: str):

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D"),
            rrb.Vertical(
                rrb.Spatial2DView(name="feature_map", origin="cam"),
                rrb.Spatial2DView(name="render", origin="render"),
            ),
        ),
        collapse_panels=True,
    )

    ru.set_logging_state(True)
    rr.init("projection test")
    rr.send_blueprint(blueprint)
    rr.serve()
    ru.pt3d_setup()

    # load mesh
    animation = AnimationArtifact.from_wandb_artifact_tag(anim_tag, download=True)
    mesh = animation.load_unposed_mesh()
    camera = animation.camera(1)

    # load image
    image = read_image("data/collins.png") / 255
    image_pil = TF.to_pil_image(image)
    rr.log("/cam", rr.Image(image_pil))
    ru.log_pt3d_fov_camera("/cam", camera, res=image.shape[1])

    # project image to mesh
    with Timer(text="project features {}"):
        vert_colors = project_vertices_to_features(mesh, camera, image)
    rr.log("mesh", ru.pt3d_mesh(mesh, vertex_colors=vert_colors.cpu()))

    # render textured mesh
    with Timer(text="render features {}"):
        feature_renderer = make_feature_renderer(camera, image.shape[1])

    tex = TexturesVertex(vert_colors.unsqueeze(0))
    mesh.textures = tex
    render = feature_renderer(mesh)
    render_pil = TF.to_pil_image(rearrange(render, "1 h w c -> c h w"))

    rr.log("render", rr.Image(render_pil))


if __name__ == "__main__":
    test_projection("joyful-jump:latest", "data/collins.png")
    time.sleep(10)
