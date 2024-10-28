from math import sqrt
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from dash import Dash, Input, Output, callback, dcc, html
from dash_bootstrap_components.themes import BOOTSTRAP
from einops import einsum, rearrange
from torch import Tensor
from torchvision.io import read_image

from text3d2video.disk_multidict import TensorDiskMultiDict
from text3d2video.feature_visualization import reduce_feature_map

out_path = Path("outs/tensors")
multidict_path = out_path / "tensors"

multidict = TensorDiskMultiDict(multidict_path)

layer_names = sorted(multidict.key_values("layer"))
time_steps = sorted(multidict.key_values("timestep"))

n_imgs = len(list(out_path.iterdir())) - 1
images = [
    TF.to_pil_image(read_image(str(out_path / f"image_{i}.png"))) for i in range(n_imgs)
]


TIMESTEP_INPUT_ID = "time-input"
TEMPERATURE_INPUT_ID = "temp-input"
LAYER_INPUT_ID = "layer-input"
HEAD_INPUT_ID = "head-input"
FRAME_IDX_INPUT_ID = "frame-idx-input"

X_GRAPH_ID = "image-graph"
WEIGHTS_GRAPH_ID = "weights-graph"
FRAME_IMG_ID = "frame-img"

FRAME_IDX_STORE_ID = "frame-idx-store"
PIXEL_STORE_ID = "pixel-store"


@callback(
    Output(X_GRAPH_ID, "figure"),
    Output(WEIGHTS_GRAPH_ID, "figure"),
    Input(HEAD_INPUT_ID, "value"),
    Input(LAYER_INPUT_ID, "value"),
    Input(TIMESTEP_INPUT_ID, "value"),
    Input(TEMPERATURE_INPUT_ID, "value"),
    Input(FRAME_IDX_STORE_ID, "data"),
    Input(PIXEL_STORE_ID, "data"),
)
def plot_attention(
    head_idx: int,
    layer_name: str,
    time_step_idx: int,
    temperature: float,
    frame_idx: dict,
    pixel_data: dict,
):
    if temperature is None:
        temperature = 1

    frame_idx = frame_idx["frame_idx"]

    time_step_idx = int(time_step_idx)

    identifier = {"layer": layer_name, "timestep": time_steps[time_step_idx]}

    x = multidict[identifier | {"name": "x"}][frame_idx]
    qrys = multidict[identifier | {"name": "query"}][frame_idx]
    keys = multidict[identifier | {"name": "key"}][frame_idx]

    n_heads = 8
    layer_res = int(sqrt(qrys.shape[0]))
    inner_dim = keys.shape[-1]
    head_dim = inner_dim // n_heads

    qrys_multihead = qrys.view(-1, n_heads, head_dim).transpose(1, 2)
    keys_multihead = keys.view(-1, n_heads, head_dim).transpose(1, 2)

    # get pixel index in flattened tensor
    pixel_coord = Tensor([pixel_data["x"], pixel_data["y"]]).int()
    pixel_idx_flat = pixel_coord[1] * layer_res + pixel_coord[0]

    # get query embedding for pixel and head
    pixel_qry = qrys_multihead[pixel_idx_flat, :, head_idx]

    # get keys and values for the given head
    head_keys = keys_multihead[:, :, head_idx]

    # compute weights
    weights = einsum(pixel_qry, head_keys, "d , n d -> n")
    weights = F.softmax(weights / (temperature * sqrt(head_dim)))

    # reshape weights and values
    # weights_square = rearrange(weights, "(h w) -> h w", w=layer_res)
    weights_square = rearrange(weights, "(n h w) -> h (n w)", w=layer_res, h=layer_res)

    x_square = rearrange(x, "(h w) d -> d h w", h=layer_res)
    x_square_rgb = reduce_feature_map(x_square)

    weights_imshow = px.imshow(
        weights_square, color_continuous_scale="viridis", title="Weights"
    )
    weights_imshow.update_xaxes(showticklabels=False)
    weights_imshow.update_yaxes(showticklabels=False)
    weights_imshow.update_layout(
        coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), title="x"
    )

    x_imshow = px.imshow(x_square_rgb, color_continuous_scale="viridis", title="X")
    x_imshow.update_xaxes(showticklabels=False)
    x_imshow.update_yaxes(showticklabels=False)
    x_imshow.update_layout(
        coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), title="x"
    )
    x_imshow.update_traces(hoverinfo="skip")

    x_imshow.add_trace(
        go.Scatter(
            x=[pixel_data["x"]],
            y=[pixel_data["y"]],
            mode="markers",
            marker=dict(color="red", size=10),
        )
    )

    return x_imshow, weights_imshow


@callback(
    Output(FRAME_IDX_STORE_ID, "data"),
    Input(FRAME_IDX_INPUT_ID, "value"),
)
def update_frame(value):
    if value is None:
        value = 0

    return {"frame_idx": value}


@callback(
    Output(PIXEL_STORE_ID, "data"),
    Input(X_GRAPH_ID, "clickData"),
)
def update_pixel(clickData):
    if clickData is None:
        return {"x": 0, "y": 0}

    x = clickData["points"][0]["x"]
    y = clickData["points"][0]["y"]

    return {"x": x, "y": y}


@callback(
    Output(FRAME_IMG_ID, "src"),
    Input(FRAME_IDX_STORE_ID, "data"),
)
def update_frame_img(frame_idx: dict):
    frame_idx = int(frame_idx["frame_idx"])

    img_path = str((out_path / f"image_{frame_idx}.png").absolute())
    img = TF.to_pil_image(read_image(img_path))
    return img_path


img_path = str((out_path / "image_0.png").absolute())
img = TF.to_pil_image(read_image(img_path))


app = Dash(external_stylesheets=[BOOTSTRAP])
app.title = "Attention Visualization"
app.layout = html.Div(
    children=[
        dcc.Store(id=PIXEL_STORE_ID, data={}),
        dcc.Store(id=FRAME_IDX_STORE_ID, data=0),
        html.H1("Attention Visualization"),
        dbc.Row(
            [
                dbc.Col(html.Img(id=FRAME_IMG_ID, src=img)),
                dbc.Col(dcc.Graph(id=X_GRAPH_ID, config={"displayModeBar": False})),
                dbc.Col(
                    dcc.Graph(id=WEIGHTS_GRAPH_ID, config={"displayModeBar": False})
                ),
            ],
        ),
        dcc.Slider(
            id=TIMESTEP_INPUT_ID,
            min=0,
            max=len(time_steps) - 1,
            value=time_steps[0],
            step=1,
            marks=time_steps,
        ),
        dcc.Dropdown(
            id=LAYER_INPUT_ID,
            options=layer_names,
            placeholder="Select Layer",
            value=layer_names[0],
        ),
        dcc.Dropdown(
            id=HEAD_INPUT_ID,
            options=[0, 1, 2, 3, 4, 5, 6, 7],
            multi=False,
            placeholder="Select Heads",
            value=0,
        ),
        dcc.Input(
            id=TEMPERATURE_INPUT_ID,
            type="number",
            value=1,
            placeholder="Temperature",
        ),
        dcc.Input(
            id=FRAME_IDX_INPUT_ID,
            type="number",
            value=1,
            placeholder="Frame Index",
        ),
    ],
    style={
        "padding": "2em",
        "display": "flex",
        "flex-direction": "column",
        "gap": "1em",
    },
)
app.run()
