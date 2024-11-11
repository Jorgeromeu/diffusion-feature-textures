from math import sqrt
from typing import List

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import torch.nn.functional as F
from dash import Dash, Input, Output, callback, dcc, html
from dash_bootstrap_components.themes import BOOTSTRAP
from einops import einsum, rearrange
from omegaconf import OmegaConf
from torch import Tensor

from text3d2video.artifacts.attn_features_artifact import AttentionFeaturesArtifact
from text3d2video.feature_visualization import reduce_feature_map

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

FRAMES_DIV_ID = "frames-div"


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

    x_imshow = px.imshow(x_square_rgb)
    weights_imshow = px.imshow(
        weights_square,
        color_continuous_scale="viridis",
    )

    for imshow in [weights_imshow, x_imshow]:
        imshow.update_xaxes(showticklabels=False)
        imshow.update_yaxes(showticklabels=False)
        imshow.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        imshow.update_traces(hoverinfo="skip")

    # add pixel marker
    x_imshow.add_trace(
        go.Scatter(
            x=[pixel_data["x"]],
            y=[pixel_data["y"]],
            mode="markers",
            marker=dict(color="red", size=10),
        )
    )

    x_imshow.update_layout(
        xaxis=dict(range=[0, layer_res]),
        yaxis=dict(range=[layer_res, 0]),
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
    Output(FRAMES_DIV_ID, "children"),
    Input(FRAME_IDX_STORE_ID, "data"),
)
def update_generated_frames(frame_idx: dict):
    frame_idx = frame_idx["frame_idx"]
    return generated_frames(
        images, target_indices, do_multiframe_attn, selected_frame=frame_idx
    )


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


def generated_frames(
    images: List,
    target_indices: List[int],
    do_multiframe_attn: bool = False,
    selected_frame: int = 2,
):
    if target_indices is None:
        target_indices = []

    border_style = {"border": "6px solid red"}

    children = []
    for i, img in enumerate(images):
        style = {
            "height": "auto",
            "min-width": "0px",
            "max-width": "300px",
        }

        if do_multiframe_attn and i in target_indices:
            style.update(border_style)

        if not do_multiframe_attn and i == selected_frame:
            style.update(border_style)

        if i != selected_frame:
            style.update({"opacity": 0.5})

        image = html.Img(src=img, style=style)

        children.append(image)

    return html.Div(
        children=children,
        style={
            "display": "flex",
            "justify-content": "flex-start",
            "gap": "0.5em",
        },
    )


def controls(layer_names: List[str], time_steps: List[int]):
    layer_select = dcc.Dropdown(
        id=LAYER_INPUT_ID,
        options=layer_names,
        placeholder="Select Layer",
        value=layer_names[0],
    )

    time_slider = dcc.Slider(
        id=TIMESTEP_INPUT_ID,
        min=0,
        max=len(time_steps) - 1,
        value=time_steps[0],
        step=1,
        marks=time_steps,
    )

    head_select = dcc.Dropdown(
        id=HEAD_INPUT_ID,
        options=[0, 1, 2, 3, 4, 5, 6, 7],
        multi=False,
        placeholder="Select Heads",
        value=0,
    )

    temp_slider = dcc.Input(
        id=TEMPERATURE_INPUT_ID,
        type="number",
        value=1,
        placeholder="Temperature",
    )

    frame_select = dcc.Input(
        id=FRAME_IDX_INPUT_ID,
        type="number",
        value=1,
        placeholder="Frame Index",
    )

    controls = {
        "layer": layer_select,
        "time": time_slider,
        "head": head_select,
        "temp": temp_slider,
        "frame": frame_select,
    }

    return dbc.Card(
        [
            html.Div(
                children=[dbc.Label(lbl), control],
                style={},
            )
            for lbl, control in controls.items()
        ],
        body=True,
        style={"display": "flex", "flex-direction": "column", "gap": "1em"},
    )


def attn_visualization():
    item_style = {
        "height": "auto",
        "width": "50%",
    }

    return html.Div(
        children=[
            dcc.Graph(
                id=X_GRAPH_ID,
                config={"displayModeBar": False},
                style=item_style,
            ),
            dcc.Graph(
                id=WEIGHTS_GRAPH_ID,
                config={"displayModeBar": False},
                style=item_style,
            ),
        ],
        style={
            "display": "flex",
            "justify-content": "left",
            "padding": "1em",
            "gap": "1em",
        },
    )


if __name__ == "__main__":
    artifact_tag = "attn_data:v1"
    attn_data = AttentionFeaturesArtifact.from_wandb_artifact_tag(
        artifact_tag, download=True
    )

    run = attn_data.logged_by()
    conf = OmegaConf.create(run.config)
    target_indices = list(conf.experiment.target_frame_indices)
    do_multiframe_attn = conf.experiment.do_multiframe_attn

    multidict = attn_data.get_features_diskdict()
    images = attn_data.get_images()
    layer_names = sorted(multidict.key_values("layer"))
    time_steps = sorted(multidict.key_values("timestep"))

    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "Attention Visualization"
    app.layout = html.Div(
        children=[
            dcc.Store(id=PIXEL_STORE_ID, data={}),
            dcc.Store(id=FRAME_IDX_STORE_ID, data=0),
            controls(layer_names, time_steps),
            dbc.Card(
                [
                    html.H2("Frames"),
                    html.Div(id=FRAMES_DIV_ID),
                ],
                body=True,
            ),
            dbc.Card([attn_visualization()], body=True),
        ],
        style={
            "padding": "2em",
            "display": "flex",
            "flex-direction": "column",
            "gap": "1em",
        },
    )
    app.run_server(debug=True, dev_tools_hot_reload=True)
