import os
import shutil
from itertools import cycle
from pathlib import Path

import numpy as np
import pyglet as pgl
from flatland.envs.rail_env import RailEnv
from flatland.utils.graphics_pgl import PGLGL
from flatland.utils.rendertools import AgentRenderVariant, RenderLocal
from PIL import Image, ImageDraw, ImageFont

FILE_PATH = Path(__file__).parent.as_posix()

_window = None
reuse_window = True


def debug_show(env: RailEnv, mode="human", save_images_path=None) -> None:
    """Render the environment to a window or to an rgbarray .

    Args:
        env (RailEnv): environment to render

        mode (str, optional):
            "human" or "rgb_array". "human" will open a window and render to it.
            "rgb_array" will return an array of the rendered image. Defaults to "human".

        save_images_path (str, optional):
            If mode is "rgb_array" and save_images_path is not None,
            save the image to save_images_path. Defaults to None.

    Returns:
        PIL.Image: if mode is "rgb_array".
        None: if mode is "human".
    """
    if len(env.dev_pred_dict) == 0:
        env.dev_pred_dict = {i: [] for i in range(env.number_of_agents)}
    if len(env.dev_obs_dict) == 0:
        env.dev_obs_dict = {i: [] for i in range(env.number_of_agents)}

    if mode == "rgb_array":
        render_image = env.render(
            mode="rgb_array",
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS,
            show_debug=False,
            clear_debug_text=True,
            show=False,
            screen_height=env.height * 30,
            screen_width=env.width * 30,
            show_observations=True,
            show_predictions=True,
            show_rowcols=True,
            return_image=True,
        )
        render_image = Image.fromarray(render_image)
        filename = "frame_{timestep:04d}.jpg".format(timestep=env._elapsed_steps)

        # watermark
        watermark = "{timestep:04d}".format(timestep=env._elapsed_steps)
        fontStyle = ImageFont.truetype(os.sep.join([FILE_PATH, "simhei.ttf"]), 25)
        d = ImageDraw.Draw(render_image)
        color = "#FF5A5F"  # (0, 0, 0, 15)
        height = render_image.height
        width = render_image.width
        d.text((width - 100, height - 50), watermark, fill=color, font=fontStyle)

        # d.text((420, 450), watermark, fill=color, font=fontStyle)

        if save_images_path is not None:
            if env._elapsed_steps == 1:
                shutil.rmtree(save_images_path, ignore_errors=True)
                os.makedirs(save_images_path)
            render_image.save(os.sep.join([save_images_path, filename]))

        return np.asarray(render_image)

    elif mode == "human":
        env.render(
            mode="human",
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS,
            show_debug=True,
            clear_debug_text=True,
            show=True,
            screen_height=env.height * 30,
            screen_width=env.width * 30,
            show_observations=True,
            show_predictions=True,
            show_rowcols=True,
            return_image=False,
        )
    else:
        raise ValueError("mode must be 'human' or 'rgb_array'")


def get_window(width, height):
    if reuse_window:
        global _window
        if _window is None:
            _window = pgl.window.Window(
                resizable=True, vsync=False, width=width, height=height
            )
        else:
            _window.set_size(width, height)
        return _window
    else:
        return pgl.window.Window(
            resizable=True, vsync=False, width=width, height=height
        )


# hack PGLGL.show() to avoid image resize and put event loop behind frame render.
def show(self, block=False, from_event=False):
    if not self.window_open:
        self.open_window()

    if self.close_requested:
        if not self.closed:
            self.close_window()
        return

    # tStart = time.time()

    pil_img = self.alpha_composite_layers()

    # convert our PIL image to pyglet:
    bytes_image = pil_img.tobytes()
    pgl_image = pgl.image.ImageData(
        pil_img.width,
        pil_img.height,
        # self.window.width, self.window.height,
        "RGBA",
        bytes_image,
        pitch=-pil_img.width * 4,
    )

    pgl_image.blit(0, 0)
    self._processEvents()
    # tEnd = time.time()
    # print("show time: ", tEnd - tStart)


# hack PGLGL.open_window().
# set its window size to (self.widthPx, self.heightPx)
# alow to reuse a global window object
def open_window(self):
    # print("open_window - pyglet")
    assert self.window_open is False, "Window is already open!"
    self.window = get_window(width=self.widthPx, height=self.heightPx)
    # self.__class__.window.title("Flatland")
    # self.__class__.window.configure(background='grey')
    self.window_open = True

    @self.window.event
    def on_draw():
        # print("pyglet draw event")
        self.window.clear()
        self.show(from_event=True)
        # print("pyglet draw event done")

    @self.window.event
    def on_resize(width, height):
        # print(f"The window was resized to {width}, {height}")
        self.show(from_event=True)
        self.window.dispatch_event("on_draw")
        # print("pyglet resize event done")

    @self.window.event
    def on_close():
        self.close_requested = True


# Overwrite PGLGL.open_window() and PGLGL.show() by our customized ones
PGLGL.show = show
PGLGL.open_window = open_window

#### hack RenderLocal to draw lines between agent and targets
def render_prediction(self, agent_handles, prediction_dict):
    self.render_line_between_agents_and_targets()
    return RenderLocal_render_prediction(self, agent_handles, prediction_dict)


def render_line_between_agents_and_targets(self):
    rt = self.__class__
    targets = set(agent.target for agent in self.env.agents)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    colors = {t: c for t, c in zip(targets, cycle(colors))}
    for agent in self.env.agents:
        if agent.position is not None:
            x0, y0 = np.matmul(agent.position, rt.row_col_to_xy) + rt.x_y_half
            x1, y1 = np.matmul(agent.target, rt.row_col_to_xy) + rt.x_y_half
            self.gl.plot(
                [x0, x1], [y0, y1], color=colors[agent.target], layer=1, opacity=100
            )


RenderLocal_render_prediction = RenderLocal.render_prediction
RenderLocal.render_prediction = render_prediction
RenderLocal.render_line_between_agents_and_targets = (
    render_line_between_agents_and_targets
)
