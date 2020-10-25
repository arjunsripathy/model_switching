"""Visualizer class for CarWorlds."""

import math
from os.path import dirname, join
from typing import Dict, Optional

import numpy as np
import pyglet
import pyglet.gl as gl
import pyglet.graphics as graphics

from interact_drive.car import Car
from interact_drive.world import CarWorld, StraightLane

default_asset_dir = join(dirname(__file__), "assets")


class CarVisualizer(object):
    """
    Class that visualizes a CarWorld.

    Attributes:
        world (CarWorld): the CarWorld this CarVisualizer is visualizing.

    """

    def __init__(
        self,
        world: CarWorld,
        name: str = "car_sim",
        follow_main_car: bool = False,
        window_args: Optional[Dict] = None,
        asset_dir: Optional[str] = None,
        display_model = False,
        display_y = False,
    ):
        """
        Initializes this car visualizer.

        Args:
            world: the `CarWorld` to visualize.
            name: the name of the scenario.
            follow_main_car: whether or not the camera should follow the main
                car (once one has been specified).
            window_args: a dict of arguments to pass to the window created by
                the visualizer.
            asset_dir: the directory containing image assets. If not specified,
                we use the `default_asset_dir` specified in this file.
        """
        self.world = world
        self.follow_main_car = follow_main_car

        if window_args is None:
            window_args = dict(
                caption=name, height=600, width=600, fullscreen=False
            )
        else:
            window_args["caption"] = name
        self.window_args = window_args

        self.window = pyglet.window.Window(**self.window_args)
        self.reset()

        if asset_dir is None:
            asset_dir = default_asset_dir
        self.asset_dir = asset_dir

        self.obj_atts = dict()
        
        self.car_sprites = {
            c: self.car_sprite(c, asset_dir=self.asset_dir)
            for c in [
                "orange",
                "gray",
                "blue",
            ]
        }
        
        self.speed_label = pyglet.text.Label(
            "Speed: ",
            font_name="Times New Roman",
            font_size=24,
            x=30,
            y=self.window.height - 60,
            anchor_x="left",
            anchor_y="top",
        )

        self.display_model = display_model
        if (self.display_model):
            self.model_label = pyglet.text.Label(
                "Model: ",
                font_name="Times New Roman",
                font_size=24,
                x=self.window.width - 30,
                y=self.window.height - 30,
                anchor_x="right",
                anchor_y="top",
            )
        self.display_y = display_y
        if (self.display_y):
            self.y_label = pyglet.text.Label(
                "Y: ",
                font_name="Times New Roman",
                font_size=24,
                x=30,
                y=self.window.height - 30,
                anchor_x="left",
                anchor_y="top",
            )


        self._grass = pyglet.image.load(
            join(asset_dir, "grass.png")
        ).get_texture()

        self.obstacle_sprites = {"cone": self.cone_sprite()}

        self.main_car = None

    def centered_image(self, filename):
        """
        Helper function that centers and returns the image located at `filename`.
        """
        # img = pyglet.resource.image(filename)
        img = pyglet.image.load(filename).get_texture()
        img.anchor_x = img.width / 2.
        img.anchor_y = img.height / 2.
        return img

    '''
    Records the width and height for an object
    class in the world.  Currently only supports
    all objects of the same class having the same
    dimension.
    '''
    def set_att(self, obj_name, wh, ang = None):
        if (ang is not None):
            ang *= np.pi/180
        
        if (obj_name in self.obj_atts):
            assert self.obj_atts[obj_name]["wh"] == wh, f"Two {obj_name}s given different size"
            assert self.obj_atts[obj_name]["ang"] == ang, f"Two {obj_name}s given different angle"
        else:
            self.obj_atts[obj_name] = {"wh": wh, "ang": ang}


    def car_sprite(self, color, scale=0.15 / 600.,
                   asset_dir=default_asset_dir) -> pyglet.sprite.Sprite:
        """
        Helper function that returns a sprite of an appropriately-colored car.
        """
        sprite = pyglet.sprite.Sprite(
            self.centered_image(join(asset_dir, 'car-{}.png'.format(color))),
            subpixel=True)
        sprite.scale = scale

        width_height = [sprite.width, sprite.height]
        self.set_att("car", wh = width_height)

        return sprite

    def cone_sprite(self, scale=0.07 / 600.,
                   asset_dir=default_asset_dir) -> pyglet.sprite.Sprite:
        """
        Helper function that returns a sprite of a cone.
        """
        sprite = pyglet.sprite.Sprite(
            self.centered_image(join(asset_dir, 'cone.png')),
            subpixel=True)
        sprite.scale = scale

        width_height = [sprite.width, sprite.height]
        self.set_att("cone", wh = width_height, ang = sprite.rotation)

        return sprite

    def set_main_car(self, index):
        """
        Sets the main car to follow with the camera and display the speed of.
        """
        self.main_car = self.world.cars[index]

    def reset(self):
        """
        Resets the visualized by closing the current window and opening
        a new window.
        """
        self._close_window()
        self._open_window()

    def _open_window(self):
        if self.window is None:
            self.window = pyglet.window.Window(**self.window_args)
            self.window.on_draw = self._draw_world
            self.window.dispatch_event("on_draw")

    def _close_window(self):
        self.window.close()
        self.window = None

    def render(
        self, display: bool = True, return_rgb: bool = False
    ) -> Optional[np.array]:
        """
        Renders the state of self.world. If display=True, then we display
        the result in self.window. If return_rgb=True, we return the result
        as an RGB array.

        Args:
            display: whether to display the result in self.window.
            return_rgb: whether to return the result as an rgb array.

        Returns:
            rgb_representation: If return_rgb=True, we return an np.array
                    of shape (x, y, 3), representing the rendered image.
                    Note: on MacOS systems, the rgb array is twice as large
                    as self.window's width/height parameters suggest.
        """
        pyglet.clock.tick()
        window = self.window

        window.switch_to()
        window.dispatch_events()

        window.dispatch_event("on_draw")

        if return_rgb:
            # copy the buffer into an np.array
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            img_data = buffer.get_image_data()
            arr = np.fromstring(img_data.get_data(), dtype=np.uint8, sep="")

            # default format is RGBA, and we need to turn this into RGB.
            from sys import platform as sys_pf

            if sys_pf == "darwin":
                # if system is mac, the image is twice as large as the window
                width, height = self.window.width * 2, self.window.height * 2
            else:
                width, height = self.window.width, self.window.height

            arr = arr.reshape([width, height, 4])
            arr = arr[::-1, :, 0:3]

        if display:
            window.flip()  # display the buffered image in the window

        if return_rgb:
            return arr

    def _draw_world(self):
        """Draws the world into the pyglet buffer."""
        self.window.clear()

        # start image drawing mode
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        self._center_camera()

        self._draw_background()
        for lane in self.world.lanes:
            self._draw_lane(lane)
        for car in self.world.cars:
            self._draw_car(car)
        for obstacle in self.world.obstacles:
            self._draw_obstacle(obstacle)
        if (hasattr(self.world, "car_clones")):
            for car in self.world.car_clones:
                self._draw_car(car, state_tensor = False)

        # end image drawing mode
        gl.glPopMatrix()

        if self.main_car is not None:
            self.speed_label.text = f"Speed: {self.world.cars[0].state.numpy()[2]:.2f}"
            
            if (self.display_model):
                try:
                    model_name = self.world.cars[0].planner.planner.model.name
                except:
                    model_name = "N/A"
                self.model_label.text = f"Model: {model_name}"
                self.model_label.draw()
            if (self.display_model):
                car_y = self.world.cars[0].state[1]
                self.y_label.text = f"Y: {car_y:.2f}"
                self.y_label.draw()
            

        self.speed_label.draw()

    def _draw_background(self):
        """Draws the grass background."""
        gl.glEnable(self._grass.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(self._grass.target, self._grass.id)

        # specifies radius of grass area in the visualization
        # we assume that no car environment is going to have cars going outside
        radius = 20.0

        # increase this to decrease the size of the grass tiles
        texture_shrink_factor = 5.0

        # v2f: tells gl to draw a background with vertices in 2d, where
        #       coordinates are in float
        # t2f: tells gl how the texture of the background should be read from
        #       the grass file.
        graphics.draw(
            4,
            gl.GL_QUADS,
            (
                "v2f",
                (
                    -radius,
                    -radius,
                    radius,
                    -radius,
                    radius,
                    radius,
                    -radius,
                    radius,
                ),
            ),
            (
                "t2f",
                (
                    0.0,
                    0.0,
                    texture_shrink_factor * radius,
                    0.0,
                    texture_shrink_factor * radius,
                    texture_shrink_factor * radius,
                    0.0,
                    texture_shrink_factor * radius,
                ),
            ),
        )
        gl.glDisable(self._grass.target)

    def _draw_car(self, car: Car, state_tensor = True):
        """Draws a `Car`."""
        if (state_tensor):
            state = car.state.numpy()
        else:
            state = car.state
        color = car.color
        opacity = car.opacity * 255
        sprite = self.car_sprites[color]
        sprite.x, sprite.y = state[0], state[1]
        sprite.rotation = -state[3] * 180.0 / math.pi
        sprite.opacity = opacity
        sprite.draw()

    def _draw_lane(self, lane: StraightLane):
        """Draws a `StraightLane`."""
        # first, draw the asphalt
        gl.glColor3f(0.4, 0.4, 0.4)  # set color to gray
        graphics.draw(
            4,
            gl.GL_QUAD_STRIP,
            (
                "v2f",
                np.hstack(
                    [
                        lane.p - 0.5 * lane.w * lane.n,
                        lane.p + 0.5 * lane.w * lane.n,
                        lane.q - 0.5 * lane.w * lane.n,
                        lane.q + 0.5 * lane.w * lane.n,
                    ]
                ),
            ),
        )
        # next, draw the white lines between lanes

        gl.glColor3f(1.0, 1.0, 1.0)  # set color to white
        graphics.draw(
            4,
            gl.GL_LINES,
            (
                "v2f",
                np.hstack(
                    [
                        lane.p - 0.5 * lane.w * lane.n,
                        lane.q - 0.5 * lane.w * lane.n,
                        lane.p + 0.5 * lane.w * lane.n,
                        lane.q + 0.5 * lane.w * lane.n,
                    ]
                ),
            ),
        )

    def _draw_obstacle(self, obstacle):
        obs_type, (x, y) = obstacle
        obs_sprite = self.obstacle_sprites[obs_type]
        obs_sprite.x = x
        obs_sprite.y = y
        obs_sprite.draw()

    def _center_camera(self):
        """Sets the camera coordinates."""
        if self.main_car is not None and self.follow_main_car:
            x = self.main_car.state[0]
            y = self.main_car.state[1]
        else:
            x, y = 0.0, 0.0
        z = 0.0
        # set the camera to be +1/-1 from the center coordinates
        gl.glOrtho(x - 1.0, x + 1.0, y - 1.0, y + 1.0, z - 1.0, z + 1.0)


##


def main():
    """Visualizes three cars in a merging scenario."""
    from experiments.merging import ThreeLaneTestCar
    from interact_drive.world import ThreeLaneCarWorld
    from interact_drive.car import FixedVelocityCar

    world = ThreeLaneCarWorld()
    our_car = ThreeLaneTestCar(
        world,
        np.array([0, -0.5, 0.8, np.pi / 2]),
        horizon=5,
        weights=np.array([0.1, 0.0, 0.0, -100.0, -1.0, -10]),
    )
    other_car_1 = FixedVelocityCar(
        world,
        np.array([0.1, -0.7, 0.8, np.pi / 2]),
        horizon=5,
        color="gray",
        opacity=1.0,
    )
    other_car_2 = FixedVelocityCar(
        world,
        np.array([0.1, -0.2, 0.8, np.pi / 2]),
        horizon=5,
        color="gray",
        opacity=1.0,
    )
    world.add_cars([our_car, other_car_1, other_car_2])
    world.reset()

    vis = CarVisualizer(world, asset_dir=default_asset_dir,)
    vis.set_main_car(index=0)
    vis.reset()
    vis.render()
    for i in range(15):
        world.step()
        print(i)
        vis.render()


if __name__ == "__main__":
    main()
