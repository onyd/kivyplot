
from kivy.core.text import Label as CoreLabel
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Ellipse, Line, Rectangle, Color, InstructionGroup, Callback, Mesh, Canvas, PushMatrix,  PopMatrix, Rotate
from kivy.graphics.tesselator import Tesselator
from kivy.properties import NumericProperty, ListProperty, ObjectProperty, BooleanProperty, StringProperty, ColorProperty, ReferenceListProperty, OptionProperty
from kivy.uix.widget import Widget
import numpy as np
from kivy.graphics.fbo import Fbo
from kivy.metrics import sp
from kivy.event import EventDispatcher
from itertools import chain
import numpy as np
import colorsys
from collections import defaultdict
from kivyplot.plot.Legend import *
from kivy.core.window import Window
from kivyplot.utils.Tooltip import Tooltip
from kivy.clock import Clock
from matplotlib import cm
from scipy.stats import gaussian_kde
import functools

"""ID counter for new graphic element"""
ID = 1


def numpyfy(func):
    @functools.wraps(func)
    def wrapper_numpyfy(self, points, *args):
        return func(self, points=np.array(points, dtype='float32'), *args)
    return wrapper_numpyfy


class GraphicElement(EventDispatcher):
    color = ColorProperty()

    def __init__(self, mapping_fn, **kwargs) -> None:
        super().__init__(**kwargs)
        self.instructions = InstructionGroup()
        self.mapping_fn = mapping_fn

        self.setup()

    def setup(self, update=False):
        self.instructions.add(Color(rgba=self.color))
        self.bind(color=self._update_color)

        self.build_instructions(self.instructions)
        if not update:
            self._update()

    def _update_color(self, instance, value):
        self.instructions.children[0].rgba = value

    def _update(self, *args):
        self.instructions.clear()
        self.setup(update=True)

    def build_instructions(self, instruction_group):
        raise NotImplementedError

    def get_instructions(self):
        return self.instructions


class LineElelement(GraphicElement):
    step = NumericProperty()

    xmin = NumericProperty()
    xmax = NumericProperty()
    ymin = NumericProperty()
    ymax = NumericProperty()
    bounding_box = ReferenceListProperty(xmin, xmax, ymin, ymax)

    def __init__(self, mapping_fn, **kwargs) -> None:
        super().__init__(mapping_fn, **kwargs)
        self.bind(bounding_box=self._update,
                  step=self._update)


class HLineElement(LineElelement):
    def build_instructions(self, instruction_group):
        y_right_ticks = np.arange(0, self.ymax, self.step)
        y_left_ticks = np.arange(0, self.ymin, -self.step)
        for y in chain(y_left_ticks, y_right_ticks):
            vx1, vy1 = self.mapping_fn([self.xmin, y])
            vx2, vy2 = self.mapping_fn([self.xmax, y])
            instruction_group.add(
                Line(points=[vx1, vy1, vx2, vy2], width=1))


class VLineElement(LineElelement):
    def build_instructions(self, instruction_group):
        x_right_ticks = np.arange(0, self.xmax, self.step)
        x_left_ticks = np.arange(0, self.xmin, -self.step)
        for x in chain(x_left_ticks, x_right_ticks):
            vx1, vy1 = self.mapping_fn([x, self.ymin])
            vx2, vy2 = self.mapping_fn([x, self.ymax])
            instruction_group.add(
                Line(points=[vx1, vy1, vx2, vy2], width=1))


class AxisElement(GraphicElement):
    label = StringProperty(None, allownone=True)
    label_font_size = NumericProperty(24)
    tick_size = NumericProperty(sp(6))
    tick_label_font_size = NumericProperty(12)

    step = NumericProperty(0.2)

    mn = NumericProperty(-1)
    mx = NumericProperty(1)
    interval = ReferenceListProperty(mn, mx)

    axis_mapping = ObjectProperty(lambda x: x)

    show_opposite_line = BooleanProperty(False)

    graph = ObjectProperty()

    def __init__(self, mapping_fn, **kwargs) -> None:
        self.tick_instructions = InstructionGroup()

        super().__init__(mapping_fn, **kwargs)
        self.bind(step=self._update,
                  interval=self._update,
                  show_opposite_line=self._update)
        self.setup_axis()

    def _update(self, *args):
        self.setup_axis()
        super()._update(*args)

    def setup_axis(self):
        raise NotImplementedError


class XAxisElement(AxisElement):
    ymin = NumericProperty()
    ymax = NumericProperty()

    width = NumericProperty(100)

    def __init__(self, mapping_fn, **kwargs) -> None:
        super().__init__(mapping_fn, **kwargs)
        self.bind(width=self._update,
                  ymin=self._update,
                  ymax=self._update)

    def setup_axis(self):
        self.tick_instructions.clear()
        margin = 0
        # Ticks
        x_right_ticks = np.arange(self.step, self.mx, self.step)
        x_left_ticks = np.arange(0, self.mn, -self.step)
        for x in chain(x_left_ticks, x_right_ticks):
            vx, vy = self.mapping_fn([x, self.ymin])
            self.tick_instructions.add(
                Line(points=[vx, vy+self.tick_size/2, vx, vy-self.tick_size/2], width=1))
            tick_label = CoreLabel(
                text=f"{self.axis_mapping(round(x, 2))}", font_size=self.tick_label_font_size)
            tick_label.refresh()
            texture = tick_label.texture

            # Update margin
            if margin < texture.height+2*self.tick_size:
                margin = texture.height+2*self.tick_size

            self.tick_instructions.add(Rectangle(texture=texture, pos=[
                vx-texture.width/2, vy-texture.height-self.tick_size], size=texture.size))

        # Label
        if self.label is not None:
            vx, vy = self.mapping_fn([self.mx / 2, self.ymin])
            label = CoreLabel(
                text=f"{self.label}", font_size=self.label_font_size)
            label.refresh()
            texture = label.texture
            margin += texture.height+2*self.tick_size

            self.tick_instructions.add(Rectangle(texture=texture, pos=[
                vx-texture.width/2, vy-texture.height-2*self.tick_size], size=texture.size))

        self.graph.y_margin = margin


class YAxisElement(AxisElement):
    xmin = NumericProperty()
    xmax = NumericProperty()

    height = NumericProperty(100)

    def __init__(self, mapping_fn, **kwargs) -> None:
        super().__init__(mapping_fn, **kwargs)
        self.bind(height=self._update,
                  xmin=self._update,
                  xmax=self._update)

    def setup_axis(self):
        self.tick_instructions.clear()
        margin = 0
        # Ticks
        y_right_ticks = np.arange(self.step, self.mx, self.step)
        y_left_ticks = np.arange(0, self.mn, -self.step)
        for y in chain(y_left_ticks, y_right_ticks):
            vx, vy = self.mapping_fn([self.xmin, y])
            self.tick_instructions.add(
                Line(points=[vx+self.tick_size/2, vy, vx-self.tick_size/2, vy], width=1))
            tick_label = CoreLabel(
                text=f"{self.axis_mapping(round(y, 2))}", font_size=self.tick_label_font_size)
            tick_label.refresh()
            texture = tick_label.texture

            # Update margin
            if margin < texture.width+2*self.tick_size:
                margin = texture.width+2*self.tick_size

            self.tick_instructions.add(Rectangle(texture=texture, pos=[
                vx-texture.width-self.tick_size, vy-texture.height/2], size=texture.size))

        # Label
        if self.label is not None:
            vx, vy = self.mapping_fn([self.xmin, self.mx / 2])
            label = CoreLabel(
                text=f"{self.label}", font_size=self.label_font_size)
            label.refresh()
            texture = label.texture
            margin += texture.height+2*self.tick_size

            self.tick_instructions.add(PushMatrix())
            self.tick_instructions.add(
                Rotate(origin=[vx-texture.height/2-2*self.tick_size, vy], angle=90))
            self.tick_instructions.add(
                Rectangle(texture=texture, pos=[
                    vx-texture.height-2*self.tick_size, vy-texture.height/2], size=texture.size))
            self.tick_instructions.add(PopMatrix())

        self.graph.x_margin = margin


class XArrowAxisElement(XAxisElement):
    def build_instructions(self, instruction_group):
        # line
        x_axis_points = self.mapping_fn([
            (self.mn, self.ymin), (self.mx, self.ymin)])
        self.x_axis_line = Line(
            points=[*x_axis_points[0], *x_axis_points[1]], width=1)
        instruction_group.add(self.x_axis_line)

        # arrow
        arrow_x, arrow_y = self.mapping_fn([self.mx, self.ymin])
        instruction_group.add(
            Line(points=[arrow_x-self.tick_size, arrow_y+self.tick_size, arrow_x, arrow_y, arrow_x-self.tick_size, arrow_y-self.tick_size], width=1))

        instruction_group.add(self.tick_instructions)


class YArrowAxisElement(YAxisElement):
    def build_instructions(self, instruction_group):
        # line
        y_axis_points = self.mapping_fn([
            (self.xmin, self.mn), (self.xmin, self.mx)])
        instruction_group.add(
            Line(points=[*y_axis_points[0], *y_axis_points[1]], width=1))

        # arrow
        arrow_x, arrow_y = self.mapping_fn([self.xmin, self.mx])
        instruction_group.add(
            Line(points=[arrow_x+self.tick_size, arrow_y-self.tick_size, arrow_x, arrow_y, arrow_x-self.tick_size, arrow_y-self.tick_size], width=1))

        instruction_group.add(self.tick_instructions)


class XBoxAxisElement(XAxisElement):
    def build_instructions(self, instruction_group):
        # line
        x_axis_points = self.mapping_fn([
            (self.mn, self.ymin), (self.mx, self.ymin)])
        self.x_axis_line = Line(
            points=[*x_axis_points[0], *x_axis_points[1]], width=1)
        instruction_group.add(self.x_axis_line)

        if self.show_opposite_line:
            x_axis_points = self.mapping_fn([
                (self.mn, self.ymax), (self.mx, self.ymax)])
            self.x_axis_line = Line(
                points=[*x_axis_points[0], *x_axis_points[1]], width=1)
            instruction_group.add(self.x_axis_line)

        instruction_group.add(self.tick_instructions)


class YBoxAxisElement(YAxisElement):
    def build_instructions(self, instruction_group):
        # line
        y_axis_points = self.mapping_fn([
            (self.xmin, self.mn), (self.xmin, self.mx)])
        instruction_group.add(
            Line(points=[*y_axis_points[0], *y_axis_points[1]], width=1))

        if self.show_opposite_line:
            y_axis_points = self.mapping_fn([
                (self.xmax, self.mn), (self.xmax, self.mx)])
            instruction_group.add(
                Line(points=[*y_axis_points[0], *y_axis_points[1]], width=1))

        instruction_group.add(self.tick_instructions)


class Plot2DElement(GraphicElement):
    """Base class for graphic element in the Graph2D widget"""
    tooltip_text = StringProperty()

    def __init__(self, mapping_fn, **kwargs) -> None:
        global ID
        self.id = ID
        ID += 1

        self.picking_instructions = InstructionGroup()

        super().__init__(mapping_fn, **kwargs)

    def setup(self, update=False):
        self.picking_instructions.add(
            Color(rgb=tuple(x / 255 for x in self.get_cid())))
        self.build_instructions(self.picking_instructions)

        super().setup(update)

    def _update(self, *args):
        self.instructions.clear()
        self.picking_instructions.clear()
        self.setup(update=True)

    def get_cid(self):
        """Convert the id to a color tuple used for picking"""
        r = self.id & 255
        g = (self.id & (255 << 8)) >> 8
        b = (self.id & (255 << 16)) >> 16
        return r, g, b

    def get_picking_instructions(self):
        return self.picking_instructions


class Dot(Plot2DElement):
    pos = ListProperty()
    radius = NumericProperty()

    def _update(self, *args):
        p = self.mapping_fn(self.pos)
        pos = [p[0]-self.radius, p[1]-self.radius]
        size = [self.radius*2, self.radius*2]

        self.instructions.children[2].pos = pos
        self.instructions.children[2].size = size

        self.picking_instructions.children[2].pos = pos
        self.picking_instructions.children[2].size = size

    def build_instructions(self, instruction_group):
        instruction_group.add(Ellipse())
        self.bind(pos=self._update, radius=self._update)


class BreakLine(Plot2DElement):
    points = ListProperty()
    width = NumericProperty()
    dash_length = NumericProperty(1)
    dash_offset = NumericProperty(0)

    def _update(self, *args):
        points = []
        for p in self.points:
            points.extend(self.mapping_fn(p))
        self.instructions.children[2].points = points
        self.instructions.children[2].width = self.width
        self.instructions.children[2].dash_length = self.dash_length
        self.instructions.children[2].dash_offset = self.dash_offset

        self.picking_instructions.children[2].points = points
        self.picking_instructions.children[2].width = self.width

    def build_instructions(self, instruction_group):
        instruction_group.add(Line())
        self.bind(points=self._update,
                  width=self._update,
                  dash_length=self._update,
                  dash_offset=self._update)


class HRange(Plot2DElement):
    start = NumericProperty()
    end = NumericProperty()
    y = NumericProperty()
    tick_height = NumericProperty(sp(24))
    width = NumericProperty()

    def build_instructions(self, instruction_group):
        (start_x, start_y), (end_x, end_y) = self.mapping_fn(
            [(self.start, self.y), (self.end, self.y)])
        self.start_tick = Line(points=[start_x, start_y-self.tick_height/2,
                               start_x, start_y+self.tick_height/2],
                               width=self.width)
        self.range = Line(points=[start_x, start_y, end_x, end_y],
                          width=self.width)
        self.end_tick = Line(points=[end_x, start_y-self.tick_height/2, end_x, end_y+self.tick_height/2],
                             width=self.width)
        instruction_group.add(self.start_tick)
        instruction_group.add(self.range)
        instruction_group.add(self.end_tick)


class Box(Plot2DElement):
    pos = ListProperty()
    size = ListProperty()

    def _update(self, *args):
        (x1, y1), (x2, y2) = self.mapping_fn(
            [self.pos, (self.pos[0]+self.size[0], self.pos[1]+self.size[1])])
        pos = [x1, y1]
        size = [x2-x1, y2-y1]

        self.instructions.children[2].pos = pos
        self.instructions.children[2].size = size

        self.picking_instructions.children[2].pos = pos
        self.picking_instructions.children[2].size = size

    def build_instructions(self, instruction_group):
        instruction_group.add(Rectangle())
        self.bind(pos=self._update,
                  size=self._update)


class Polygon(Plot2DElement):
    points = ListProperty()

    def build_instructions(self, instruction_group):
        vertices = []
        for p in self.points:
            vertices.extend(self.mapping_fn(p))
        tess = Tesselator()
        tess.add_contour(vertices)
        tess.tesselate()

        polygons = [Mesh(vertices=v, indices=i,
                         mode='triangle_fan') for v, i in tess.meshes]
        for polygon in polygons:
            instruction_group.add(polygon)


class Bar(Plot2DElement):
    pos = ListProperty()
    width = NumericProperty()

    def _update(self, *args):
        d = self.width/2
        (x1, y1), (x2, y2) = self.mapping_fn(
            [(self.pos[0]-d, 0), (self.pos[0]+d, self.pos[1])])
        pos = [x1, y1]
        size = [x2-x1, y2-y1]

        self.instructions.children[2].pos = pos
        self.instructions.children[2].size = size

        self.picking_instructions.children[2].pos = pos
        self.picking_instructions.children[2].size = size

    def build_instructions(self, instruction_group):
        instruction_group.add(Rectangle())
        self.bind(pos=self._update,
                  width=self._update)


class Graph2D(Widget):
    """Graph widget which draws axis and given data"""
    x_margin = NumericProperty(sp(38))
    y_margin = NumericProperty(sp(38))
    margin = ReferenceListProperty(x_margin, y_margin)

    stepx = NumericProperty(0.2)
    stepy = NumericProperty(0.2)
    step = ReferenceListProperty(stepx, stepy)

    xmin = NumericProperty(-1)
    xmax = NumericProperty(1)
    ymin = NumericProperty(-1)
    ymax = NumericProperty(1)
    bounding_box = ReferenceListProperty(xmin, ymin, xmax, ymax)

    x_axis_style = OptionProperty('arrow', options=['arrow', 'box'])
    y_axis_style = OptionProperty('arrow', options=['arrow', 'box'])
    axis_style = ReferenceListProperty(x_axis_style, y_axis_style)

    x_axis_mapping = ObjectProperty(lambda x: x)
    y_axis_mapping = ObjectProperty(lambda x: x)
    axis_mapping = ReferenceListProperty(x_axis_mapping, y_axis_mapping)

    x_axis_label = StringProperty()
    y_axis_label = StringProperty()
    axis_label = ReferenceListProperty(x_axis_label, y_axis_label)

    show_x_axis = BooleanProperty(True)
    show_y_axis = BooleanProperty(True)
    show_axis = ReferenceListProperty(show_x_axis, show_y_axis)

    show_v_lines = BooleanProperty(False)
    show_h_lines = BooleanProperty(False)
    show_lines = ReferenceListProperty(show_v_lines, show_h_lines)

    elements = ListProperty([])

    def __init__(self, show_hover_data=True, **kwargs) -> None:
        self.canvas = Canvas()
        self.picking = show_hover_data
        if self.picking:
            self.cid_to_element = {}
        self.x_axis = None
        self.y_axis = None
        self.setup()

        super().__init__(**kwargs)

        # Setup tooltip
        if show_hover_data:
            self._tooltip = None
            self.hovered = None
            Window.bind(mouse_pos=self.on_mouse_pos)

            self.closed = True

    def _update_tooltip(self, x, y, row=1, col=1):
        e = self.get_element_at(x, y)
        if e is None:
            if not self.closed:
                self.close_tooltip()
            return False
        if e is self.hovered:
            self._tooltip.pos = (x, y)
            return False

        # The element has changed
        if not self.closed:
            self.close_tooltip()
        self.hovered = e
        self._tooltip = Tooltip(text=e.tooltip_text)
        return True

    def on_mouse_pos(self, *args):
        if not self.get_root_window():
            return
        pos = args[1]
        if not self._update_tooltip(*pos):
            return

        # close if it's opened
        if self.collide_point(*self.to_parent(*self.to_widget(*pos))):
            if self.closed:
                self.display_tooltip()

    def close_tooltip(self, *args):
        Window.remove_widget(self._tooltip)
        self.hovered = None
        self.closed = True

    def display_tooltip(self, *args):
        Window.add_widget(self._tooltip)
        self.closed = False

    @numpyfy
    def to_viewport_space(self, points):
        pmin = np.array([self.xmin, self.ymin])
        pmax = np.array([self.xmax, self.ymax])
        size = np.array([self.width, self.height])
        m = np.array([self.x_margin, self.y_margin])
        if len(points.shape) in (1, 2):
            return (points-pmin) / (pmax-pmin) * \
                (size - 2*m) + m
        else:
            raise ValueError(
                "To convert points into viewport space, points must be of the form [x, y] or [[x, y], ...]")

    def update(self, *args):
        if self.show_x_axis:
            self.x_axis._update()
        if self.show_y_axis:
            self.y_axis._update()
        if self.show_h_lines:
            self.h_lines._update()
        if self.show_v_lines:
            self.v_lines._update()

        self.update_data()

    def update_data(self, *args):
        for element in self.elements:
            element._update()

    def on_margin(self, *args):
        self.update()

    def on_bounding_box(self, *args):
        # Update x axis
        if self.show_x_axis:
            self.x_axis.mn = self.xmin
            self.x_axis.mx = self.xmax
            self.x_axis.ymin = self.ymin
            self.x_axis.ymax = self.ymax

        # Update y axis
        if self.show_y_axis:
            self.y_axis.mn = self.ymin
            self.y_axis.mx = self.ymax
            self.y_axis.xmin = self.xmin
            self.y_axis.xmax = self.xmax

        # Update horizontal lines
        if self.show_h_lines:
            self.h_lines.bounding_box = self.bounding_box

        # Update vertical lines
        if self.show_v_lines:
            self.v_lines.bounding_box = self.bounding_box

    def on_step(self, *args):
        # Update x axis
        if self.show_x_axis:
            self.x_axis.step = self.stepx

        # Update y axis
        if self.show_y_axis:
            self.y_axis.step = self.stepy

        # Update horizontal lines
        if self.show_h_lines:
            self.h_lines.step = self.stepy

        # Update vertical lines
        if self.show_v_lines:
            self.v_lines.step = self.stepx

    def on_axis_mapping(self, *args):
        if self.x_axis:
            self.x_axis.axis_mapping = self.x_axis_mapping
        if self.y_axis:
            self.y_axis.axis_mapping = self.y_axis_mapping

    def on_x_axis_label(self, *args):
        if self.show_x_axis:
            self.x_axis.label = self.x_axis_label

    def on_y_axis_label(self, *args):
        if self.show_y_axis:
            self.y_axis.label = self.y_axis_label

    def on_show_x_axis(self, *args):
        if self.show_x_axis:
            self.on_x_axis_style()
        else:
            self.x_axes_instructions.clear()

    def on_show_y_axis(self, *args):
        if self.show_y_axis:
            self.on_y_axis_style()
        else:
            self.y_axes_instructions.clear()

    def on_x_axis_style(self, *args):
        self.x_axes_instructions.clear()

        if self.show_x_axis:
            if self.x_axis_style == 'arrow':
                self.x_axis = XArrowAxisElement(
                    self.to_viewport_space,
                    graph=self,
                    ymin=self.ymin,
                    mn=self.xmin,
                    mx=self.xmax,
                    step=self.stepx,
                    color=(0, 0, 0))
            elif self.x_axis_style == 'box':
                self.x_axis = XBoxAxisElement(
                    self.to_viewport_space,
                    graph=self,
                    ymin=self.ymin,
                    ymax=self.ymax,
                    mn=self.xmin, mx=self.xmax,
                    step=self.stepx,
                    color=(0, 0, 0))
                if self.y_axis is not None:
                    self.y_axis.show_opposite_line = self.y_axis_style == 'box'
                self.x_axis.show_opposite_line = self.y_axis_style == 'box'

            self.x_axes_instructions.add(self.x_axis.get_instructions())
        else:
            self.y_margin = 0

    def on_y_axis_style(self, *args):
        self.y_axes_instructions.clear()

        if self.show_y_axis:
            if self.y_axis_style == 'arrow':
                self.y_axis = YArrowAxisElement(
                    self.to_viewport_space,
                    graph=self,
                    xmin=self.xmin,
                    mn=self.ymin, mx=self.ymax,
                    step=self.stepy,
                    color=(0, 0, 0))
            elif self.y_axis_style == 'box':
                self.y_axis = YBoxAxisElement(
                    self.to_viewport_space,
                    graph=self,
                    xmin=self.xmin,
                    xmax=self.xmax,
                    mn=self.ymin, mx=self.ymax,
                    step=self.stepy,
                    color=(0, 0, 0))
                if self.x_axis is not None:
                    self.x_axis.show_opposite_line = self.x_axis_style == 'box'
                self.y_axis.show_opposite_line = self.x_axis_style == 'box'

            self.y_axes_instructions.add(self.y_axis.get_instructions())
        else:
            self.x_margin = 0

    def on_show_v_lines(self, *args):
        if self.show_v_lines:
            self.v_lines = VLineElement(
                self.to_viewport_space, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, step=self.stepy, color=(0, 0, 0))

            self.v_lines_instructions.add(self.v_lines.get_instructions())
        else:
            self.v_lines_instructions.clear()

    def on_show_h_lines(self, *args):
        if self.show_h_lines:
            self.h_lines = HLineElement(
                self.to_viewport_space, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, step=self.stepy, color=(0, 0, 0))

            self.h_lines_instructions.add(self.h_lines.get_instructions())
        else:
            self.h_lines_instructions.clear()

    def on_size(self, *args):
        self._adjust_aspect()
        self.box.rectangle = [1, 1, self.width-1, self.height-1]
        self.update()

    def get_element_at(self, x, y):
        assert(self.picking)
        try:
            return self.cid_to_element[tuple(self.picking_fbo.get_pixel_color(x, y))[:3]]
        except KeyError:
            return None

    def setup(self):
        with self.canvas:
            Color(rgb=(1, 1, 1))
            self._viewport = Rectangle(size=self.size, pos=self.pos)
            Color(rgb=(0, 0, 0))
            self.box = Line(
                rectangle=[1, 1, self.width-1, self.height-1], width=1)
            self.fbo = Fbo(size=self.size,
                           with_depthbuffer=False, compute_normal_mat=False,
                           clear_color=(1., 1., 1., 1.))
            if self.picking:
                self.picking_fbo = Fbo(size=self.size,
                                       with_depthbuffer=False, compute_normal_mat=False,
                                       clear_color=(0., 0., 0., 1.))

        with self.fbo:
            Callback(self._clear_buffer)
            self.v_lines_instructions = InstructionGroup()
            self.h_lines_instructions = InstructionGroup()
            self.x_axes_instructions = InstructionGroup()
            self.y_axes_instructions = InstructionGroup()
            self.instructions = InstructionGroup()

        if self.picking:
            with self.picking_fbo:
                Callback(self._clear_picking_buffer)
                self.picking_instructions = InstructionGroup()

        self.on_x_axis_style()
        self.on_y_axis_style()
        self.on_show_v_lines()
        self.on_show_h_lines()
        self._viewport.texture = self.fbo.texture

    def _clear_buffer(self, *args):
        self.fbo.clear_buffer()

    def _clear_picking_buffer(self, *args):
        self.picking_fbo.clear_buffer()

    def add_points(self, points, color=(0, 0, 1, 1), radius=5, tooltip_text=''):
        for p in points:
            dot = Dot(self.to_viewport_space, pos=p, radius=radius, color=color,
                      tooltip_text=tooltip_text)
            if self.picking:
                self.cid_to_element[dot.get_cid()] = dot
                self.picking_instructions.add(dot.get_picking_instructions())
            self.instructions.add(dot.get_instructions())
            self.elements.append(dot)

    def add_breakline(self, points, color=(0, 0, 1, 1), width=1, tooltip_text='', dash_length=1, dash_offset=0):
        breakline = BreakLine(self.to_viewport_space,
                              points=points, color=color,
                              width=width, tooltip_text=tooltip_text, dash_length=dash_length, dash_offset=dash_offset)
        if self.picking:
            self.cid_to_element[breakline.get_cid()] = breakline
            self.picking_instructions.add(
                breakline.get_picking_instructions())
        self.instructions.add(breakline.get_instructions())
        self.elements.append(breakline)

    def add_polygon(self, points, color=(0, 0, 1, 1), tooltip_text=''):
        poly = Polygon(self.to_viewport_space,
                       points=points,
                       color=color,
                       tooltip_text=tooltip_text)
        if self.picking:
            self.cid_to_element[poly.get_cid()] = poly
            self.picking_instructions.add(poly.get_picking_instructions())
        self.instructions.add(poly.get_instructions())
        self.elements.append(poly)

    def add_hrange(self, y, start, end, width=1, color=(0, 0, 0, 1), tooltip_text=''):
        r = HRange(self.to_viewport_space,
                   start=start,
                   end=end,
                   y=y,
                   width=width,
                   color=color,
                   tooltip_text=tooltip_text)
        if self.picking:
            self.cid_to_element[r.get_cid()] = r
            self.picking_instructions.add(r.get_picking_instructions())
        self.instructions.add(r.get_instructions())
        self.elements.append(r)

    def add_box(self, pos, size, color=(0, 0, 1, 1), tooltip_text=''):
        box = Box(self.to_viewport_space,
                  pos=pos,
                  size=size,
                  color=color,
                  tooltip_text=tooltip_text)

        if self.picking:
            self.cid_to_element[box.get_cid()] = box
            self.picking_instructions.add(box.get_picking_instructions())
        self.instructions.add(box.get_instructions())
        self.elements.append(box)

    def add_bars(self, points, color=(0, 0, 1, 1), width=1, tooltip_text=''):
        for p in points:
            e = Bar(self.to_viewport_space,
                    pos=p, width=width, color=color, tooltip_text=tooltip_text)
            if self.picking:
                self.cid_to_element[e.get_cid()] = e
                self.picking_instructions.add(e.get_picking_instructions())
            self.instructions.add(e.get_instructions())
            self.elements.append(e)

    def clear(self):
        self.data.clear()
        self.instructions.clear()
        if self.picking:
            self.picking_instructions.clear()
            self.cid_to_element.clear()
        self.update_data()

    def _adjust_aspect(self, *args):
        if self.width > 0 and self.height > 0:
            self.fbo.size = self.size
            self._viewport.size = self.size
            self._viewport.pos = self.pos
            self._viewport.texture = self.fbo.texture

            if self.picking:
                self.picking_fbo.size = self.size


class Plot2D(GridLayout):
    do_show_legend = BooleanProperty(False)

    def __init__(self, rows=1, cols=1, **kwargs):
        super().__init__(rows=rows,  cols=cols, **kwargs)
        # Setup widget
        self.spacing = sp(12)
        self.legend = None
        self.labels = []
        self.setup_graphs()
        self.bind(pos=self.update, size=self.update,
                  rows=self.setup_graphs, cols=self.setup_graphs)

    def setup_graphs(self, *args):
        self.clear_widgets()
        for _ in range(self.cols):
            for _ in range(self.rows):
                self.add_widget(Graph2D())

    def get_graph(self, i, j):
        return self.children[len(self.children)-(i*self.cols+j+1)]

    def on_do_show_legend(self, *args):
        if self.legend is None:
            self.legend = Legend()
            self.add_widget(self.legend)
        else:
            self.remove_widget(self.legend)

    def update_legend(self, *args):
        if self.do_show_legend:
            self.legend.clear_widgets()
            for label in self.labels:
                self.legend.add_widget(label)

    def update(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(rgb=(1, 1, 1))
            Rectangle(pos=self.pos, size=self.size)

    def scatter(self, points, color=(0, 0, 1, 1), radius=5, label=None, tooltip_text='', row=0, col=0):
        self.get_graph(row, col).add_points(points, color=color,
                                            radius=radius, tooltip_text=tooltip_text)
        if label is not None:
            self.labels.append(DotLegendItem(
                label=label, color=color, radius=radius))
        self.update_legend()

    def plot(self, points, color=(0, 0, 1, 1), width=2, label=None, filled=False, opacity=0.5, tooltip_text='', row=0, col=0):
        if filled:
            self.get_graph(row, col).add_polygon([(points[0][0], 0)]+points+[(points[-1][0], 0)], color=(*color[: 3], opacity),
                                                 tooltip_text=tooltip_text)
        self.get_graph(row, col).add_breakline(points, color=color,
                                               width=width, tooltip_text=tooltip_text)
        if label is not None:
            self.labels.append(CurveLegendItem(
                label=label, color=color, line_width=width))
        self.update_legend()
        self.get_graph(row, col).update_data()

    def violin(self, data, color=(0, 0.6, 0.8, 1), show_mean=True, show_median=True, bandwidth=None, cut=1, splitted=True, row=0, col=0):
        mx = float(data.max().max())
        mn = float(data.min().min())

        margin = 0
        if splitted:
            self.rows = len(data.columns)
            self.cols = 1

        for i, c in enumerate(data):
            d = data[c]
            if splitted:
                y = 1
                row = i
                col = 0
            else:
                y = len(data.columns)-i

            d_mn, d_mx = float(d.min()), float(d.max())

            kde = gaussian_kde(d, bandwidth)
            bw = kde.factor * d.std()

            # Update margin due to cut
            if margin < bw * cut:
                margin = float(bw * cut)

            # Compute the smooth density
            X = np.linspace(d_mn - bw * cut,
                            d_mx + bw * cut, 100)
            density = kde.evaluate(X)
            density /= density.max() * 2.5
            points_up = np.array([(X[0], y)]+[[float(x), y+ds]
                                              for x, ds in zip(X, density)]+[(X[-1], y)])
            points_down = np.array([(X[0], y)]+[[float(x), y-ds]
                                                for x, ds in zip(X, density)]+[(X[-1], y)])
            # Add violin graphics
            self.get_graph(row, col).add_breakline(
                points_up, color=(0, 0, 0, 1), width=2)
            self.get_graph(row, col).add_polygon(points_up,
                                                 color=color)

            self.get_graph(row, col).add_breakline(
                points_down, color=(0, 0, 0, 1), width=2)
            self.get_graph(row, col).add_polygon(points_down,
                                                 color=color)

            # Add quartiles range
            q = np.quantile(d, [0.25, 0.75])
            self.get_graph(row, col).add_breakline(
                points=[(d_mn, y), (d_mx, y)], color=(0, 0, 0, 1), width=2)
            self.get_graph(row, col).add_breakline(
                points=[(q[0], y), (q[1], y)], color=(0, 0, 0, 1), width=5)

            # Add descriptive statistics
            if show_mean:
                mean = float(d.mean())
                self.get_graph(row, col).add_points(
                    points=[(mean, y)],
                    color=(1, 0, 0, 1))
            if show_median:
                med = float(np.median(d))
                self.get_graph(row, col).add_points(
                    points=[(med, y)],
                    color=(0, 1, 0, 1))

            # Setup the graph properties
            if splitted:
                self.get_graph(row, col).axis_style = ('box', 'box')
                self.get_graph(row, col).xmin = mn-0.1*(mx-mn)-margin
                self.get_graph(row, col).xmax = mx+0.1*(mx-mn)+margin
                self.get_graph(row, col).ymin = 0
                self.get_graph(row, col).ymax = 2
                self.get_graph(
                    row, 0).stepx = 10**(int(np.log10(mx))-1)*len(data.columns)
                self.get_graph(row, col).stepy = 1
                self.get_graph(
                    row, 0).y_axis_mapping = lambda y, r=row: data.columns[r]
                self.get_graph(row, col).update_data()
                
        if not splitted:
            self.get_graph(row, col).axis_style = ('box', 'box')
            self.get_graph(row, col).xmin = mn-0.1*(mx-mn)-margin
            self.get_graph(row, col).xmax = mx+0.1*(mx-mn)+margin
            self.get_graph(row, col).ymin = 0
            self.get_graph(row, col).ymax = len(data.columns)+1 if not splitted else 1
            self.get_graph(
                row, 0).stepx = 10**(int(np.log10(mx))-1)*len(data.columns)
            self.get_graph(row, col).stepy = 1
            self.get_graph(
                row, 0).y_axis_mapping = lambda y: data.columns[int(y)-1]
            self.get_graph(row, col).update_data()
        self.update_legend()

    def bars(self, points, color=(0, 0, 1, 1), width=1, label=None, tooltip_text='', row=0, col=0):
        self.get_graph(row, col).add_bars(points, color=color,
                                          width=width, tooltip_text=tooltip_text)
        if label is not None:
            self.labels.append(BarLegendItem(
                label=label, color=color))
        self.update_legend()
        self.get_graph(row, col).update_data()

    def box(self, data, show_mean=True, show_median=True, row=0, col=0):
        mx = float(data.max().max())
        mn = float(data.min().min())

        self.get_graph(row, col).xmin = mn-0.1*(mx-mn)
        self.get_graph(row, col).xmax = mx+0.1*(mx-mn)
        self.get_graph(row, col).ymin = 0
        self.get_graph(row, col).ymax = len(data.columns)+1
        self.get_graph(
            row, col).stepx = 10**(int(np.log10(mx))-1)*len(data.columns)
        self.get_graph(row, col).stepy = 1
        # self.get_graph(row, col).y_axis_mapping = lambda y: names[int(y)]

        for i, col in enumerate(data):
            d = data[col]
            y = len(data.columns)-i
            self.get_graph(row, col).add_hrange(
                y=y, start=float(d.min()), end=float(d.max()))

            box_height = 0.75
            q = np.quantile(d, [0.25, 0.75])
            self.get_graph(row, col).add_box(
                pos=(q[0], y-box_height/2), size=(q[1]-q[0], box_height))
            if show_mean:
                mean = float(d.mean())
                self.get_graph(row, col).add_breakline(
                    points=[(mean, y-box_height/2), (mean, y+box_height/2)],
                    color=(1, 0, 0, 1))
            if show_median:
                med = float(np.median(d))
                self.get_graph(row, col).add_breakline(
                    points=[(med, y-box_height/2), (med, y+box_height/2)],
                    color=(0, 1, 0, 1))
        self.update_legend()
        self.get_graph(row, col).update_data()

    def heatmap(self, matrix, x_labels=None, y_labels=None, cmap='viridis', row=0, col=0):
        matrix = np.array(matrix)
        mx = matrix.max()
        mn = matrix.min()

        self.get_graph(row, col).xmin, self.get_graph(row, col).ymin = 0, 0
        self.get_graph(row, col).ymax = matrix.shape[0] + 1
        self.get_graph(row, col).xmax = matrix.shape[1] + 1
        self.get_graph(row, col).stepx = 1
        self.get_graph(row, col).stepy = 1
        if x_labels:
            self.get_graph(
                row, col).x_axis_mapping = lambda x: x_labels[int(x)-1]
        if y_labels:
            self.get_graph(
                row, col).y_axis_mapping = lambda y: y_labels[int(y)-1]

        cmap = cm.get_cmap(cmap)

        def color(x):
            return cmap((x-mn) / (mx-mn))

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                x = j+0.5
                y = matrix.shape[0]-i-0.5
                self.get_graph(row, col).add_box(pos=(x, y),
                                                 size=(1, 1),
                                                 color=color(matrix[i][j]),
                                                 tooltip_text=str(matrix[i][j]))
        self.update_legend()
        self.get_graph(row, col).update_data()

    def hist(self, labels, row=0, col=0):
        h = defaultdict(int)
        for label in labels:
            h[label] += 1
        my = max(h.values())
        X = list(h.keys())
        labels_to_index = {l: i for i, l in enumerate(X, start=1)}

        HSV = {l: (i/len(h), 0.5, 0.9) for i, l in enumerate(h)}
        colors = {k: (*colorsys.hsv_to_rgb(*v), 1) for k, v in HSV.items()}

        self.get_graph(row, col).xmin = 0
        self.get_graph(row, col).xmax = len(h)+1
        self.get_graph(row, col).ymin = 0
        self.get_graph(row, col).ymax = my
        self.get_graph(row, col).stepx = 1
        self.get_graph(row, col).stepy = 10**(int(np.log10(my))-1)*len(h)
        self.get_graph(row, col).x_axis_mapping = lambda x: X[int(x)-1]

        for x, y in h.items():
            self.bars([(labels_to_index[x], y)],
                      color=colors[x], width=0.9, label=str(x))

    def clear(self):
        for j in range(self.cols):
            for i in range(self.rows):
                self.get_graph(i, j).clear()
