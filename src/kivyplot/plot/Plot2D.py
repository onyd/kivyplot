
from kivy.core.text import Label as CoreLabel
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Ellipse, Line, Rectangle, Color, InstructionGroup, Callback, Mesh, Canvas
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
    step = NumericProperty(0.2)
    axis_mapping = ObjectProperty(lambda x: x)

    def __init__(self, mapping_fn, **kwargs) -> None:
        super().__init__(mapping_fn, **kwargs)
        self.bind(step=self._update)


class XAxisElement(AxisElement):
    xmin = NumericProperty(-1)
    xmax = NumericProperty(1)
    interval = ReferenceListProperty(xmin, xmax)

    width = NumericProperty(100)

    def __init__(self, mapping_fn, **kwargs) -> None:
        super().__init__(mapping_fn, **kwargs)
        self.bind(interval=self._update,
                  width=self._update)


class YAxisElement(AxisElement):
    ymin = NumericProperty(-1)
    ymax = NumericProperty(1)
    interval = ReferenceListProperty(ymin, ymax)

    height = NumericProperty(100)

    def __init__(self, mapping_fn, **kwargs) -> None:
        super().__init__(mapping_fn, **kwargs)
        self.bind(interval=self._update,
                  height=self._update)


class XArrowAxisElement(XAxisElement):
    y = NumericProperty()
    arrow_size = NumericProperty(sp(6))

    def build_instructions(self, instruction_group):
        # line
        x_axis_points = self.mapping_fn([
            (self.xmin, self.y), (self.xmax, self.y)])
        self.x_axis_line = Line(
            points=[*x_axis_points[0], *x_axis_points[1]], width=1)
        instruction_group.add(self.x_axis_line)

        # arrow
        arrow_x, arrow_y = self.mapping_fn([self.xmax, self.y])
        instruction_group.add(
            Line(points=[arrow_x-self.arrow_size, arrow_y+self.arrow_size, arrow_x, arrow_y, arrow_x-self.arrow_size, arrow_y-self.arrow_size], width=1))

        # ticks
        x_right_ticks = np.arange(self.step, self.xmax, self.step)
        x_left_ticks = np.arange(0, self.xmin, -self.step)
        for x in chain(x_left_ticks, x_right_ticks):
            vx, vy = self.mapping_fn([x, self.y])
            instruction_group.add(
                Line(points=[vx, vy+self.arrow_size/2, vx, vy-self.arrow_size/2], width=1))
            tick_label = CoreLabel(
                text=f"{self.axis_mapping(round(x, 2))}", font_size=10)
            tick_label.refresh()
            texture = tick_label.texture
            instruction_group.add(Rectangle(texture=texture, pos=[
                vx-texture.width/2, vy-texture.height-self.arrow_size], size=texture.size))


class YArrowAxisElement(YAxisElement):
    x = NumericProperty()
    arrow_size = NumericProperty(sp(6))

    def build_instructions(self, instruction_group):
        # line
        y_axis_points = self.mapping_fn([
            (self.x, self.ymin), (self.x, self.ymax)])
        instruction_group.add(
            Line(points=[*y_axis_points[0], *y_axis_points[1]], width=1))

        # arrow
        arrow_x, arrow_y = self.mapping_fn([self.x, self.ymax])
        instruction_group.add(
            Line(points=[arrow_x+self.arrow_size, arrow_y-self.arrow_size, arrow_x, arrow_y, arrow_x-self.arrow_size, arrow_y-self.arrow_size], width=1))

        # ticks
        y_right_ticks = np.arange(self.step, self.ymax, self.step)
        y_left_ticks = np.arange(0, self.ymin, -self.step)
        for y in chain(y_left_ticks, y_right_ticks):
            vx, vy = self.mapping_fn([self.x, y])
            instruction_group.add(
                Line(points=[vx+self.arrow_size/2, vy, vx-self.arrow_size/2, vy], width=1))
            tick_label = CoreLabel(
                text=f"{self.axis_mapping(round(y, 2))}", font_size=10)
            tick_label.refresh()
            texture = tick_label.texture
            instruction_group.add(Rectangle(texture=texture, pos=[
                vx-texture.width-self.arrow_size, vy-texture.height/2], size=texture.size))


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
    margin = NumericProperty(sp(38))

    stepx = NumericProperty(0.2)
    stepy = NumericProperty(0.2)
    step = ReferenceListProperty(stepx, stepy)

    xmin = NumericProperty(-1)
    xmax = NumericProperty(1)
    ymin = NumericProperty(-1)
    ymax = NumericProperty(1)
    bounding_box = ReferenceListProperty(xmin, ymin, xmax, ymax)

    x_axis_style = OptionProperty('arrow', options=['arrow'])
    y_axis_style = OptionProperty('arrow', options=['arrow'])
    axis_style = ReferenceListProperty(x_axis_style, y_axis_style)

    x_axis_mapping = ObjectProperty(lambda x: x)
    y_axis_mapping = ObjectProperty(lambda x: x)
    axis_mapping = ReferenceListProperty(x_axis_mapping, y_axis_mapping)

    show_x_axis = BooleanProperty(True)
    show_y_axis = BooleanProperty(True)
    show_axis = ReferenceListProperty(show_x_axis, show_y_axis)

    show_v_lines = BooleanProperty(False)
    show_h_lines = BooleanProperty(False)
    show_lines = ReferenceListProperty(show_v_lines, show_h_lines)

    elements = ListProperty([])

    def __init__(self, picking=True, **kwargs) -> None:
        self.canvas = Canvas()
        self.picking = picking
        if self.picking:
            self.cid_to_element = {}
        self.setup()

        super().__init__(**kwargs)

    @numpyfy
    def to_viewport_space(self, points):
        pmin = np.array([self.xmin, self.ymin])
        pmax = np.array([self.xmax, self.ymax])
        size = np.array([self.width, self.height])

        if len(points.shape) in (1, 2):
            return (points-pmin) / (pmax-pmin) * \
                (size - 2*self.margin) + self.margin
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
            self.x_axis.xmin = self.xmin
            self.x_axis.xmax = self.xmax
            self.x_axis.y = self.ymin

        # Update y axis
        if self.show_y_axis:
            self.y_axis.ymin = self.ymin
            self.y_axis.ymax = self.ymax
            self.y_axis.x = self.xmin

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
        if self.show_x_axis:
            if self.x_axis_style == 'arrow':
                self.x_axis = XArrowAxisElement(
                    self.to_viewport_space, y=self.ymin, xmin=self.xmin, xmax=self.xmax, step=self.stepx, color=(0, 0, 0))

            self.x_axes_instructions.add(self.x_axis.get_instructions())
        else:
            self.x_axes_instructions.clear()

    def on_y_axis_style(self, *args):
        if self.show_y_axis:
            if self.y_axis_style == 'arrow':
                self.y_axis = YArrowAxisElement(
                    self.to_viewport_space, x=self.xmin, ymin=self.ymin, ymax=self.ymax, step=self.stepy, color=(0, 0, 0))

            self.y_axes_instructions.add(self.y_axis.get_instructions())
        else:
            self.y_axes_instructions.clear()

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


class Plot2D(BoxLayout):
    do_show_legend = BooleanProperty(False)

    def __init__(self, show_hover_data=True, **kwargs):
        super().__init__()
        # Setup widget
        self.spacing = sp(12)
        self.bind(minimum_size=self.setter('size'))
        self.graph = Graph2D(**kwargs)
        self.legend = None
        self.labels = []
        self.add_widget(self.graph)
        self.bind(pos=self.update, size=self.update)

        # Setup tooltip
        if show_hover_data:
            self._tooltip = None
            self.hovered = None
            Window.bind(mouse_pos=self.on_mouse_pos)

            self.closed = True

    def _update_tooltip(self, x, y):
        e = self.graph.get_element_at(x, y)
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

    def scatter(self, points, color=(0, 0, 1, 1), radius=5, label=None, tooltip_text=''):
        self.graph.add_points(points, color=color,
                              radius=radius, tooltip_text=tooltip_text)
        if label is not None:
            self.labels.append(DotLegendItem(
                label=label, color=color, radius=radius))
        self.update_legend()

    def plot(self, points, color=(0, 0, 1, 1), width=2, label=None, filled=False, opacity=0.5, tooltip_text=''):
        if filled:
            self.graph.add_polygon([(points[0][0], 0)]+points+[(points[-1][0], 0)], color=(*color[: 3], opacity),
                                   tooltip_text=tooltip_text)
        self.graph.add_breakline(points, color=color,
                                 width=width, tooltip_text=tooltip_text)
        if label is not None:
            self.labels.append(CurveLegendItem(
                label=label, color=color, line_width=width))
        self.update_legend()
        self.graph.update_data()

    def violin(self, names, data, color=(0, 0.6, 0.8, 1), show_mean=True, show_median=True, bandwidth=None, cut=1):
        data = np.array(data)
        mx = float(data.max())
        mn = float(data.min())

        margin = 0

        for i, d in enumerate(data):
            y = len(data)-i
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
            self.graph.add_breakline(points_up, color=(0, 0, 0, 1), width=2)
            self.graph.add_polygon(points_up,
                                   color=color)

            self.graph.add_breakline(points_down, color=(0, 0, 0, 1), width=2)
            self.graph.add_polygon(points_down,
                                   color=color)

            # Add quartiles range
            q = np.quantile(d, [0.25, 0.75])
            self.graph.add_breakline(
                points=[(d_mn, y), (d_mx, y)], color=(0, 0, 0, 1), width=2)
            self.graph.add_breakline(
                points=[(q[0], y), (q[1], y)], color=(0, 0, 0, 1), width=5)

            # Add descriptive statistics
            if show_mean:
                mean = float(d.mean())
                self.graph.add_points(
                    points=[(mean, y)],
                    color=(1, 0, 0, 1))
            if show_median:
                med = float(np.median(d))
                self.graph.add_points(
                    points=[(med, y)],
                    color=(0, 1, 0, 1))

        # Setup the graph properties
        self.graph.xmin = mn-0.1*(mx-mn)-margin
        self.graph.xmax = mx+0.1*(mx-mn)+margin
        self.graph.ymin = 0
        self.graph.ymax = len(data)+1
        self.graph.stepx = 10**(int(np.log10(mx))-1)*len(data)
        self.graph.stepy = 1

        self.update_legend()
        self.graph.update_data()

    def bars(self, points, color=(0, 0, 1, 1), width=1, label=None, tooltip_text=''):
        self.graph.add_bars(points, color=color,
                            width=width, tooltip_text=tooltip_text)
        if label is not None:
            self.labels.append(BarLegendItem(
                label=label, color=color))
        self.update_legend()
        self.graph.update_data()

    def box(self, names, data, show_mean=True, show_median=True):
        data = np.array(data)
        mx = float(data.max())
        mn = float(data.min())

        self.graph.xmin = mn-0.1*(mx-mn)
        self.graph.xmax = mx+0.1*(mx-mn)
        self.graph.ymin = 0
        self.graph.ymax = len(data)+1
        self.graph.stepx = 10**(int(np.log10(mx))-1)*len(data)
        self.graph.stepy = 1
        # self.graph.y_axis_mapping = lambda y: names[int(y)]

        for i, d in enumerate(data):
            y = len(data)-i
            self.graph.add_hrange(
                y=y, start=float(d.min()), end=float(d.max()))

            box_height = 0.75
            q = np.quantile(d, [0.25, 0.75])
            self.graph.add_box(
                pos=(q[0], y-box_height/2), size=(q[1]-q[0], box_height))
            if show_mean:
                mean = float(d.mean())
                self.graph.add_breakline(
                    points=[(mean, y-box_height/2), (mean, y+box_height/2)],
                    color=(1, 0, 0, 1))
            if show_median:
                med = float(np.median(d))
                self.graph.add_breakline(
                    points=[(med, y-box_height/2), (med, y+box_height/2)],
                    color=(0, 1, 0, 1))
        self.update_legend()
        self.graph.update_data()

    def heatmap(self, matrix, x_labels=None, y_labels=None, cmap='viridis'):
        matrix = np.array(matrix)
        mx = matrix.max()
        mn = matrix.min()

        self.graph.xmin, self.graph.ymin = 0, 0
        self.graph.ymax = matrix.shape[0] + 1
        self.graph.xmax = matrix.shape[1] + 1
        self.graph.stepx = 1
        self.graph.stepy = 1
        if x_labels:
            self.graph.x_axis_mapping = lambda x: x_labels[int(x)-1]
        if y_labels:
            self.graph.y_axis_mapping = lambda y: y_labels[int(y)-1]

        cmap = cm.get_cmap(cmap)

        def color(x):
            return cmap((x-mn) / (mx-mn))

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                x = j+0.5
                y = matrix.shape[0]-i-0.5
                self.graph.add_box(pos=(x, y),
                                   size=(1, 1),
                                   color=color(matrix[i][j]),
                                   tooltip_text=str(matrix[i][j]))
        self.update_legend()
        self.graph.update_data()

    def hist(self, labels):
        h = defaultdict(int)
        for label in labels:
            h[label] += 1
        my = max(h.values())
        X = list(h.keys())
        labels_to_index = {l: i for i, l in enumerate(X, start=1)}

        HSV = {l: (i/len(h), 0.5, 0.9) for i, l in enumerate(h)}
        colors = {k: (*colorsys.hsv_to_rgb(*v), 1) for k, v in HSV.items()}

        self.graph.xmin = 0
        self.graph.xmax = len(h)+1
        self.graph.ymin = 0
        self.graph.ymax = my
        self.graph.stepx = 1
        self.graph.stepy = 10**(int(np.log10(my))-1)*len(h)
        self.graph.x_axis_mapping = lambda x: X[int(x)-1]

        for x, y in h.items():
            self.bars([(labels_to_index[x], y)],
                      color=colors[x], width=0.9, label=str(x))

    def clear(self):
        self.graph.clear()
