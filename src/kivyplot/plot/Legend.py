from kivy.core.text import Label as CoreLabel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Ellipse, Line, Rectangle, Color
from kivy.properties import NumericProperty, StringProperty, ColorProperty
from kivy.uix.widget import Widget
from kivy.metrics import sp

class LegendItem(Widget):
    """Base class for legend item"""

    label = StringProperty()
    font_size = NumericProperty(24)
    color = ColorProperty()

    symbol_width = NumericProperty(sp(32))

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_hint = None, None
        self.bind(pos=self.update)

    def update(self, *args):
        self.canvas.after.clear()
        with self.canvas.after:
            self.as_instructions()

    def get_label_texture(self):
        l = CoreLabel(text=self.label, font_size=self.font_size)
        l.refresh()
        self.width = self.symbol_width + l.texture.width
        self.height = l.texture.height
        return l.texture

    def as_instructions(self):
        raise NotImplementedError()


class DotLegendItem(LegendItem):
    radius = NumericProperty()

    def as_instructions(self):
        texture = self.get_label_texture()
        return [Color(rgba=self.color),
                Ellipse(
                    pos=[self.x+self.symbol_width / 2-self.radius,
                         self.y+self.height/2-self.radius],
                    size=[self.radius*2, self.radius*2]),
                Color(rgb=(0, 0, 0)),
                Rectangle(texture=texture, pos=[self.x+self.symbol_width, self.y], size=texture.size)]


class CurveLegendItem(LegendItem):
    line_width = NumericProperty()

    def as_instructions(self):
        texture = self.get_label_texture()
        return [Color(rgba=self.color),
                Line(points=[self.x+0.1*self.symbol_width, self.y+texture.height/2, 0.9 *
                     self.x+self.symbol_width, self.y+texture.height/2], width=self.line_width),
                Color(rgb=(0, 0, 0)),
                Rectangle(texture=texture, pos=[self.x+self.symbol_width, self.y], size=texture.size)]


class BarLegendItem(LegendItem):
    def as_instructions(self):
        texture = self.get_label_texture()
        return [Color(rgba=self.color),
                Rectangle(pos=[self.x+0.1*self.symbol_width, self.y+0.25*texture.height],
                          size=[0.8*self.symbol_width, 0.5*texture.height]),
                Color(rgb=(0, 0, 0)),
                Rectangle(texture=texture, pos=[
                          self.x+self.symbol_width, self.y], size=[0.9*texture.width, 0.9*texture.height])]


class Legend(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content = BoxLayout(size_hint=(
            None, None), orientation='vertical', spacing=sp(12), padding=sp(6))
        self.content.bind(minimum_size=self.content.setter('size'))
        self.size_hint_x = None
        self.content.bind(width=self.setter('width'))
        super().add_widget(self.content)
        self.update()
        self.bind(pos=self.update)
        self.bind(size=self.update)

    def update(self, *args):
        with self.canvas.before:
            Color(rgb=(1, 1, 1))
            Rectangle(pos=self.pos, size=self.size)
            Color(rgb=(0, 0, 0))
            Line(rectangle=[self.x, self.y,
                 self.width, self.height], width=1)

    def add_widget(self, widget, *args, **kwargs):
        return self.content.add_widget(widget, *args, **kwargs)

    def remove_widget(self, widget, *args, **kwargs):
        return self.content.remove_widget(widget, *args, **kwargs)

    def clear_widgets(self, children=None):
        return self.content.clear_widgets(children)
