from kivy.uix.label import Label
from kivy.graphics import Rectangle, Color


class Tooltip(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = None, None
        self.size = self.texture_size
        self.bind(texture_size=self.setter('size'))
        self.bind(pos=self.update
                  )

    def update(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(rgb=(0.2, 0.2, 0.2))
            Rectangle(size=self.size, pos=self.pos)
