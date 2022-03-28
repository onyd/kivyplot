from kivy.app import App
from kivy.clock import Clock
from kivyplot.math.transform import vec
from kivyplot.extras.geometries import *
from Plot3D import Plot3D
import numpy as np

class MainApp(App):

    def build(self):

        points = [vec(
            2*np.random.random()-1,
            2*np.random.random()-1,
            2*np.random.random()-1)
            for _ in range(20)]

        self.root = Plot3D(20)
        self.root.add_points(*points)
        self.root.add_points(np.array([0, 0, 0]), color=(1.0, 0.0, 0.0))
        self.root.add_points(np.array([1, 0, 0]), color=(0.0, 1.0, 0.0))
        self.root.add_points(np.array([0, 1, 0]), color=(0.0, 1.0, 0.0))
        self.root.add_points(np.array([0, 0, 1]), color=(0.0, 1.0, 0.0))
        self.root.render()
        #Clock.schedule_interval(self.remove_rd, 1/5.0)

        return self.root

    # def remove_rd(self, *args):
    #     obj = np.random.choice(self.root.scene.children)
    #     print(obj)
    #     self.root.scene.remove(obj)
    #     self.root.renderer.reload(self.root.scene)


if __name__ == '__main__':
    MainApp().run()
