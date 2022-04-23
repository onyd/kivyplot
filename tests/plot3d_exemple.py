from kivy.app import App
from kivyplot.extras.geometries import *
from kivyplot import Plot3D
import numpy as np

class MainApp(App):

    def build(self):

        points = [np.array([
            2*np.random.random()-1,
            2*np.random.random()-1,
            2*np.random.random()-1])
            for _ in range(20)]

        self.root = Plot3D()
        self.root.add_points(*points)
        self.root.add_points(np.array([0, 0, 0]), color=(1.0, 0.0, 0.0))
        self.root.add_points(np.array([1, 0, 0]), color=(0.0, 1.0, 0.0))
        self.root.add_points(np.array([0, 1, 0]), color=(0.0, 1.0, 0.0))
        self.root.add_points(np.array([0, 0, 1]), color=(0.0, 1.0, 0.0))
        self.root.render()

        return self.root

if __name__ == '__main__':
    MainApp().run()
