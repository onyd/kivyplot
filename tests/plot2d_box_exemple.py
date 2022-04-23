from kivy.app import App
import numpy as np
from kivyplot import Plot2D


class MainApp(App):

    def build(self):
        N = 5
        data = [np.random.poisson(10, 50) * 10 for _ in range(N)]


        self.root = Plot2D()
        self.root.box([f"C{i}" for i in range(N)], data)
        return self.root


if __name__ == '__main__':
    MainApp().run()
