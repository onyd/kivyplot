from kivy.app import App
import numpy as np
from kivyplot import Plot2D


class MainApp(App):

    def build(self):
        N = 20
        data = np.random.normal(0, 1, (N, N)) * 10

        self.root = Plot2D()
        self.root.heatmap(data, [f"C{i}" for i in range(N)], [
                          f"C{i}" for i in range(N)])
        return self.root


if __name__ == '__main__':
    MainApp().run()
