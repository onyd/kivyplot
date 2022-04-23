from kivy.app import App
import numpy as np
from kivyplot import Plot2D


class MainApp(App):

    def build(self):
        N = 20
        points = [np.array([
            i,
            np.sin(2*np.pi*i/(N-1) + np.pi/2)])
            for i in range(N)]

        self.root = Plot2D(xmin=0, xmax=N, stepx=1)
        self.root.plot(points, label='label plot', filled=True)

        return self.root


if __name__ == '__main__':
    MainApp().run()
