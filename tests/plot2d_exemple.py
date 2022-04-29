from kivy.app import App
import numpy as np
from kivyplot import Plot2D

class MainApp(App):

    def build(self):

        points = [np.array([
            2*np.random.random()-1,
            2*np.random.random()-1])
            for _ in range(20)]

        self.root = Plot2D(cols=2)
        self.root.scatter(points, label='label scatter', tooltip_text="test")
        self.root.plot(points, label='label plot')
        self.root.bars([(0.5, 0.5)], width=0.1,
                       label='label bar', tooltip_text='bar', col=1)

        return self.root


if __name__ == '__main__':
    MainApp().run()
