from kivy.app import App
import numpy as np
import pandas as pd
from kivyplot import Plot2D

class MainApp(App):

    def build(self):
        N = 5
        data = pd.DataFrame({f'C{i}' : np.random.poisson(10, 50) for i in range(N)})

        self.root = Plot2D()
        self.root.violin(data)
        return self.root


if __name__ == '__main__':
    MainApp().run()
