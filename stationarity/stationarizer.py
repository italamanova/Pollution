from matplotlib import pyplot


class Stationarity:
    def __init__(self, df):
        self.df = df
        self.box_cox_lambda = 1
        self.diff = 1
        self.sdiff = 1
        self.criteria = ['adf', 'kpss']

    def check_adf(self):
        pass

    def check_kpss(self):
        pass

    def make_box_cox(self):
        pass

    def make_diff(self):
        diff = self.df.diff()
        pyplot.plot(diff)
        pyplot.show()

    def make_sdiff(self):
        diff = self.df.diff(periods=24)
        pyplot.plot(diff)
        pyplot.show()

    def stationarize(self):
        df = self.df
        

