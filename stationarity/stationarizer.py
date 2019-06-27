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
        pass

    def make_sdiff(self):
        pass

    def stationarize(self):
        df = self.df
        

