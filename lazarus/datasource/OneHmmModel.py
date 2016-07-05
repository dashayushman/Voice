class OneHmmModel:
    'Common base class for all Models'
    def __init__(self,model):
        self.model = model

    def getModel(self):
        return self.model
