class Data(object):
    def __init__(self, conf, training, test, dislike):
        self.config = conf
        self.training_data = training
        self.test_data = test 
        self.dislike_data = dislike
