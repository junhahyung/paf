class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        if self.config.train:
            print("Saving model...")
            if self.config.get('save_weight') == True:
                self.model.save_weights(checkpoint_path)
            if self.config.get('save_weight') == False:
                self.model.save_model(checkpoint_path)

            print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        if self.config.train:
            print("Loading model checkpoint {} ...\n".format(checkpoint_path))
            self.model.load_weights(checkpoint_path)
            print("Model loaded")

    def build_model(self):
        raise NotImplementedError