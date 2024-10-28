import os
import yaml

back = lambda x: os.path.dirname(x)
base_path = back(back(os.getcwd()))

class ConfigLoader:
    def __init__(self, cfg_file:str):
        # Load the yaml configuration file
        with open(cfg_file, 'r') as file:
            cfgs = self.data = yaml.safe_load(file)

        self.using_kaggle = cfgs['using_kaggle']

        if self.using_kaggle and 'kaggle_override' in cfgs:
            cfgs.update(cfgs['kaggle_override'])

        self.base_path = base_path

        self.model_save_folder = os.path.normpath(os.path.join(self.base_path, cfgs['output']['path']))

    def get_configs(self):
        return self.data

    