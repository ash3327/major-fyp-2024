import yaml

class ConfigLoader:
    def __init__(self, cfg_file:str):
        # Load the yaml configuration file
        with open(cfg_file, 'r') as file:
            self.data = yaml.safe_load(file)

    def get_configs(self):
        return self.data