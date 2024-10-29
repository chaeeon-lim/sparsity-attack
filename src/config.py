import yaml

class ConfigLoader:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)

    @classmethod
    def load_config(cls, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
        
# Test codes
if __name__ == "__main__":
    # Example usage:
    config_file = 'configs/cifar10_conv2.yml'
    loader = ConfigLoader(config_file)
    print(loader.config)  # prints the loaded configuration
    print(loader.dataset)
