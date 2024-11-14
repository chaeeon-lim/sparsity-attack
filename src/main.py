import os
import sys
from argparse import ArgumentParser
from rich.console import Console

from config import ConfigLoader
from sparsity import SparsityModel, get_sparsity_function
from models import get_model
from utils import get_dataset, draw_decrease_in_activation_sparsity

console = Console()

def parse_config():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path"
                        , help="Path of configuration file"
                        , type=str
                        , default="configs/cifar10_conv2.yml")

    #TODO: Add arguments
    args = parser.parse_args()
    config = ConfigLoader(args.config_path).config
    return config

def main():
    #TODO: Parsing arguments
    config = parse_config()

    #TODO: Prepare dataset
    if('dataset' in config.keys()):
        train_images, train_labels, test_images, test_labels = get_dataset(config['dataset'])
    

    #TODO: Get model
    model = get_model(config['baseModel'])
    solver = SparsityModel(model, **config['sparsity'])
    solver.data_clean = train_images

    #TODO: Evaluate the adversarial results
    if "evaluate" in config.keys():
        evalConfig = config['evaluate']

        if "betas" in evalConfig.keys():
            betas = evalConfig['betas']
        else: 
            betas = [solver.beta]
        result = solver.eval_decrease_in_activation_sparsity(evalConfig)

        if evalConfig['plotGraph']:
            draw_decrease_in_activation_sparsity(betas, result)

     
    #TODO: Create adversarial inputs       
    # if "generate" in config.keys():
    #     genConfig = config['generate']
    #     savePath = genConfig['saveName']
    #     os.makedirs(savePath, exist_ok=True)

    #TODO: Save the adversarial inputs.
    
    return 0

if __name__ == "__main__":
    sys.exit(main())