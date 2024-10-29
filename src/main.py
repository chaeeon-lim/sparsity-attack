import os
import sys
from argparse import ArgumentParser
from rich.console import Console

from config import ConfigLoader
from sparsity import SparsityModel, get_sparsity_function
from models import get_model
from utils import get_dataset, write_tensor_list, read_tensor_list, draw_decrease_in_activation_sparsity

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
    model = get_model(config['model'])
    solver = SparsityModel(model, **config['sparsity'])

    #TODO: Evaluate the adversarial results
    def parse_range(givenRange):
        result = givenRange.split(sep=":", maxsplit=1)
        if len(result) == 2:
            return [int(result[0]), int(result[0])+1]
        else:
            return result
    if "evaluate" in config.keys():
        evalConfig = config['evaluate']
        result = {}
        targetIndexStart, targetIndexEnd = parse_range(evalConfig['xClean']) if 'xClean' in evalConfig.keys() else [0,1]
        betaOrig = int(solver.beta)
        funcOrig = solver.sparsity_function
        if "betas" in evalConfig.keys():
            betas = evalConfig['betas']
        else: 
            betas = [solver.beta]
        
        if 'useSaved' in evalConfig.keys():
            for beta, path in zip(betas, evalConfig['useSaved']):
                result[beta] = read_tensor_list(path)
        else:
            if 'functions' in evalConfig.keys():
                funcs = [get_sparsity_function(f) for f in evalConfig["functions"]]
            else:
                funcs = [solver.sparsity_function]
            for func in funcs:
                funcName = func.__name__
                with console.status(f"[bold green] evaluating with function: [italic red]{funcName}[bold green]") as status:
                    result[funcName] = []
                    for beta in betas:
                        solver.beta = beta
                        solver.sparsity_function = func
                        ratio = solver.get_decrease_in_activation_sparsity(train_images[targetIndexStart:targetIndexEnd])
                        result[funcName].append(ratio)
                        console.log(f"beta: {beta} completed (ratio: {ratio})", end="")
                    if 'saveName' in evalConfig.keys():
                        path = evalConfig['savePath'] if 'savePath' in evalConfig.keys() else "outputs/"
                        write_tensor_list(result[funcName], f"{path}/{evalConfig['saveName']}_{funcName}.tfrecrod")

        if evalConfig['plotGraph']:
            draw_decrease_in_activation_sparsity(betas, result)

        solver.beta = betaOrig
        solver.sparsity_function = funcOrig
     
    #TODO: Create adversarial inputs       
    # if "generate" in config.keys():
    #     genConfig = config['generate']
    #     savePath = genConfig['saveName']
    #     os.makedirs(savePath, exist_ok=True)

    #TODO: Save the adversarial inputs.
    
    return 0

if __name__ == "__main__":
    sys.exit(main())