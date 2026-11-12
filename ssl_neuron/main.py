import json
import argparse
from ssl_neuron.train import Trainer
from ssl_neuron.graphdino import create_model
from ssl_neuron.datasets import build_dataloader

TRIAL = "trial14"
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to config file.', type=str, default=f'./configs/{TRIAL}.json')


def main(args):
    # load config
    config = json.load(open(args.config))
    config['trainer']['trial'] = TRIAL
    # load data
    print('Loading dataset: {}'.format(config['data']['class']))
    train_loader, val_loader = build_dataloader(config)

    # build model 
    model = create_model(config)
    trainer = Trainer(config, model, [train_loader, val_loader])

    print('Start training.')
    trainer.train()
    print('Done.')



if __name__ == '__main__':    
    args = parser.parse_args()
    main(args)