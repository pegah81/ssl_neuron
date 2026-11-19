import json
import argparse

# Pin this training process to the Pascal GPU. It will appear to PyTorch as
# logical cuda:0. Set this before importing modules that load PyTorch.
import os
import torch 

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("Running:", __file__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
for i in range(torch.cuda.device_count()):
    print(f"Visible GPU {i}: {torch.cuda.get_device_name(i)}")

from ssl_neuron.train import Trainer
from ssl_neuron.graphdino import create_model
from ssl_neuron.datasets import build_dataloader

TRIAL = "trial19"
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
