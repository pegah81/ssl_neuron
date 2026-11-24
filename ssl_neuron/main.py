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

TRIAL = "trial23"
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
    checkpoint_path = os.path.join(config['trainer']['ckpt_dir'], 'trial22.pt')
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Expand the old xyz-only node projection to include per-node r_median.
    # Preserve the pretrained xyz weights and let the new radius weights learn.
    # model_state = model.state_dict()
    # for encoder_name in ('student_encoder', 'teacher_encoder'):
    #     weight_name = f'{encoder_name}.to_node_embedding.0.weight'
    #     if weight_name not in state_dict or weight_name not in model_state:
    #         continue
    #     old_weight = state_dict[weight_name]
    #     new_weight = model_state[weight_name].clone()
    #     if old_weight.shape[0] == new_weight.shape[0] and old_weight.shape[1] == 3 and new_weight.shape[1] == 4:
    #         new_weight[:, :3] = old_weight
    #         state_dict[weight_name] = new_weight

    model.load_state_dict(state_dict)

    trainer = Trainer(config, model, [train_loader, val_loader])

    print('Start training.')
    trainer.train()
    print('Done.')



if __name__ == '__main__':    
    args = parser.parse_args()
    main(args)
