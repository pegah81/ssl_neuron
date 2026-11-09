import os
import torch
import torch.optim as optim
import numpy as np
from ssl_neuron.utils import AverageMeter, compute_eig_lapl_torch_batch
import time
from datetime import timedelta
from data.data_utils import extract_latents

class Trainer(object):
    def __init__(self, config, model, dataloaders):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.ckpt_dir = config['trainer']['ckpt_dir']
        self.save_every = config['trainer']['save_ckpt_every']

        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader= dataloaders[1]

        ### trainings params
        self.max_iter = config['optimizer']['max_iter']
        self.init_lr = config['optimizer']['lr']
        self.exp_decay = config['optimizer']['exp_decay']
        self.lr_warmup = torch.linspace(0., self.init_lr,  steps=(self.max_iter // 50)+1)[1:]
        self.lr_decay = self.max_iter // 5
        
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=0)

        self.train_loss_history = []
        self.val_loss_history = []
        
        
    def set_lr(self): 
        if self.curr_iter < len(self.lr_warmup):
            lr = self.lr_warmup[self.curr_iter]
        else:
            lr = self.init_lr * self.exp_decay ** ((self.curr_iter - len(self.lr_warmup)) / self.lr_decay)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
        

    
   
    def train(self):
        self.curr_iter = 0
        epoch = 0
        start_time = time.perf_counter()

        while self.curr_iter < self.max_iter:
            # Run one epoch.
            epoch_start = time.perf_counter()
            self._train_epoch(epoch)
            self._validate_epoch(epoch)

            # Time elapsed and estimated remaining time.
            elapsed = time.perf_counter() - start_time
            avg_iter_time = elapsed / self.curr_iter
            remaining_iters = self.max_iter - self.curr_iter
            remaining = avg_iter_time * remaining_iters

            print(
                f"Epoch {epoch} | "
                f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                f"ETA: {timedelta(seconds=int(remaining))} | "
                f"Epoch time: {timedelta(seconds=int(time.perf_counter() - epoch_start))}"
            )

            if epoch % self.save_every == 0:
                # Save checkpoint.
                self._save_checkpoint(epoch)

            epoch += 1

        self._save_checkpoint(epoch, final=True)

        total_time = time.perf_counter() - start_time
        print(
            f"\nTraining complete! "
            f"Total time: {timedelta(seconds=int(total_time))}"
        )
        try:
            loss_path = os.path.join(
                self.config['trainer']['losses_dir'],
                self.config['trainer']['trial']
            )
            os.makedirs(loss_path, exist_ok=True)
            np.save(
                os.path.join(loss_path, 'training_loss.npy'),
                np.array(self.train_loss_history)
            )

            np.save(
                os.path.join(loss_path, 'val_loss.npy'),
                np.array(self.val_loss_history)
            )

            extract_latents(self.config)


        except Exception as e:
            print(f"Could not save training losses: {e}")



    def _train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        for i, data in enumerate(self.train_loader, 0):
            f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data]
            n = a1.shape[0]

            # compute positional encoding
            l1 = compute_eig_lapl_torch_batch(a1)
            l2 = compute_eig_lapl_torch_batch(a2)
            
            self.lr = self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)
            
            loss = self.model(f1, f2, a1, a2, l1, l2)

            # optimize 
            loss.sum().backward()
            self.optimizer.step()
            
            # update teacher weights
            self.model.update_moving_average()
            
            losses.update(loss.detach(), n)
            self.curr_iter += 1

        self.loss = losses.avg.item() if torch.is_tensor(losses.avg) else losses.avg
        self.train_loss_history.append(self.loss)
        print('Epoch {} | Loss {:.4f}'.format(epoch, losses.avg))


    def _validate_epoch(self, epoch):
        self.model.eval()
        losses = AverageMeter()

        with torch.no_grad():
            for data in self.val_loader:
                f1, f2, a1, a2 = [
                    x.float().to(self.device, non_blocking=True)
                    for x in data
                ]

                l1 = compute_eig_lapl_torch_batch(a1)
                l2 = compute_eig_lapl_torch_batch(a2)

                loss = self.model(f1, f2, a1, a2, l1, l2)

                losses.update(loss.detach(), a1.shape[0])
        val_loss = losses.avg.item()
        self.val_loss_history.append(val_loss)
        print(f"Epoch {epoch} | Validation Loss: {val_loss:.4f}")

        self.model.train()

    def _save_checkpoint(self, epoch, final=False):
        if final:
            filename = f"{self.config['trainer']['trial']}.pt"
        else:
            filename = f"ckpt_{epoch}.pt"
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print('Save model after epoch {} as {}.'.format(epoch, filename))