import os
import torch
import torch.optim as optim
import numpy as np
from ssl_neuron.utils import AverageMeter, compute_eig_lapl_torch_batch
import time
from datetime import timedelta
from data.data_utils import extract_latents, save_knn_accuracy
from utils import knn_accuracy
import pandas as pd

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

        if epoch == 0 or (epoch + 1) % 50 == 0:
            self._evaluate_knn(epoch)

        self.model.train()

    def _save_checkpoint(self, epoch, final=False):
        if final:
            filename = f"{self.config['trainer']['trial']}.pt"
        else:
            filename = f"ckpt_{epoch}.pt"
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print('Save model after epoch {} as {}.'.format(epoch, filename))

    def _extract_knn_latents(self):
        self.model.eval()

        datasets = [
            (self.train_loader.dataset, "train"),
            (self.val_loader.dataset, "val")
        ]

        latents = []

        with torch.no_grad():
            for ds, split in datasets:
                for i in range(len(ds)):
                    features, neighbors, cell_id = ds.__getsingleitem__(i)

                    features = torch.as_tensor(
                        features, dtype=torch.float32
                    ).unsqueeze(0).to(self.device)

                    n_nodes = features.shape[1]

                    adj = torch.zeros(
                        1, n_nodes, n_nodes, device=self.device
                    )

                    # Build adjacency from the neighbor dictionary.
                    for node, node_neighbors in neighbors.items():
                        node = int(node)

                        if node < 0 or node >= n_nodes:
                            continue

                        if isinstance(node_neighbors, dict):
                            node_neighbors = node_neighbors.keys()

                        for neighbor in node_neighbors:
                            neighbor = int(neighbor)

                            if 0 <= neighbor < n_nodes:
                                adj[0, node, neighbor] = 1.0

                    # Compute positional encoding exactly as in training.
                    lapl = compute_eig_lapl_torch_batch(adj)

                    # Extract CLS latent from the EMA teacher encoder.
                    latent, _ = self.model.teacher_encoder(
                        features, adj, lapl
                    )

                    latents.append({
                        "cell_id": cell_id,
                        "latent": latent.squeeze(0).cpu().numpy(),
                        "split": split
                    })

        latents = pd.DataFrame(latents)

        # Attach cell-type labels.
        metadata = pd.read_csv(
            os.path.join(self.config["data"]["path"], "cell_meta.csv"),
            usecols=["cell_id", "celltype_final"],
            dtype={"cell_id": str},
        )
        latents["cell_id"] = latents["cell_id"].astype(str)
        latents = latents.merge(metadata, on="cell_id", how="left")

        return latents


    def _evaluate_knn(self, epoch):

        latents = self._extract_knn_latents()

        train = latents[latents["split"] == "train"].copy()
        val = latents[latents["split"] == "val"].copy()
        train_val = latents.copy()

        k = self.config["trainer"].get("knn_k", 10)

        train_acc, _ = knn_accuracy(
            latents=train,
            subset=train,
            k=k,
            balanced=False
        )

        val_acc, _ = knn_accuracy(
            latents=train_val,
            subset=val,
            k=k,
            balanced=False
        )

        balanced_train_acc, _ = knn_accuracy(
            latents=train,
            subset=train,
            k=k,
            balanced=True
        )

        balanced_val_acc, _ = knn_accuracy(
            latents=train_val,
            subset=val,
            k=k,
            balanced=True
        )

        print(
            f"\nEpoch {epoch + 1} | KNN accuracy"
            f"\nTrain accuracy: {train_acc:.4f}"
            f"\nValidation accuracy: {val_acc:.4f}"
            f"\nBalanced train accuracy: {balanced_train_acc:.4f}"
            f"\nBalanced validation accuracy: {balanced_val_acc:.4f}"
        )

        results = pd.DataFrame([{
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "balanced_train_acc": balanced_train_acc,
            "balanced_val_acc": balanced_val_acc,
            "k": k
        }])

        output_dir = os.path.join(self.config["trainer"]["acc_dir"])

        os.makedirs(output_dir, exist_ok=True)

        trial = self.config["trainer"]["trial"]
        file_path = os.path.join(output_dir, f"{trial}.csv")

        results.to_csv(
            file_path,
            mode="a",
            header=not os.path.exists(file_path),
            index=False
        )

        print(f"KNN results saved to {file_path}")


        

