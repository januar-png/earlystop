import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class Earlystop:
    def __init__(self, model, config=None, early_stop_patience=10, outdir="Model", target_score=None):
        self.model = model
        self.config = config
        self.early_stop_patience = early_stop_patience
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.best_score = -np.inf
        self.best_loss = np.inf
        self.counter = 0
        self.train_losses = []  
        self.val_losses = []    
        self.train_scores = []
        self.val_scores = []    
        self.epoch = 0
        self.best_epoch = 0
        self.fig_loss = None    
        self.fig_score = None
        self.target_score = target_score

    def log(self, train_loss, val_loss, train_score, val_score):
        self.train_losses.append(train_loss)  
        self.val_losses.append(val_loss)      
        self.train_scores.append(train_score)
        self.val_scores.append(val_score)
        print(f"Epoch {len(self.train_losses)}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_score={train_score:.4f}, val_score={val_score:.4f}")

    def save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": self.config
        }
        torch.save(state, os.path.join(self.outdir, "checkpoint.pth"))
        torch.save(self.config, os.path.join(self.outdir, "config.pth"))

    def early_stopping(self, model, monitor="val_score"):  #
        stop = False
        if monitor == "val_score":  
            reference = self.val_scores[-1] 
            improve = reference > self.best_score
        elif monitor == "val_loss":  
            reference = self.val_losses[-1]  
            improve = reference < self.best_loss
        else:
            raise Exception('Only supports monitor={"val_loss", "val_score"}')

        if improve:
            if monitor.endswith("_loss"):
                self.best_loss = reference
            elif monitor.endswith("_score"):
                self.best_score = reference

            self.counter = 0

            torch.save(model.state_dict(), os.path.join(self.outdir, "best_model.pth"))
        else:
            self.counter += 1
            best = self.best_loss if monitor.endswith("_loss") else self.best_score
            self.best_epoch += 1
            print(f"EarlyStop patience = {self.counter:2}. Best {monitor}: {best:.4f}")

        if self.counter >= self.early_stop_patience or (self.target_score is not None and self.target_score <= reference):
            print(f"Training early stopped. No improvement in the last {self.early_stop_patience} epochs or reached target score.")
            print(f"Early Stopping at epoch: {self.epoch} | Best {monitor}: {best:.4f}")
            stop = True

        self.epoch += 1
        return stop

    def reset_early_stop(self):
        self.counter = 0

    def plot_loss(self):  
        if self.fig_loss is None:
            self.fig_loss = plt.figure()
        plt.plot(self.train_losses, label="train")  
        plt.plot(self.val_losses, label="val")     
        plt.legend()
        plt.title("Loss vs. Epoch")  
        plt.xlabel("Epoch")
        plt.ylabel("Loss")  
        plt.savefig(os.path.join(self.outdir, "loss.png"))
        plt.show()

    def plot_score(self):
        if self.fig_score is None:
            self.fig_score = plt.figure()
        plt.plot(self.train_scores, label="train")
        plt.plot(self.val_scores, label="val")
        plt.legend()
        plt.title("Score vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.savefig(os.path.join(self.outdir, "score.png"))
        plt.show()

    def loss_runtime_plotting(self):  
        plt.ion()
        plt.plot(self.train_losses, label="train")  
        plt.plot(self.val_losses, label="val")      
        plt.legend()
        plt.title("Loss vs. Epoch") 
        plt.xlabel("Epoch")
        plt.ylabel("Loss")  
        plt.draw()
        plt.pause(0.001)

    def score_runtime_plotting(self):
        plt.ion()
        plt.plot(self.train_scores, label="train")
        plt.plot(self.val_scores, label="val")
        plt.legend()
        plt.title("Score vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.draw()
        plt.pause(0.001)
