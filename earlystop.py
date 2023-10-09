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
        self.best_cost = np.inf
        self.counter = 0
        self.train_costs = []
        self.test_costs = []
        self.train_scores = []
        self.test_scores = []
        self.epoch = 0
        self.best_epoch = 0
        self.fig_cost = None
        self.fig_score = None
        self.target_score = target_score
        
    def log(self, train_cost, test_cost, train_score, test_score):
        self.train_costs.append(train_cost)
        self.test_costs.append(test_cost)
        self.train_scores.append(train_score)
        self.test_scores.append(test_score)
        print(f"Epoch {len(self.train_costs)}: train_cost={train_cost:.4f}, test_cost={test_cost:.4f}, train_score={train_score:.4f}, test_score={test_score:.4f}")
    
    def save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": self.config
        }
        torch.save(state, os.path.join(self.outdir, "checkpoint.pth"))
        torch.save(self.config, os.path.join(self.outdir, "config.pth"))
    
    def early_stopping(self, model, monitor="test_score"):
        stop = False
        if monitor == "test_score":
            reference = self.test_scores[-1] 
            improve = reference > self.best_score 
        elif monitor == "test_cost":
            reference = self.test_costs[-1] 
            improve = reference < self.best_cost
        else:
            raise Exception('Only supports monitor={"test_cost", "test_score"}')
        
        if improve:
            if monitor.endswith("_cost"):
                self.best_cost = reference
            elif monitor.endswith("_score"):
                self.best_score = reference

            self.counter = 0
            
            torch.save(model.state_dict(), os.path.join(self.outdir, "best_model.pth"))
        else:
            self.counter += 1
            best = self.best_cost if monitor.endswith("_cost") else self.best_score
            self.best_epoch += 1
            print(f"EarlyStop patience = {self.counter:2}. Best {monitor}: {best:.4f}")

        if self.counter >= self.early_stop_patience or (self.target_score is not None and self.target_score <= reference):
            print(f"Training early stopped. tidak ada peningkatan pada {self.early_stop_patience} epoch terakhir atau mencapai target score.")
            print(f"Early Stopping pada epoch: {self.epoch} | Best {monitor}: {best:.4f}")
            stop = True

        self.epoch += 1
        return stop
    
    
    def reset_early_stop(self):
        self.counter = 0
    
    def plot_cost(self):
        if self.fig_cost is None:
            self.fig_cost = plt.figure()
        plt.plot(self.train_costs, label="train")
        plt.plot(self.test_costs, label="test")
        plt.legend()
        plt.title("Cost vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.savefig(os.path.join(self.outdir, "cost.png"))
        plt.show()
    
    def plot_score(self):
        if self.fig_score is None:
            self.fig_score = plt.figure()
        plt.plot(self.train_scores, label="train")
        plt.plot(self.test_scores, label="test")
        plt.legend()
        plt.title("Score vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.savefig(os.path.join(self.outdir, "score.png"))
        plt.show()
    
    def cost_runtime_plotting(self):
        plt.ion()
        plt.plot(self.train_costs, label="train")
        plt.plot(self.test_costs, label="test")
        plt.legend()
        plt.title("Cost vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.draw()
        plt.pause(0.001)
    
    def score_runtime_plotting(self):
        plt.ion()
        plt.plot(self.train_scores, label="train")
        plt.plot(self.test_scores, label="test")
        plt.legend()
        plt.title("Score vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.draw()
        plt.pause(0.001)
