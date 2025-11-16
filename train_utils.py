import torch
import matplotlib.pyplot as plt
from collections import defaultdict


def draw_fig(history, epochs, save_path=None):

    tr_loss = history['train_loss']
    val_loss = history['val_loss']
    tr_acc = history['train_acc']
    val_acc = history['val_acc']

    x = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, tr_loss, '.-', label='Train Loss')
    plt.plot(x, val_loss, '.-', label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, tr_acc, '.-', label='Train Accuracy')
    plt.plot(x, val_acc, '.-', label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plots to {save_path}")
    # plt.show()


def get_scheduler(optimizer, lr_patience=10, lr_factor=0.3):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=lr_patience, verbose=True, factor=lr_factor
    )


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
        else:
            self.best_score = score
            self.epochs_no_improve = 0