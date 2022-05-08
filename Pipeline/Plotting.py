import matplotlib.pyplot as plt

from Hyperparams import categories, colors

class resultPlotter:
    '''Class for plotting training results
    
    Attributes:
<<<<<<< HEAD
        results (dict[str, list]): training results (loss, dice)
=======
        results (dict[str, list]): training results (loss, dice)\
>>>>>>> d0c887d6b67d318c70a21047f38242c3a59b36ab
        result_dir (pathlib.Path): path to result directory, newly created for each run
        mode (str): training mode
    
    '''

    def __init__(self, results, result_dir, mode, val_interval):
        self.results = results
        self.result_dir = result_dir
        self.mode = mode
        self.val_interval = val_interval

    def plot_epoch_loss_val_dice(self):
        '''Saves down plot of mean losses and validation dice scores over epochs'''

        plt.figure("train", (12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f"Epoch Average Loss in mode: {self.mode}")
        x = [i + 1 for i in range(len(self.results["epoch_loss"]))]
        y = self.results["epoch_loss"]
        plt.xlabel("epoch")
        plt.plot(x, y, color="red")

        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [self.val_interval * (i + 1) for i in range(len(self.results["mean_dice"]))]
        y = self.results["mean_dice"]
        plt.xlabel("epoch")
        plt.plot(x, y, color="green")

        plt.savefig(self.result_dir / 'epoch_average_loss.png')

    def plot_tissue_dice(self):


        plt.figure("train", (18, 6))

        for i, tc in enumerate(categories):
            plt.subplot(2, 5, i)
            plt.title(f"Val Mean Dice {tc}")
            x = [self.val_interval * (i + 1) for i in range(len(self.results[tc]))]
            y = self.results[tc]
            plt.xlabel("epoch")
            plt.plot(x, y, color=colors[i])

        plt.tight_layout()
        plt.savefig(self.result_dir / 'dice_subplots.png')