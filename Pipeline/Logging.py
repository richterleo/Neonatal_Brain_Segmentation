import os
import pandas as pd
import tempfile
import time

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from utils import get_kernels_strides

class ModeIncompatibleError(Exception):
    '''Raised when incompatible mode is chosen

    Attributes:
        mode (str): training mode
        message (str): error message
    '''

    def __init__(self, mode, message= "Invalid mode was selected."):
        self.mode = mode
        self.message = message
        super().__init__(self.message)

class resultsLogger:
    '''Class for logging and saving meta data, hyperparams and metrics during and after training
    
    Attributes:
        mode (str): training mode
        root_dir (pathlib.Path): path to root directory
        start (str): time at start of training
        result_dir (pathlib.Path): path to results folder, newly created for every run
        meta_info (dict[str, str]): meta info about the current run
        hyperparams (dict[str, Union[int, list, str, bool]]): hyperparams of current run
        results (dict[str, Union[list, None]]): results of current run
        analysis (dict[str, list]): additional metrics calculated after training
        end (Union[None, str]): time at end of training

    '''
    categories = ["BG", "CSF", "cGM", "WM", "bg", "Ventricles", 
                "Cerebellum", "dGM", "Brainstem", 
                "Hippocampus"]
    
    tissue_classes = ["CSF", "cGM", "WM", "Ventricles", 
                "Cerebellum", "dGM", "Brainstem", 
                "Hippocampus"]

    default_hyperparams = {'lr': 1e-2,
                        'max_epochs': 20,
                        'hide_labels': False,
                        'prop_of_whole': 1,
                        'batch_size': 2,
                        'age_group': 'whole',
                        'roi_size': [128, 128, 128],
                        'pixdim': [0.5, 0.5, 0.5]}

    modes = ['baseline', 'agePrediction', 'labelBudgeting', 'transfer']

    slicing_modes = ['random', 'axial', 'sagittal', 'coronal']

    selection_modes = ['random', 'equidistant']

    def __init__(self, mode, session_info='', monai_data_dir='Pipeline', random_seed = 0):
        '''Initializes resultsLogger class.
        
        Args:
            mode (str): training mode
            session_info (str): additional information about current run
            monai_data_dir (str): environment variable for current directory
            random_seed (int): random seed for training
        
        '''
        self.check_valid_mode(mode)
        self.mode = mode
        self.root_dir = Path(self.get_root_dir(monai_data_dir))
        self.start = time.time()
        self.result_dir = self.root_dir / 'results' / f"{mode}_results{str(round(self.start))}"
        self.meta_info = {"session_date": str(datetime.today()), "result_dir": self.result_dir, "mode": self.mode,
                            "session_info": session_info, "random_seed": random_seed}
        self.hyperparams = self.populate_hyperparams()
        self.results = self.populate_results()
        self.analysis = {stat: [] for stat in ["loss", "image_id", "subj_age", "epoch", "step"]}
        self.end = None
    
    @staticmethod
    def check_valid_mode(mode):
        '''Checks if mode is valid'''
        if mode not in resultsLogger.modes:
                raise ModeIncompatibleError(mode)


    def create_result_folder(self):
        '''Creates unique results folder'''
        try:
            self.result_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Result folder already exists.")
        else:
            print(f"Results are saved in: {self.result_dir}.")

    def populate_results(self):
        '''Create dict to save down metrics during training.'''
        
        tissue_dict = {tc: [] for tc in resultsLogger.categories}
        sum_dict = {stat: [] for stat in ["epoch_loss", "mean_dice", "mean_dice_imp", 
                    "best_mean_dice", "best_epoch"]}
        time_dict = {"training_time": None}

        return tissue_dict | sum_dict | time_dict
    
    def populate_hyperparams(self):
        '''Create dict to save down hyperparams.'''

        hyperparam_dict = resultsLogger.default_hyperparams
        kernels, strides = get_kernels_strides(hyperparam_dict["roi_size"], hyperparam_dict["pixdim"])
        hyperparam_dict["kernels"] = kernels
        hyperparam_dict["strides"] = strides

        return hyperparam_dict

    def get_root_dir(self, monai_data_dir):
        '''Sets environment variables'''

        os.environ["MONAI_DATA_DIRECTORY"] = monai_data_dir
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        directory = os.environ.get("MONAI_DATA_DIRECTORY")
        return tempfile.mkdtemp() if directory is None else directory
    
    def update_hyperparams(self, **kwargs):
        '''Overwrites hyperparams manually.'''

        roi_size = self.hyperparams["roi_size"]
        pixdim = self.hyperparams["pixdim"]

        for k,v in kwargs.items():
            self.hyperparams[k] = v
        
        if 'selection_mode' in set(kwargs.keys()):
            if kwargs['selection_mode'] not in resultsLogger.selection_modes:
                raise ModeIncompatibleError(kwargs['selection_mode'])
            assert 'slicing_mode' in set(kwargs.keys()), "selection mode was given but slicing mode is missing"
            self.hyperparams["hide_labels"] = True
        
        if 'slicing_mode' in set(kwargs.keys()):
            if kwargs['slicing_mode'] not in resultsLogger.slicing_modes:
                raise ModeIncompatibleError(kwargs['slicing_mode'])
            assert 'selection_mode' in set(kwargs.keys()), 'slicing mode was given but selection mode is missing'
            self.hyperparams["hide_labels"] = True
        
        # recalculate kernels and strides if roi_size or pixdim has changed
        if roi_size != self.hyperparams["roi_size"] or pixdim != self.hyperparams["pixdim"]:
            print(f"Kernels and strides are adjusted.")
            kernels, strides = get_kernels_strides(self.hyperparams["roi_size"], self.hyperparams["pixdim"])
            self.hyperparams["kernels"] = kernels
            self.hyperparams["strides"] = strides

    def log_tcs(self, metric_batch):
        '''Logs dice scores for each category during training.'''

        for i, cat in enumerate(resultsLogger.categories):
            metric = metric_batch[i].item()
            self.results[cat].append(metric)
    
    def log_analysis(self, loss, image_id, subj_age, step, epoch):
        '''Logs scores for error analysis'''
        
        self.analysis["loss"].append(loss.item())
        self.analysis["image_id"].append(image_id)
        self.analysis["subj_age"].append(subj_age)
        self.analysis["step"].append(step)
        self.analysis["epoch"].append(epoch)

    def save_info(self):
        '''Save down info dicts as csv files'''

        df_meta = pd.DataFrame(self.meta_info, index=[0])
        df_meta.to_csv(os.path.join(self.result_dir, 'meta_info.csv'), sep="\t", index=False)

        df_results = pd.DataFrame(self.results)
        df_results.to_csv(os.path.join(self.result_dir, 'results.csv'), sep="\t", index_label="epoch")

        df_analysis = pd.DataFrame(self.analysis)
        df_analysis.to_csv(os.path.join(self.result_dir, 'loss_analysis.csv'), sep="\t", index=False)

        hyperparams = deepcopy(self.hyperparams)
        hyperparams = {k: str(v) for k, v in hyperparams.items()}
        df_hyperparams = pd.DataFrame(hyperparams, index=[0])
        df_hyperparams.to_csv(os.path.join(self.result_dir, 'hyperparams.csv'), sep="\t", index=False)

    def restart_clock(self):
        '''Restarts clock when training starts'''
        self.start = time.time()
    
    def stop_clock(self):
        '''Stores time from class initialization to current process.'''

        t1 = time.time()
        self.end = str(round(t1))
        self.meta_info["duration"] = str(t1-self.start)



if __name__ == "__main__":
    resultlogger = resultsLogger('baseline', "only for testing")
    resultlogger.create_result_folder()
    resultlogger.update_hyperparams(roi_size = [256, 256, 256], slicing_mode="random", selection_mode="random")
    resultlogger.stop_clock()
    resultlogger.save_info()



            