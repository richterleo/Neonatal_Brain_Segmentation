import numpy as np
import os
import pandas as pd
import time

from AgeTools import sliding_window_inference_age
from BaseTypes import ModeIncompatibleError, MissingModeError, Logger
from copy import deepcopy
from Hyperparams import categories, tissue_classes, default_hyperparams, slicing_modes, selection_modes, transfer_strategies
from monai.inferers import sliding_window_inference
from Utils import get_kernels_strides


class ResultsLogger(Logger):
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

    def __init__(self, mode, slicing_mode = None, selection_mode = None, session_info='', monai_data_dir='Pipeline', random_seed = 0):
        '''Initializes resultsLogger class.
        
        Args:
            mode (str): training mode
            session_info (str): additional information about current run
            monai_data_dir (str): environment variable for current directory
            random_seed (int): random seed for training
        
        '''
        super().__init__(mode, session_info=session_info, monai_data_dir=monai_data_dir, random_seed = random_seed)
        self.result_dir = self.root_dir / 'results' / f"{mode}_results{str(round(self.start))}"
        self.meta_info["result_dir"] = self.result_dir
        self.hyperparams = self.populate_hyperparams()
        self.results = self.populate_results()
        self.analysis = {stat: [] for stat in ["loss", "image_id", "subj_age", "epoch", "step"]}
        if mode == 'agePrediction':
            self.analysis["seg_loss"] = []
            self.analysis["age_loss"] = []
        elif mode == 'labelBudgeting':
            if not (slicing_mode and selection_mode):
                raise MissingModeError()
            else:
                self.hyperparams['slicing_mode'] = slicing_mode
                self.hyperparams['selection_mode'] = selection_mode


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
        
        tissue_dict = {cat: [] for cat in categories}
        sum_dict = {stat: [] for stat in ["epoch_loss", "mean_dice", "mean_dice_imp", 
                    "best_mean_dice", "best_epoch"]}
        time_dict = {"training_time": None}

        if self.mode == 'agePrediction':
            sum_dict["age_epoch_loss"] = []
            sum_dict["seg_epoch_loss"] = []

        return tissue_dict | sum_dict | time_dict
    
    def populate_hyperparams(self):
        '''Create dict to save down hyperparams.'''

        hyperparam_dict = default_hyperparams
        kernels, strides = get_kernels_strides(hyperparam_dict["roi_size"], hyperparam_dict["pixdim"])
        hyperparam_dict["kernels"] = kernels
        hyperparam_dict["strides"] = strides

        return hyperparam_dict
    
    def update_hyperparams(self, **kwargs):
        '''Overwrites hyperparams manually.'''

        roi_size = self.hyperparams["roi_size"]
        pixdim = self.hyperparams["pixdim"]

        for k,v in kwargs.items():
            self.hyperparams[k] = v
        
        if 'selection_mode' in set(kwargs.keys()):
            if kwargs['selection_mode'] not in selection_modes:
                raise ModeIncompatibleError(kwargs['selection_mode'])
            assert 'slicing_mode' in set(kwargs.keys()), "selection mode was given but slicing mode is missing"
            self.hyperparams["hide_labels"] = True
        
        if 'slicing_mode' in set(kwargs.keys()):
            if kwargs['slicing_mode'] not in slicing_modes:
                raise ModeIncompatibleError(kwargs['slicing_mode'])
            assert 'selection_mode' in set(kwargs.keys()), 'slicing mode was given but selection mode is missing'
            self.hyperparams["hide_labels"] = True
        
        # recalculate kernels and strides if roi_size or pixdim has changed
        if roi_size != self.hyperparams["roi_size"] or pixdim != self.hyperparams["pixdim"]:
            print(f"Kernels and strides are adjusted.")
            kernels, strides = get_kernels_strides(self.hyperparams["roi_size"], self.hyperparams["pixdim"])
            self.hyperparams["kernels"] = kernels
            self.hyperparams["strides"] = strides

    def log_tcs(self, metric, metric_batch):
        '''Logs dice scores for each category during training.'''
        
        self.results_dict['mean_dice'].append(metric)

        for i, cat in enumerate(categories):
            tc_metric = metric_batch[i].item()
            self.results[cat].append(tc_metric)

        self.results['mean_dice_imp'].append(np.mean([self.results[tc] for tc in tissue_classes]))

    
    def log_analysis(self, loss, image_id, subj_age, step, epoch):
        '''Logs scores for error analysis'''
        
        self.analysis["loss"].append(loss.item())
        self.analysis["image_id"].append(image_id)
        self.analysis["subj_age"].append(subj_age)
        self.analysis["step"].append(step)
        self.analysis["epoch"].append(epoch)

    def log_age_analysis(self, loss, seg_loss, age_loss, image_id, subj_age, step, epoch):
        '''Logs scores for error analysis with added age prediction'''
        
        self.log_analysis(loss, image_id, subj_age, step, epoch)
        try:
            self.analysis['seg_loss'].append(seg_loss.item())
            self.analysis['age_loss'].append(age_loss.item())
        except KeyError:
            print(f"Analysis dict not properly defined for age prediction, missing keys")

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


class InferenceLogger(Logger):

    def __init__(self, mode, model_path=None, transfer_strategy = None, 
                model_size='big', session_info='', monai_data_dir='Pipeline', random_seed = 0):
        '''Initializes InferenceLogger class.
        
        Args:
            mode (str): training mode
            model_path (Union[str, Path]): path to trained model
            transfer_strategy (str): one of 5 transfer strategies specified in Hyperparams
            session_info (str): additional information about current run
            monai_data_dir (str): environment variable for current directory
            random_seed (int): random seed for training
        
        '''
        super().__init__(mode, session_info=session_info, monai_data_dir=monai_data_dir, random_seed = random_seed)
        self.result_dir = self.root_dir / 'results' / f"Evaluate_{mode}_results{str(round(self.start))}"
        self.meta_info["result_dir"] = self.result_dir
        self.hyperparams = self.populate_hyperparams(model_size)
        self.results = self.populate_results()
        self.model_path = model_path
        self.model_size = model_size
        self.hide_labels = None

        if self.mode == 'transfer':
            self.check_valid_mode(transfer_strategy, transfer_strategies)
            self.transfer_strategy = transfer_strategy
            self.meta_info['transfer_strategy'] = transfer_strategy

    def populate_results(self):
        '''Create dict to save down metrics during training.'''
        
        tissue_dict = {cat: [] for cat in categories}
        sum_dict = {stat: [] for stat in ["mean_dice", "mean_dice_imp"]}
        time_dict = {"training_time": None}

        if self.mode == 'agePrediction':
            sum_dict["mse_metric"] = []
            sum_dict["mae_metric"] = []

        return tissue_dict | sum_dict | time_dict
    
    def populate_hyperparams(self, model_size):
        '''Create dict to save down hyperparams.'''

        hyperparam_dict = default_hyperparams
        if model_size == 'small':
            hyperparam_dict['roi_size'] = [96, 96, 96]
            hyperparam_dict['pixdim'] = [0.6, 0.6, 0.6]   
        kernels, strides = get_kernels_strides(hyperparam_dict["roi_size"], hyperparam_dict["pixdim"])
        hyperparam_dict["kernels"] = kernels
        hyperparam_dict["strides"] = strides
        hyperparam_dict["pretrained_model_path"] = self.model_path

        return hyperparam_dict  


    def infer(self, input, model):
        '''Define sliding window inference function.
        
        Args:
            inputs (array): inputs to evaluate
            model : model to evaluate

        Returns:
            inference function
        '''
        if self.mode == 'agePrediction':

            def _compute(input):
                return sliding_window_inference_age(
                    inputs=input,
                    roi_size=self.hyperparams['roi_size'],
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5)

        else:
    
            def _compute(input):
                return sliding_window_inference(
                    inputs=input,
                    roi_size=self.hyperparams['roi_size'],
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,)

        return _compute(input)

    def log_tcs(self, metric, metric_batch):
        '''Logs dice scores for each category during training.'''
        
        self.results_dict['mean_dice'].append(metric.item())

        for i, cat in enumerate(categories):
            tc_metric = metric_batch[i].item()
            self.results[cat].append(tc_metric)

        self.results['mean_dice_imp'].append(np.mean([self.results[tc] for tc in tissue_classes]))

    def log_age_metrics(self, mse_metric, mae_metric):

        self.results['mse_metric'].append(mse_metric.item())
        self.results['mae_metric'].append(mae_metric.item())



if __name__ == "__main__":
    resultlogger = ResultsLogger('baseline', "only for testing")
    resultlogger.create_result_folder()
    resultlogger.update_hyperparams(roi_size = [256, 256, 256], slicing_mode="random", selection_mode="random")
    resultlogger.stop_clock()
    resultlogger.save_info()



            