categories = ["BG", "CSF", "cGM", "WM", "bg", "Ventricles", 
            "Cerebellum", "dGM", "Brainstem", 
            "Hippocampus"]

tissue_classes = ["CSF", "cGM", "WM", "Ventricles", 
            "Cerebellum", "dGM", "Brainstem", 
            "Hippocampus"]

default_hyperparams = {'lr': 1e-2,
                    'max_epochs': 40,
                    'hide_labels': False,
                    'batch_size': 2,
                    'age_group': 'whole',
                    'roi_size': [256, 256, 256],
                    'pixdim': [0.5, 0.5, 0.5],
                    'age_loss_weight' : 10e-3}

modes = ['baseline', 'agePrediction', 'labelBudgeting', 'transfer']

slicing_modes = ['random', 'axial', 'sagittal', 'coronal']

selection_modes = ['random', 'equidistant']

transfer_strategies = ['no_fine_tuning', 'fine_tuning', 'deep', 'medium', 'shallow', 'new_model']

transfer_strategies_lrs = {'no_fine_tuning': None, 'fine_tuning': 1e-3, 'deep': 1e-3, 'medium': 1e-3, 'shallow': 1e-3, 'new_model': 1e-3}
transfer_strategies_epochs = {'no_fine_tuning': None, 'fine_tuning': 20, 'deep': 20, 'medium': 20, 'shallow': 20, 'new_model': 40}

colors = ["blue", "brown", "purple", "yellow", "red", "magenta", "indigo", "cornflowerblue",
            "salmon", "peru"]

# when comparing whole-brain and partial annotations, the proportion of images will be balanced out
# such that the total number of training slices is constant
label_dispersion_factor = 0.3


