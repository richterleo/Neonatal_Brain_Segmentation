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
                    'roi_size': [256, 256, 256],
                    'pixdim': [0.5, 0.5, 0.5],
                    'age_loss_weight' : 10e-3}

modes = ['baseline', 'agePrediction', 'labelBudgeting', 'transfer']

slicing_modes = ['random', 'axial', 'sagittal', 'coronal']

selection_modes = ['random', 'equidistant']


