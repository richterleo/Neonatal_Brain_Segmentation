
# Pipeline for Neonatal Brain Segmentation 

Pipeline for segmenting neonatal brain MRI, developed for T1w and T2w data from the structural pipeline of the [Third Release](https://biomedia.github.io/dHCP-release-notes/) of the [Developing Human Connectome Project](http://www.developingconnectome.org/project/). 

This repository consists of 4 building blocks: 

* a deep-learning-based segmentation pipeline 
* a segmentation pipeline with added age prediction 
* experiments with label budgeting (i.e.: What is the most efficient way of dealing with scarce labeled MRI data?)
* experiments with transfer learning (i.e.: Can we adapt models pre-trained on older infants to the task of segmenting younger infants?)

## Table of Contents

* [Technologies](#technologies)
* [Import Data](#importdata)
* [Run Pipeline](#runpipeline)
* [Age Prediction](#ageprediction)
* [Label Budgeting](#labelbudgeting)
* [Transfer Learning](#transferlearning)

## Technologies

## Import Data

Downloaded data from the structural pipeline of the Third Release of the Developing Human Connectome Project should consist of folders (e.g. 'sub-CC00050XX01'). 
Raw data must be stored in './Pipeline/dHCP_raw'. Please make sure the file 'participants.tsv' is located in the same directory. Then execute `ExtractFiles.py`. '.Pipeline/dHCP' should now be populated. The folder structure should look like this:

```bash
.
├── Pipeline/
│   ├── dHCP/
│   │   ├── T1w
│   │   ├── T2w
│   │   ├── meta_data
│   │   └── labels
│   ├── dHCP_raw/
│   │   └── ...
│   └── ...
└── ReadME
```

## Run Pipeline

The main training loop is contained in `Train.py`. Hyperparameters can be changed manually in `Hyperparams.py`. Epoch-wise loss and validation Dice Scores (per tissue class and on average), among other metrics, as well as the current best model and training graphs will be saved to a newly created result folder under '.Pipeline/results'.

## Age Prediction

Additional age prediction can be performed by specifying the `mode` attribute of the ResultsLogger in `Train.py` as `agePrediction`, i.e.

  resultlogger = ResultsLogger('baseline', session_info='Only for testing')
  
and then executing the training pipeline. 

## Label Budgeting

To train a model on partially annotated MRI scans, set the `mode` of the ResultsLogger in `Train.py` to `labelBudgeting`. By default, 30% of the slices of each scan will be annotated and the number of samples will be tripled. 

## Transfer Learning
