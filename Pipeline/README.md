
# Pipeline for Neonatal Brain Segmentation 

Pipeline for segmenting neonatal brain MRI, developed for T1w and T2w data from the structural pipeline of the [Third Release](https://biomedia.github.io/dHCP-release-notes/) of the [Developing Human Connectome Project](http://www.developingconnectome.org/project/). 

This repository consists of 4 building blocks: 

* a deep-learning-based segmentation pipeline 
* a segmentation pipeline with added age prediction 
* experiments with label budgeting (i.e.: What is the most efficient way of dealing with scarce labeled MRI data?)
* experiments with transfer learning (i.e.: Can we adapt models pre-trained on older infants to the task of segmenting younger infants?)

## Table of Contents

* [Technologies](#technologies)
* [Import Data](#import data)
* [Run Pipeline](#run pipeline)
* [Age Prediction](#age prediction)
* [Label Budgeting](#label budgeting)
* [Transfer Learning](#transfer learning)

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

## Age Prediction

## Label Budgeting

## Transfer Learning
