import json
import numpy as np
import os
import pandas as pd
import re
import tempfile
import time

from pathlib import Path
from shutil import copyfile, copyfileobj, rmtree


os.environ["MONAI_DATA_DIRECTORY"] = 'Pipeline'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
directory = os.environ.get("MONAI_DATA_DIRECTORY")

root_dir = tempfile.mkdtemp() if directory is None else directory
raw_data_dir = Path(root_dir) / 'dHCP_raw'
dest_dir = Path(root_dir) / 'dHCP'

t1_dir = os.path.join(dest_dir, 'T1w')
t2_dir = os.path.join(dest_dir, 'T2w')
label_dir = os.path.join(dest_dir, 'labels')
meta_dir_pre = os.path.join(dest_dir, 'meta_data_pre')
meta_dir = os.path.join(dest_dir, 'meta_data')

full_list_dirs = [x[0] for x in os.walk(raw_data_dir)]

# filter for the right kind of folder 
folder_match = re.compile('ses-[\d]*/anat')
filtered_list_dirs = [dir for dir in full_list_dirs if folder_match.search(dir) is not None]

# filter the meta_data
file_path_list_meta_data = []
meta_data_match = re.compile('sub-[A-Z\d]*_sessions.tsv')

#Filter t1, t2 images and labels. filter for subid and sesid. example:
#'sub-CC00063BN06_ses-15104_T2w.nii.gz'
file_path_list_t1w = []
file_path_list_t2w = []
file_path_list_label = []
file_match_t1w = re.compile('sub-[A-Z\d]*_ses-[\d]*_T1w.nii.gz')
file_match_t2w = re.compile('sub-[A-Z\d]*_ses-[\d]*_T2w.nii.gz')
file_match_label = re.compile('sub-[A-Z\d]*_ses-[\d]*_desc-drawem9_dseg.nii.gz')

for dir in full_list_dirs:
    partial_file_path_list_meta_data = [os.path.join(dir, file_path) for file_path in os.listdir(dir)
                                 if meta_data_match.search(file_path)]
    partial_file_path_list_t1w = [Path(dir) / file_path for file_path in os.listdir(dir)
                                 if file_match_t1w.search(file_path)]
    print(partial_file_path_list_t1w)
    partial_file_path_list_t2w = [Path(dir) / file_path for file_path in os.listdir(dir)
                                 if file_match_t2w.search(file_path)]
    partial_file_path_list_label = [Path(dir) / file_path for file_path in os.listdir(dir)
                                        if file_match_label.search(file_path)]

    if partial_file_path_list_t1w and partial_file_path_list_label and partial_file_path_list_t2w: #709 items
        file_path_list_t1w.extend(partial_file_path_list_t1w)
        file_path_list_label.extend(partial_file_path_list_label)
        file_path_list_t2w.extend(partial_file_path_list_t2w) #there are more t2w images than t1w images

    file_path_list_meta_data.extend(partial_file_path_list_meta_data)


# Create new directories
Path(t1_dir).mkdir(parents=True, exist_ok=True)
Path(t2_dir).mkdir(parents=True, exist_ok=True)
Path(label_dir).mkdir(parents=True, exist_ok=True)
Path(meta_dir_pre).mkdir(parents=True, exist_ok=True)
Path(meta_dir).mkdir(parents=True, exist_ok=True)

#Copy T1w images, labels and metadata over into newly created folders
for source in file_path_list_t1w:
    source = str(source)
    file_ending = file_match_t1w.findall(source)[0]
    dest = t1_dir + '/' + file_ending
    copyfile(source, dest)

for source in file_path_list_t2w:
    source = str(source)
    file_ending = file_match_t2w.findall(source)[0]
    dest = os.path.join(t2_dir, file_ending)
    copyfile(source, dest)

for source in file_path_list_label:
    source = str(source)
    file_ending = file_match_label.findall(source)[0]
    dest = label_dir + '/' + file_ending
    copyfile(source, dest)

for source in file_path_list_meta_data:
    source = str(source)
    file_ending = meta_data_match.findall(source)[0]
    dest = meta_dir_pre + '/' + file_ending
    copyfile(source, dest)


# Combine collective meta data with other meta data
meta_data_collective = pd.read_csv(raw_data_dir / 'participants.tsv', sep='\t', index_col="participant_id")

# Analyze meta data, merge with collective meta data and rearrange
id_match = 'sub-(.*?)_ses'
session_match = 'ses-(.*?)_T1w'
meta_dicts = []

for img in os.listdir(t1_dir): #dicts containing images, labels and metadata
  #filter out participant_id and session_id
  participant_id = re.search(id_match, str(img)).group(1)
  session_id = re.search(session_match, str(img)).group(1)

  #load in the specific meta data for participant_id and session_id combination
  meta_pre_path = meta_dir_pre + '/sub-' + participant_id + '_sessions.tsv'
  meta_pre_path = Path(meta_pre_path)
  meta_data_specific = pd.read_csv(meta_pre_path, sep='\t', index_col="session_id")
  meta_dict_specific = pd.Series.to_dict(meta_data_specific.loc[int(session_id)])
  meta_dict_collective = pd.Series.to_dict(meta_data_collective.loc[participant_id])

  #merge both sources
  meta_dict = {**meta_dict_collective, **meta_dict_specific}

  #add the meta information to the training data
  meta_dicts.append(meta_dict)


# define custom json encoder class
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)


# copy over preprocessed meta data as json files
for meta_dict, img in zip(meta_dicts, file_path_list_t1w):

    image_path_old = Path(img)
    image_path_ending = str(image_path_old.name)
    basestr = image_path_ending[:-11]

    meta_data_ending = basestr + '_meta_data.json'
    # path where json file should be saved down
    meta_data_path = os.path.join(meta_dir, meta_data_ending)

    # dump json files
    with open(meta_data_path, 'w') as f:
        json.dump(meta_dict, f, cls=NpEncoder)

# delete unprocessed meta data
rmtree(meta_dir_pre, ignore_errors=True)