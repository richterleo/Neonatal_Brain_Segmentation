import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import time

from pathlib import Path

class ageEvaluater:

    def __init__(self, data_dict, plot_dir='AgePlots'):

        self.data_dict = data_dict
        self.scan_ages = [subj["meta_data"]["scan_age"] for subj in self.data_dict]
        self.median_age = np.median(self.scan_ages)
        self.avg_age = np.mean(self.scan_ages)
        self.plot_dir = Path(plot_dir)

        self.age_bins = self.create_age_bins()

    def create_age_bins(self, num_bins):

        return np.linspace(np.min(self.scan_ages), np.max(self.scan_ages), num_bins)    
    
    def plot_age_dist(self):
        plt.title('Histogram of Scan Ages')
        plt.xlabel('Age at Scan Time')
        plt.ylabel('Frequency')
        plt.hist(self.scan_age_list, facecolor='mediumaquamarine')
        plt.grid(True)
        
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        t0 = round(time.time())
        plot_title = 'Scan_age_distribution.png' + str(t0) 
        plt.savefig(self.result_dir / plot_title, bbox_inches='tight')

    def significant_age_difference(self, data_dict2):
        '''Check whether there is a significant age difference between two groups.
        
        Args:
            data_dict2: list of sample data dicts from second group
            
        Returns:
            Bool that signifies whether the p-value is below 5% (True)
        '''
        scan_ages2 = [subj["meta_data"]["scan_age"] for subj in data_dict2]
        _, p = stats.ttest_ind(np.array(self.scan_ages), np.array(scan_ages2), equal_var=False)

        return p < 0.05


if __name__ == "__main__":

    scan_ages = [1,2,1,6,2,5,1,1,1,3,2,4,6,2,3,4,3]
    age_bins = np.linspace(np.min(scan_ages), np.max(scan_ages), 10)              
    age_df = pd.DataFrame(data=scan_ages, columns=["scan_ages"])
    age_df["bucket"] = pd.cut(age_df.scan_ages, age_bins)

    print(age_df["bucket"])

