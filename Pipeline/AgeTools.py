import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ageEvaluater:

    def __init__(self, datacollector):

        self.datacollector = datacollector
        self.scan_ages = [subj["meta_data"]["scan_age"] for subj in self.datacollector.data_dict]
        self.median_age = np.median(self.scan_ages)
        self.avg_age = np.mean(self.scan_ages)
        self.result_dir = self.datacollector.result_dir

        self.age_bins = self.create_age_bins()

    def create_age_bins(self, num_bins):

        return np.linspace(np.min(self.scan_ages), np.max(self.scan_ages), num_bins)    
    
    def plot_age_dist(self):
        plt.title('Histogram of Scan Ages')
        plt.xlabel('Age at Scan Time')
        plt.ylabel('Frequency')
        plt.hist(self.scan_age_list, facecolor='mediumaquamarine')
        plt.grid(True)
        plt.savefig(self.result_dir / "Scan_age_distribution.png", bbox_inches='tight')


if __name__ == "__main__":

    scan_ages = [1,2,1,6,2,5,1,1,1,3,2,4,6,2,3,4,3]
    age_bins = np.linspace(np.min(scan_ages), np.max(scan_ages), 10)              
    age_df = pd.DataFrame(data=scan_ages, columns=["scan_ages"])
    age_df["bucket"] = pd.cut(age_df.scan_ages, age_bins)

    print(age_df["bucket"])