import matplotlib.pyplot as plt
import numpy as np

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