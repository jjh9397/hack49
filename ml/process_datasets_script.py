from dotenv import load_dotenv
import os
from pathlib import Path
import utils

load_dotenv()

epilepsy_filepath = os.getenv('EPILEPSY')
healthy_filepath = os.getenv('HEALTHY')
datasets_filepath = os.getenv("DATASETS")
epilepsy = Path(rf"{epilepsy_filepath}")
healthy = Path(rf"{healthy_filepath}")
datasets = Path(rf"{datasets_filepath}")


cropped_ep_edf_map = utils.createCroppedFif(datasets, epilepsy)
epoch_subj_map = utils.createEpochFif(datasets, cropped_ep_edf_map)
h_epoch_subj_map = utils.createHealthyFif(datasets, healthy)

print("ep_epoch_subj_map: ", epoch_subj_map)
print("h_epoch_subj_map: ", h_epoch_subj_map)
df, test_df = utils.createDataframe(datasets, epoch_subj_map, h_epoch_subj_map)
df.to_pickle("dataset.pkl")
test_df.to_pickle('testDataset.pkl')

# print(len(datas))