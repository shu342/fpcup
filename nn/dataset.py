"""
Data loaders for PCSE inputs and outputs.
"""
from torch import Tensor, tensor
from torch.utils.data import DataLoader, Dataset

import fpcup
from fpcup.typing import PathOrStr


class PCSEEnsembleDataset(Dataset):
    """
    Handles the loading of PCSE ensemble input/output files in bulk.
    Useful for datasets that are too big to load into memory at once.

    Note: Current approach works for medium-sized data sets; bigger data sets will require larger changes in multiple places.

    To do:
        Use transforms for loading soil/crop data?
        Multiple data directories?
    """
    def __init__(self, data_dir: PathOrStr, *, transform=None, target_transform=None):
        # Basic setup
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        # Get summary filenames - makes sure only completed runs are loaded
        self.summary_files = list(self.data_dir.glob("*.wsum"))
        self.input_files = [f.with_suffix(".wrun") for f in self.summary_files]

    def __len__(self):
        return len(self.summary_files)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        # Load input data
        input_filename = self.input_files[i]
        input_data = fpcup.model.InputSummary.from_file(input_filename).iloc[0]
        sowyear = input_data["DOS"].year
        sowdoy = input_data["DOS"].day_of_year

        input_data = input_data[["latitude", "longitude", "WAV", "RDMSOL"]].to_list() + [sowyear, sowdoy]

        # Load summary data
        summary_filename = self.summary_files[i]
        summary_data = fpcup.model.Summary.from_file(summary_filename).iloc[0]

        matyear = summary_data["DOM"].year
        matdoy = summary_data["DOM"].day_of_year

        summary_data = summary_data[["DVS", "LAIMAX", "TAGP", "TWSO"]].to_list() + [matyear, matdoy]

        return tensor(input_data), tensor(summary_data)
