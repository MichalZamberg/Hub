import numpy as np
from torch.utils.data import Dataset

class CellsDataset(Dataset):
    def __init__(self, cells):
        """
        Initalize the dataset.
        Cells is a list of cells with the columns: image_id, cell_id, centroid-0, centroid-1.
        """
        super().__init__()
        # Complete here the initlization of the dataset
        # HINT: Use caching for faster implementation

        
    def __len__(self):
        """
         Return the size of the dataset
        """

    def __getitem__(self, i):
        """
        Return a crop of all the proteins around the ith cell in the data and its cell type. The crop should be of size 60x60 pixels.
        """
        # Complete the function such that it'll return the ith cell's image, it's segmentation and its label.
        # The returned img should be a crop around the center of the ith cell.
        # The returned img should be shaped (H, W, C). where C represents the number of channels in image_path. H, W should be set to 60 pixels. such that the size of the crop will be 60x0px
        # The returned segmentation should be a crop (H, W) around the center of the ith cell. the values of the segmentation should be 1 where the cell exists and 0 otherwise.
        # Note that any cell types which is undefined (i.e. negative) should be set to 15 (15 is a new cell types we are defining here).