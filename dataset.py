import numpy as np
import os
from torch.utils.data import Dataset

class CellsDataset(Dataset):
    def __init__(self, cells, path):
        """
        Initalize the dataset.
        Cells is a list of cells with the columns: image_id, cell_id, centroid-0, centroid-1.
        """
        super().__init__()
        # Complete here the initlization of the dataset
        # HINT: Use caching for faster implementation
        self.cells = cells
        self.path = path
        
        self.image_cache = {}
        self.cells2labels_cache = {}
        self.seg_cache = {}
        
                
    def __len__(self):
        """
         Return the size of the dataset
        """
        return len(self.cells)

    def __getitem__(self, i):
        """
        Return a crop of all the proteins around the ith cell in the data and its cell type. The crop should be of size 60x60 pixels.
        """
        
        cell = self.cells.iloc[i]
        image_id = cell['image_id']
        path = self.path
        
        if image_id in self.image_cache:
            image = self.image_cache[image_id]
            cells2labels = self.cells2labels_cache[image_id]
            seg = self.seg_cache[image_id]
        else:
            image = np.load(f"{path}/data/images/{image_id}.npz")["data"]
            cells2labels = np.load(f"{path}/cells2labels/{image_id}.npz")["data"]
            seg = np.load(f"{path}/cells/{image_id}.npz")["data"]
            self.image_cache[image_id] = image         
            self.cells2labels_cache[image_id] = cells2labels
            self.seg_cache[image_id] = seg
        
        cell_id = cell['cell_id']
        label = cells2labels[cell_id]

        if label < 0:
            label = 15
        else:
            label = label
        
        size_px=60
        x_location, y_location = int(cell['centroid-0']), int(cell['centroid-1'])
        crop = image[x_location-size_px//2:x_location+size_px//2,
            y_location-size_px//2:y_location+size_px//2, :]
        seg_crop = seg[x_location-size_px//2:x_location+size_px//2,
            y_location-size_px//2:y_location+size_px//2]
        
        return crop,seg_crop == cell_id,label