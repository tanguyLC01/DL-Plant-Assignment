import os
import random
import cv2
from torch.utils.data import Dataset

class PlantVillageDataset(Dataset):
    def __init__(self, root_dir, img_mode='RGB', transform=None, train=True, test_split=0.2, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.test_split = test_split
        self.img_labels = []
        self.img_paths = []
        self.transform = transform
        self.label_to_idx = {}
        self.img_mode = img_mode
        
        for idx, label in enumerate(sorted(os.listdir(self.root_dir))):
            label_dir = os.path.join(self.root_dir, label)
            self.label_to_idx[label] = idx
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.img_paths.append(img_path)
                self.img_labels.append(idx)
                
        # Shuffle and split into train and test sets
        random.seed(seed)
        data = list(zip(self.img_paths, self.img_labels))
        random.shuffle(data)
        split_idx = int(len(data) * (1 - self.test_split))

        if self.train:
            data = data[:split_idx]
        else:
            data = data[split_idx:]

        self.image_paths, self.labels = zip(*data)
                
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]
        
        # Load Image
        image = cv2.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label