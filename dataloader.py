import torch
from datasets import load_dataset
from torchvision.transforms import v2 
from torch.utils.data import Dataset
from PIL import Image

def is_rgb_image(example):
    # Check if the image mode is RGB
    return example['image'].mode == 'RGB'

# Apply the filter

class ImageNetDataset(Dataset):
    def __init__(self, num_classes, dtype = torch.float32, split = "train", transform = None):
        self.split = split
        self.dtype = dtype
        self.num_classes = num_classes
        super().__init__()
        self.hfds = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = v2.Compose([
            v2.Resize((256, 256)),  # Resize the image to 224x224 (standard for ImageNet)
            v2.ToTensor(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
            v2.ToDtype(dtype, scale=True)  # Convert the image to a PyTorch tensor
            ])

    def __len__(self):
        return len(self.hfds[self.split])

    def __getitem__(self, index):
        datapoint = self.hfds[self.split][index]
        pil_image = datapoint["image"]
        pil_image = pil_image.convert("RGB")

        #y_number = datapoint["label"] # this is a single number ie. 539
        x_tensor = self.transform(pil_image).to(self.dtype)

        #y_vector = torch.zeros((self.num_classes), dtype = self.dtype)
        #y_vector[y_number] = 1 # set the corresponding position to 1
        return x_tensor#, y_vector
    
if __name__ == "__main__":
    x = ImageNetDataset(1000)
