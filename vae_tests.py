# build DC-AE models
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
from efficientvit.ae_model_zoo import DCAE_HF
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.apps.utils.image import DMCrop
from dataloader import ImageNetDataset
from torch.utils.data import DataLoader
import random

dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f64c128-in-1.0").to(torch.bfloat16)

# encode

device = torch.device("cuda")
dc_ae = dc_ae.to(device).eval()

transform = transforms.Compose([
    DMCrop(256), # resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
image = Image.open("/home/floatingtrees/Desktop/dog1.jpg")
x = transform(image)[None].to(torch.bfloat16).to(device)
import time
start = time.perf_counter()
import string
def save_tensor(tensor, base_path, other_paths): # shape of (1, batch, h, w)
    for i in range(15):
        characters = string.ascii_letters + string.digits
        # Generate a random string of the specified length
        random_string = ''.join(random.choice(characters) for _ in range(60))
        identifier = random_string[2:]
        directory = random_string[:2]
        dir_path = f"{base_path}/{directory}"
        os.makedirs(dir_path, exist_ok=True)
        full_path = f"{dir_path}/{identifier}.pth"
        if full_path not in other_paths:
            break
        if i == 9:
            raise ValueError("Extremely unlucky")
    
    torch.save(tensor, full_path)
    return full_path
    

splits = ["train", "validation", "test"]
import os
import json
import sys

paths = set()
for split in splits:
    dataset = ImageNetDataset(1000, dtype = torch.bfloat16, split = split)
    dataloader = DataLoader(dataset, 256, num_workers = 15, prefetch_factor= 5)
    with torch.no_grad():
        os.makedirs(split, exist_ok=True)
        base_path = f"{split}"
        for i, batch in enumerate(dataloader):
            batch = batch.to("cuda")

            latent = dc_ae.encode(batch)
            
            if i % 100 == 0:
                y = dc_ae.decode(latent[0:1, :, :, :])
                print(time.perf_counter() - start)
                sys.stdout.flush()
                save_image(y * 0.5 + 0.5, f"images/image{i}.png")
                with open('paths.json', 'w') as f:
                    json.dump(list(paths), f)
            latent = latent.to("cpu")
            latent.detach().cpu()
            for j in range(latent.shape[0]):
                save_path = save_tensor(latent[j, :, :, :].clone(), base_path, paths)
                paths.add(save_path)
