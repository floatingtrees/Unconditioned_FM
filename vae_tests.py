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
dataset = ImageNetDataset(1000)
dataloader = DataLoader(dataset, 64, num_workers = 15, prefetch_factor= 5)

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
def save_tensor(tensor, base_path): # shape of (1, batch, h, w)
    characters = string.ascii_letters + string.digits
    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(40))
    identifier = random_string[3:]
    directory = random_string[:3]
    dir_path = f"{base_path}/{directory}"
    os.makedirs(dir_path, exist_ok=False)
    full_path = f"{dir_path}/{identifier}"
    torch.save(tensor, full_path)
    return full_path
    

splits = ["train", "validation", "test"]
import os
for split in splits:
    with torch.no_grad():
        os.makedirs(split, exist_ok=False)
        base_path = f"{split}/"
        for i, batch in enumerate(dataloader):
            batch = batch.to("cuda")
            latent = dc_ae.encode(x)
            
            if i % 100 == 0:
                y = dc_ae.decode(latent)
                print(time.perf_counter() - start)
                save_image(y * 0.5 + 0.5, "demo_dc_ae.png")
            exit()