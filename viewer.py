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

dc_ae = dc_ae.to("cuda").eval()
with torch.no_grad():
    path = "/home/floatingtrees/projects/vae_compress/train/0a/ajI70sYG29plDQgWG1Y12Ox4m4tYXjIvo35EZlw751rymTRqJvfmXlS2a3.pth"
    import os 
    file_size = os.path.getsize(path)
    print(file_size)
    latent = torch.load(path, weights_only=True).to("cuda")
    #latent = latent.cpu()
    #torch.save(latent, "tester.pth")
    #file_size = os.path.getsize("tester.pth")
    #print(file_size)
    print(latent.shape)
    y = dc_ae.decode(latent.unsqueeze(0))
    save_image(y * 0.5 + 0.5, f"images/image.png")