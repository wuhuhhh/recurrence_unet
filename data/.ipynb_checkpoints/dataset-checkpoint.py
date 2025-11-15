import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class LandslideDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg','.TIF'))]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        # 格式转换，将图片都转换成png之后，去掩码中找对应的，*.png（因此这里逻辑可能需要修改，因为掩码必须要求.png）
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png').replace('.TIF', '.png')  
        
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        return image, mask


def get_data_loaders(train_image_dir, train_mask_dir, val_image_dir=None, val_mask_dir=None,
                     batch_size=4, image_size=512, num_workers=2):
    train_dataset = LandslideDataset(train_image_dir, train_mask_dir, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = None
    if val_image_dir and val_mask_dir:
        val_dataset = LandslideDataset(val_image_dir, val_mask_dir, image_size=image_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader