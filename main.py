# importing required modules
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import pathlib
import warnings

warnings.filterwarnings('ignore')

# Training dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, X_train, y_train, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        label = self.y_train.iloc[index]

        image_path = f"{self.root_dir}/{self.X_train.iloc[index]}.jpg"

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)
        return image, torch.tensor(label)

def run():
    # Data transforms
    test_transforms = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((image_size, image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])

    test_dataset = CustomDataset(f'{base_path}/cil_task_data/test',
                                 X_final, y_final,
                                 test_transforms)

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # Loading the pretrained model here
    model = resnet50(pretrained = True)
    model._fc = nn.Sequential(
        nn.Linear(2048, 100),
        nn.ReLU(),
        nn.Linear(100, 6)
    )
        
    # loading the trained weights
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)

    # Testing loop
    label = np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            _, pred = torch.max(output, 1)
            label = np.concatenate((label, np.array(pred.cpu().data)), axis=0)

    # Preparing submission file
    sample_df['digit'] = label
    sample_df['digit'] = sample_df["digit_sum"].astype(int)
    sample_df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    # This helps make all other paths relative
    base_path = pathlib.Path().absolute()
    model_name = 'resnet50'
    image_size = 256
    BATCH_SIZE = 5
    num_workers = 2
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    root_dir = f"{base_path}/cil_task_data"

    # DEFINING hyperameters
    pretrained_model = f'{base_path}/model1.pth'
    sample_csv_path = f'{base_path}/test.csv'
    sample_df = pd.read_csv(sample_csv_path)

    X_final = sample_df['id']
    y_final = sample_df["digit"]
    run()