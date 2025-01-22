import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import DataLoader, random_split
import random

torch.manual_seed(666)


def read_txt_to_list(file_path):
    """        self.cross_layer_1 = MultiHeadCrossAttention(model_dim=dim, num_heads=8, dropout=0.5)
    读取指定路径的 TXT 文档,并将每一行作为一个字符串存储在列表中。
    
    Args:
        file_path (str): TXT 文档的路径。
        
    Returns:
        list: 包含 TXT 文档每一行的字符串列表。
    """
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: {file_path} does not exist.")
        return []
    except IOError:
        print(f"Error: Unable to read {file_path}.")
        return []
    return lines



analy_1 = read_txt_to_list("/data4/zxf/dataset/pheme/pheme_analysis_1.txt")

analy_2 = read_txt_to_list("/data4/zxf/dataset/pheme/pheme_analysis_2.txt")




class ImageTextDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = []
        self.root_dir = root_dir


        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                img_id, text, label = row
                img_path = os.path.join(self.root_dir, img_id) + ".jpg"
                self.data.append((img_path, text, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, text, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
        return image, text, analy_1[idx], analy_2[idx], torch.tensor(label)



def custom_collate(batch):
    images, texts, data1, data2, labels = zip(*batch)
    
    # 保持 images 和 texts 为列表格式
    images = list(images)
    texts = list(texts)
    data1 = list(data1)
    data2 = list(data2)
    # 将 labels 堆叠成张量
    labels = torch.stack(labels, dim=0)
    
    return  texts, images, data1, data2, labels



# Usage example
def load_data_pheme(batch_size):
    all_dataset = ImageTextDataset('/data4/zxf/dataset/pheme/content.csv', '/data4/zxf/dataset/pheme/images/')

    # Create the train and test datasets
    train_dataset = torch.utils.data.Subset(all_dataset, [i for i in range(len(all_dataset)) if i % 8 != 0])
    test_dataset = torch.utils.data.Subset(all_dataset, [i for i in range(len(all_dataset)) if i % 8 == 0])
    
    # Define dataset split ratios
    train_ratio, dev_ratio, test_ratio = 70, 11, 19

    # Calculate dataset sizes
    total_size = len(all_dataset)
    train_size = int(total_size * train_ratio / 100)
    dev_size = int(total_size * dev_ratio / 100)

    # Create shuffled indices
    indices = list(range(total_size))
    random.shuffle(indices)

    # Split dataset based on calculated sizes
    train_indices = indices[:train_size]
    test_indices = indices[train_size + dev_size:]

    # Create the train, dev, and test datasets
    train_dataset = torch.utils.data.Subset(all_dataset, train_indices)

    test_dataset = torch.utils.data.Subset(all_dataset, test_indices)


    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return train_loader, test_loader



if __name__ == '__main__':
    train_loader, test_loader = load_data_pheme(batch_size = 1)

    for x1,x2,x3,x4,x5 in train_loader:
        print(x1)
        continue

