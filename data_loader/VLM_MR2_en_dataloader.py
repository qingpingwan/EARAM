import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader



def read_txt_to_list(file_path):

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


analy_1_en_train = read_txt_to_list("./data_3_MR2/MR2_en_train_analysis_1.txt")

analy_2_en_train = read_txt_to_list("./data_3_MR2/MR2_en_train_analysis_2.txt")

analy_1_en_test = read_txt_to_list("./data_3_MR2/MR2_en_test_analysis_1.txt")

analy_2_en_test = read_txt_to_list("./data_3_MR2/MR2_en_test_analysis_2.txt")



class ImageCaptionDataset(Dataset):
    def __init__(self, json_file, img_dir, analy_1, analy_2):

        self.data = self.load_data(json_file)
        self.img_dir = img_dir
        self.analy_1 = analy_1
        self.analy_2 = analy_2

        
    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        key = list(self.data.keys())[idx]
        item = self.data[key]
        
        caption = item['caption']
        img_path = os.path.join(self.img_dir, item['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = item['label']
        
        return caption, image, self.analy_1[idx], self.analy_2[idx], torch.tensor(label)
    
    
def custom_collate(batch):
    texts, images, data1, data2, labels = zip(*batch)
    

    images = list(images)
    texts = list(texts)
    data1 = list(data1)
    data2 = list(data2)

    labels = torch.stack(labels, dim=0)
    
    return  texts, images, data1, data2, labels


# Usage example
def load_test_MR2(batch_size):
    all_dataset = ImageCaptionDataset('./data_3_MR2/dataset_merge/en_test.json', './data_3_MR2/dataset_merge', analy_1_en_test, analy_2_en_test)

    # Create the data loaders
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return test_loader

def load_train_MR2(batch_size):
    all_dataset = ImageCaptionDataset('./data_3_MR2/dataset_merge/en_train.json', './data_3_MR2/dataset_merge',analy_1_en_train,analy_2_en_train)

    # Create the data loaders
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return test_loader


if __name__ == "__main__":
    dataload = load_test_MR2(1)
    for x1,x2,x3,x4,x5 in dataload:
        print(x3)
        break
    