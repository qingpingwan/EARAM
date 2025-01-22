
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel



from sklearn.metrics import precision_recall_fscore_support


devices = "cuda:3"

clip_model = CLIPModel.from_pretrained("/data4/zxf/hf/openai/clip-vit-large-patch14").to(devices)
processor = CLIPProcessor.from_pretrained("/data4/zxf/hf/openai/clip-vit-large-patch14")


class VisualProjection(nn.Module):
    def __init__(self, visual_projection):
        super().__init__()
        self.visual_projection = visual_projection

    def forward(self, x):
        """
        (batch, len, 768) -> (batch, len, 768)
        """
        x = self.visual_projection(x)
        return x


class TextProjection(nn.Module):
    def __init__(self, visual_projection):
        super().__init__()
        self.visual_projection = visual_projection

    def forward(self, x):
        """
        (batch, len, 768) -> (batch, len, 768)
        """
        x = self.visual_projection(x)
        
        return x
    

visual_projection = clip_model.visual_projection
Text_projection = clip_model.text_projection


#  VisualProjection
Visual_module = VisualProjection(visual_projection).to(devices)
# input_tensor_1 = outputs["vision_model_output"]["last_hidden_state"]
# Visual_output_tensor = Visual_module(input_tensor_1)



Text_module = TextProjection(Text_projection).to(devices)
# input_tensor_2 = outputs["text_model_output"]["last_hidden_state"]
# Text_output_tensor = Text_module(input_tensor_2)

# print(Text_output_tensor.size())
# print(Visual_output_tensor.size())




def CLIP_pipeline(x1,x2,x3,x4):
    tmp_inputs_text = processor(text=x1, return_tensors="pt", padding=True,truncation=True, max_length=77).to(devices)
    tmp_inputs_image = processor(images=x2, return_tensors="pt").to(devices)
    tmp_inputs_analy1= processor(text=x3, return_tensors="pt", padding=True,truncation=True, max_length=77).to(devices)
    tmp_inputs_analy2 = processor(text=x4, return_tensors="pt", padding=True,truncation=True, max_length=77).to(devices)
    
    outputs_1 = clip_model.text_model(**tmp_inputs_text)
    outputs_2 = clip_model.vision_model(**tmp_inputs_image)
    outputs_tensor_1 = outputs_1["last_hidden_state"]
    Text_output_tensor = Text_module(outputs_tensor_1)
    outputs_tensor_2 = outputs_2["last_hidden_state"]
    Visual_output_tensor = Visual_module(outputs_tensor_2)
    outputs_3 = clip_model.text_model(**tmp_inputs_analy1)["last_hidden_state"]
    outputs_4 = clip_model.text_model(**tmp_inputs_analy2)["last_hidden_state"]
    outputs_3 = Text_module(outputs_3)
    outputs_4 = Text_module(outputs_4)
    
    
    return Text_output_tensor, Visual_output_tensor, outputs_3, outputs_4

# x1 = ["a photo of a cat", "a photo of a dog"]
# x2 = [Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw),
#       Image.open(requests.get("http://images.cocodataset.org/val2017/000000397133.jpg", stream=True).raw)]

# x1,x2 = CLIP_pipeline(x1,x2)

# print(x1.size())
# print(x2.size())



def classification_metrics(predicted, labels):
    """
    predicted (torch.Tensor): (batch_size,)
    labels (torch.Tensor):(batch_size,)
    """
    

    y_pred = predicted.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    print("true news:")
    print(f"Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1-score={f1[0]:.4f}")
    
    print("fake news:")
    print(f"Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1-score={f1[1]:.4f}")