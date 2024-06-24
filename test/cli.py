from model.moca import MoCANet
from model.llaval3.model.builder import load_pretrained_model
from model.llaval3.mm_utils import tokenizer_image_token, get_model_name_from_path
from model.llaval3.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llaval3.conversation import conv_templates
import torch
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

vit_size = 384
seg_size = 256
vit_trans = A.Compose([
    A.Resize(vit_size, vit_size),
    A.Normalize(mean = [0.53017267, 0.31335013, 0.31556818], std = [0.33659712, 0.23721003, 0.24286168]),
    ToTensorV2()
])
seg_trans = A.Compose([
    A.Resize(seg_size, seg_size),
    A.Normalize(mean = [0.53017267, 0.31335013, 0.31556818], std = [0.33659712, 0.23721003, 0.24286168]),
    ToTensorV2()
])
def pad_or_truncate(tensor, max_length=32,pad_size=256):
    current_length = tensor.shape[0]
    
    if current_length > max_length:
        return tensor[:max_length]
    else:
        padding = torch.zeros((max_length - current_length, 3,pad_size,pad_size), device=tensor.device)
        return torch.cat((tensor, padding), dim=0)
class MoCALLaVAl3(torch.nn.Module):
    def __init__(self):
        super(MoCALLaVAl3,self).__init__()
        self.mocanet = MoCANet()
        llm_model_path = os.path.expanduser('./model/hicur-npc-llama3')
        model_name = get_model_name_from_path(llm_model_path)
        tokenizer, model, _, _ = load_pretrained_model(llm_model_path, None, model_name)
        self.llava = model
        self.tokenizer = tokenizer
        self.router_model = SentenceTransformer('sentence-transformers_distilbert-base-nli-stsb-mean-tokens')
        self.map = pd.read_csv('./data/mapwithlabel.csv')
    
    def route_select(self,input_sentence):
        sentences_list = self.map.content.to_list()
        sentence_embeddings = self.router_model.encode(sentences_list + [input_sentence])
        input_embedding = sentence_embeddings[-1]
        cosine_scores = np.dot(sentence_embeddings, input_embedding) / (np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(input_embedding))
        most_similar_index = np.argmax(cosine_scores[:-1])
        most_similar_sentence = sentences_list[most_similar_index]
        print(most_similar_sentence)
        similarity_score = cosine_scores[most_similar_index]
        if similarity_score > 0.6:
            return self.map.label.iloc[most_similar_index]
        return 0
    def forward(self,x):
        with torch.inference_mode():
            q = x['q']
            map_idx = self.route_select(q)
            if map_idx == 0:
                image_file = x['img']
                qs = DEFAULT_IMAGE_TOKEN + '\n' + q
                conv = conv_templates["llama3"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                image = cv2.imread(image_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                aug = vit_trans(image = image)
                image_tensor = aug['image']
                output_ids = self.llava.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).cuda(),
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512,
                    use_cache=True)
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                return outputs
            elif map_idx == 1:
                image_file = x['img']
                qs = DEFAULT_IMAGE_TOKEN + '\n' + 'Please provide a detailed description of this image for disease area segmentation.'
                conv = conv_templates["llama3"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                image = cv2.imread(image_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                aug = vit_trans(image = image)
                image_tensor = aug['image']
                output_ids = self.llava.forward(
                    input_ids,
                    output_hidden_states=True,
                    images=image_tensor.unsqueeze(0).cuda(),
                    image_sizes=[image.size],
                )
                llava_hidden = (output_ids.hidden_states[-2]*0.3+output_ids.hidden_states[-1]* 0.7)
                padding_size = 1024 - llava_hidden.shape[1]
                llava_hidden= torch.nn.functional.pad(llava_hidden, (0, 0, 0, padding_size)).squeeze(dim=0)
                aug1 = seg_trans(image = image)
                seg_tensor = aug1['image']
                seg = self.mocanet.get_seg(seg_tensor.unsqueeze(dim=0),llava_hidden)
                output_ids = self.llava.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).cuda(),
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512,
                    use_cache=True)
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                return seg, outputs
            else:
                q = x['des']
                if q == '':
                    return 'Please input the information about the patient.'
                conv = conv_templates["llama3"].copy()
                conv.append_message(conv.roles[0], q + "Please help analyze the patient's prognosis")
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = self.llava.forward(
                    input_ids,
                    output_hidden_states=True)
                llava_hidden = output_ids.hidden_states[-1]
                padding_size = 1024 - llava_hidden.shape[1]
                llava_hidden= torch.nn.functional.pad(llava_hidden, (0, 0, 0, padding_size))
                imgs_timm = []
                files = os.listdir(x['file'])
                for file in files:
                    img = cv2.imread(x['file']+'/'+file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    aug1 = seg_trans(image = img)
                    image_tensor1 = aug1['image']
                    imgs_timm.append(image_tensor1)
                xx = {}
                xx['imgs_timm'] = pad_or_truncate(torch.stack(imgs_timm)).unsqueeze(dim=0).cpu()
                xx['llava'] = llava_hidden.squeeze(dim=0).float().cpu()
                return self.mocanet.get_pre(xx)

model = MoCALLaVAl3()
x = {}
x['q'] = "Predicting the patient's survival situation"
x['file'] = "./data/img/275043"
x['des'] = "患者信息:\n    - 性别: ..."
model(x)
                