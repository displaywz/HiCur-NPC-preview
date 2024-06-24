import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from transformers import TextStreamer
from PIL import Image
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
import pandas as pd
image_size = 384
valid_trans = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean = [0.53017267, 0.31335013, 0.31556818], std = [0.33659712, 0.23721003, 0.24286168]),
    ToTensorV2()
])
sentences = "Please provide a description of this image."
def get_random_sentence(sentence_list):
    return random.choice(sentence_list)
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    print(model)
    df = pd.read_csv(args.df_path)
        
    if args.task_type == 'seg':
        for idx in tqdm(len(df)):
            row = df.iloc[idx]
            image_file = args.image_folder + row['image']
            qs = 'Please provide a detailed description of this image for disease area segmentation.'
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates["llama3"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            aug = valid_trans(image = image)
            image_tensor = aug['image']
            with torch.inference_mode():
                output_ids = model.forward(
                        input_ids,
                        output_hidden_states=True,
                        images=image_tensor.unsqueeze(0).cuda(),
                        image_sizes=[image.size],
                )
                llava_hidden = output_ids.hidden_states[-1]
                padding_size = 1024 - llava_hidden.shape[1]
                llava_hidden= torch.nn.functional.pad(llava_hidden, (0, 0, 0, padding_size))
                torch.save(llava_hidden, args.output_folder + row['image'].replace('.png','.pt'))
    elif args.task_type == 'pro':
        for idx in tqdm(len(df)):
            row = df.iloc[idx]
            qs = row['patient_description'] + 'Please provide a detailed description of this image for disease area segmentation.'
            conv = conv_templates["llama3"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
            output_ids = model.forward(
                    input_ids,
                    output_hidden_states=True)
            llava_hidden = output_ids.hidden_states[-1]
            padding_size = 1024 - llava_hidden.shape[1]
            llava_hidden= torch.nn.functional.pad(llava_hidden, (0, 0, 0, padding_size))
            torch.save(llava_hidden, args.output_folder + row['image'].replace('.png','.pt'))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./model/llava-hicur-cga-sft")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--df-path", type=str, default="../StageIII-FGF/data/seg-df.csv") 
    parser.add_argument("--task-type", type=str, default="seg") #seg (Segment) or pro (Prognosis)
    parser.add_argument("--image-folder", type=str, default="../StageIII-FGF/data/seg-images/")
    parser.add_argument("--output-folder", type=str, default="../StageIII-FGF/data/seg-images-cga-pt/")
    parser.add_argument("--conv-mode", type=str, default='llama3')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    eval_model(args)
