import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import os
from difflib import get_close_matches
import random
import re
import argparse
import gc
from summary import caculate_accuracy_mmad
 

def get_image_path(image_path):
    if image_path.split("/")[0] == "DS-MVTec":
        new_image_path = os.path.join("MVTecAD", image_path.split("/")[1], "test", image_path.split("/")[-2], image_path.split("/")[-1])
    elif image_path.split("/")[0] == "VisA":
        if image_path.split("/")[-2] == "bad": temp_flag="Anomaly"
        elif image_path.split("/")[-2] == "good": temp_flag="Normal"
        new_image_path = os.path.join("ViSA", image_path.split("/")[1], "Data", "Images", temp_flag, image_path.split("/")[-1])
    else:new_image_path=image_path
    new_image_path = os.path.join("./eval/dataset", new_image_path)
    return new_image_path

def load_json(json_path):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())
    return anno

def write_json(json_path,data):
    with open(json_path, 'w') as f:
        json.dump(data, f)

def parse_answer(response_text, options=None):
        pattern = re.compile(r'\b([A-E])\b')
        answers = pattern.findall(response_text)
        flag = False

        if len(answers) == 0 and options is not None:
            flag = True
            options_values = list(options.values())
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                closest_match = closest_matches[0]
                for key, value in options.items():
                    if value == closest_match:
                        answers.append(key)
                        break
        return answers, flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="OpenBMB/MiniCPM-V-2_6")
    args = parser.parse_args()

    torch.manual_seed(0)
    
    model = AutoModel.from_pretrained(args.ckpt, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    answer_list=[]
    ANSWER_REQ="Answer with the option's letter from the given choices directly."
    evaluate_info=load_json("./mmad.json")
    print(len(evaluate_info))

    for ori_image_path, querys in tqdm(evaluate_info.items()):
        for query in querys["conversation"]:
            options="\n"
            for k, v in query["Options"].items(): options += k + ". " + v + "\n" 
            prompt = query["Question"] + options + ANSWER_REQ

            image_path = get_image_path(ori_image_path)
            image = Image.open(image_path).convert('RGB')

            # First round chat 
            question = prompt#"Tell me the model of this aircraft."
            msgs = [{'role': 'user', 'content': [image, question]}]

            answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer
            )
            # print(answer)

            gpt_answer, flag=parse_answer(answer, options=query["Options"])
            answer_list.append(
                {
                    "image": ori_image_path,
                    "question_type": query["type"],
                    "correct_answer": query["Answer"],
                    "gpt_answer": gpt_answer[0],
                    "answer_error": flag
                }
            )

    write_json("answer.json", answer_list)
    caculate_accuracy_mmad("answer.json")


if __name__=="__main__":
    main()