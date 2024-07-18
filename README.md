# PersLLM: LLM Personified Training 

This is the source code for [PersLLM: A Personified Training Approach for Large Language Models](https://arxiv.org/abs/2407.12393). Currently, the code comments are not perfect, and we will continue to supplement them.

You can try our [online demo](http://label.shuzibeijing.cn:5173/). We will provide more personified agents and more functions in the future.

## Overview
![image](https://github.com/Ellenzzn/PersLLM/blob/main/figs/personified-main.png)

We propose a framework for LLM personified training, emphasizing the social practice, consistency, and dynamic development of the personality. We conduct experiments on the Harry Potter personified dataset. The multi-agent conversation and the human-agent interaction experiments also prove the effectiveness of personified training.

## Requirement

The model is implemented using PyTorch. The versions of main packages used are shown below.

- numpy==1.21.5
- torch==1.13.1
- transformers==4.30.2
- model-center==1.0.2
- bmtrain==1.0.0
  
To set up the dependencies, you can run the following command:
```
pip install -r requirements.txt
```

## Data Annotation

- You can directly download the annotated data for HP dataset from [Google Drive](https://drive.google.com/drive/folders/1DEliZQD_XU-Ev5eNDU_VgHjxNphqjzJE?usp=sharing). 

- If you want to conduct your own personified data, please first prepare two txt files separately containing the knowledge contents and the speech records for the target personality, with one paragraph in each line. 

- Go to `data_annotation/` and run `run.py`. Then follow the instructions step by step to conduct personified conversational tuning data. For example:

```
>> python run.py --QA_pattern thought --COT
您好！
请输入您的两个数据文件的相对路径(用空格分隔不同的文件路径): 
>> data/harry/early_knowledge.txt data/harry/early_style.txt
预处理后的数据已存入同一目录下。
请输入可用的openai api_key:
>> sk-xxxxxxx
请输入你的agent称呼：
>> Harry Potter
您期望以何种方式训练一个agent？1. 直接基于GPT-4的agent. 2. 训练一个模型做专用agent. 3. 标注风格化指令微调数据. 4. 标注auto-DPO数据. 请输入你的选项编号：
>> 2
```

- After the annotation process, you can run `python find_sim_ques.py` to deduplicate the annotated data items and divide them into training/test sets. 

- You can also run `find_sim_response.py` to get automatic DPO data from them.

## Personified Training

- Go to `training_codes/` if you want to train your own models. You have to prepare the backbone model in the [ModelCenter](https://github.com/OpenBMB/ModelCenter) format. You can also try to expand the word embeddings and the tokenizer vocab list to define your temporal labels. In our case, we add 3 special tokens ('<TIME-I>', '<TIME-II>' and '<TIME-III>').

- First go to `data/` to pre-process your annotated data. Define the input and output file path, the tokenizer, and the max length before you run `bash process.sh`.

- Go to `src` to conduct the training process. Define the related hyper-parameters and file paths before you run `bash train.sh`.

## Citation

Please cite our work if you like it.
```
@article{zeng2024persllm,
  title   = {PersLLM: A Personified Training Approach for Large Language Models},
  author  = {Zheni, Zeng and Jiayi, Chen and Huimin, Chen and Yukun, Yan and Yuxuan, Chen and Zhiyuan, Liu and Maosong, Sun},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2407.12393}
}
```
