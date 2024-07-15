import argparse
import pdb
from utils.passage_embedding import get_embedding
from agent import Agent
import json


def main(args):
    
    print("您好！")
    print("请您在data目录下新建一个以agent英文(或拼音)名字命名的文件夹，并将您的数据文件放置在该目录下。")
    print("注意，您需要整理出两个数据文件，一个作为知识库文件，将其命名为knowledge.txt，另一个作为风格模仿文件，将其命名为style.txt")
    file_path_list = input(
        "请输入您的两个数据文件的相对路径(用空格分隔不同的文件路径)，例如data/linhuiyin/knowledge.txt data/linhuiyin/style.txt：").split()
    knowledge_file_path, style_file_path = file_path_list[0], file_path_list[1]
    emb_file_list = get_embedding(file_path_list, args.IR_model)
    knowledge_emb_path, style_emb_path = emb_file_list[0], emb_file_list[1]
    print("预处理后的数据已存入同一目录下。")
    api_key = str(input("请输入可用的openai api_key:"))
    agent_name = str(input("请输入你的agent称呼："))
    
    scholar_agent = Agent(api_key, agent_name, args.gpt_model, args.IR_model, knowledge_file_path, knowledge_emb_path,
                          style_file_path,
                          style_emb_path, args.QA, args.QA_pattern, args.COT, args.time)

        

    print("您期望以何种方式训练一个agent？")
    print("1. 直接基于GPT-4的agent.")
    print("2. 训练一个模型做专用agent.")
    print("3. 标注风格化指令微调数据.")
    print("4. 标注auto-DPO数据.")
    num = int(input("请输入你的选项编号："))
    if num == 1:
        scholar_agent.gpt_chat()
    elif num == 2:
        scholar_agent.model_agent(file_path=args.file_path)
    elif num == 3:
        scholar_agent.gpt_chat_file(inps, times, f'data/{agent_name}/{agent_name}_role.txt')
        inps = []
        times = []
        lines = open(args.role_data).readlines()
        for line in lines:
            js = json.loads(line)
            inp = js['data'][0]
            inps.append(inp)
            times.append(js['data'][1])
    elif num == 4:
        lines = open(f'data/{agent_name}/'+args.dpo_data).readlines()
        list_ques = []
        list_know = []
        list_style = []
        for idx in range(int(len(lines)/5)):
            list_ques.append(lines[5*idx])
            list_know.append(lines[5*idx+2])
            list_style.append(lines[5*idx+3])
        scholar_agent.dpo_agent(list_ques, list_know, list_style, file_path=args.file_path)
    else:
        print("输入不合法!")
               

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")

    parser.add_argument("--IR_model", type=str, default="IR-model/gtr-base")
    parser.add_argument("--gpt_model", type=str, default="gpt-4-turbo")
    parser.add_argument("--QA", type=int, default=50)
    parser.add_argument("--QA_pattern", type=str, default="thought")
    parser.add_argument('--COT', action='store_true')
    parser.add_argument('--time', default=None)
    parser.add_argument('--role_data', defaut='data/mix_ultrachat.json')
    parser.add_argument('--dpo_data', default='dpo_lateques_for_early.txt')
    parser.add_argument('--file_path', default='')


    args = parser.parse_args()

    main(args)
