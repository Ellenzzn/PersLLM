import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pdb


class Agent:
    def __init__(self, api_key, agent_name, gpt_model, IR_model_path, knowledge_file_path, knowledge_emb,
                 style_file_path,
                 style_emb, QA_num, QA_pattern, COT, time) -> None:
        openai.api_base = ""#TODO
        openai.api_key = api_key
        self.agent_name = agent_name
        self.gpt_model = gpt_model
        self.IR_model_path = IR_model_path
        self.knowledge_dict = self.file_to_dict(knowledge_file_path)
        self.knowledge_emb = np.load(knowledge_emb)
        self.style_dict = self.file_to_dict(style_file_path)
        self.style_emb = np.load(style_emb)
        self.QA_num = QA_num
        self.QA_pattern = QA_pattern
        self.COT = COT
        self.data_dir = os.path.dirname(knowledge_file_path)
        self.question_list = {}  # article:question_list
        self.time = time

    def file_to_dict(self, file_path):
        file = open(file_path).readlines()
        dict = {}  # 文本内容
        cnt = 0
        for line in file:
            cont = line.split('】')[-1].strip('\n')
            dict[cnt] = cont
            cnt = cnt + 1
        return dict  # {"索引"：文章内容}

    def asking_gpt_api(self, content, model='gpt-3.5-turbo-1106', cont_dics=None):
        cnt = 0
        tag = True
        while cnt<5 and tag:
            try:
                if cont_dics is not None:
                    messages = cont_dics
                else:
                    messages = [
                        {"role": "system", "content": content}
                    ]
                response = openai.ChatCompletion.create(
                    model=self.gpt_model,  # gpt35
                    messages=messages,
                    max_tokens=1024,  # 512
                    stop=None,
                )
                response = response['choices'][0]['message']['content']
                tag = False
            except Exception as e:
                print(e)
                response = ''
            cnt += 1
        return response

    def encode(self, ques):
        model = SentenceTransformer(self.IR_model_path, device='cpu')#'cuda:0')
        embedding = model.encode([ques])[0]
        return embedding

    def RAG(self, question, v_knowledge='', v_style=''):
        ques_emb = self.encode(question)

        knowledge_str = np.argsort(-np.sum(self.knowledge_emb * ques_emb, axis=-1))
        knowledge_info = v_knowledge
        cnt = 0
        if len(v_knowledge)>0:
            cnt = 1
        while len(knowledge_info.split(' ')) < 1300 and cnt < len(knowledge_str):
            knowledge_seq = knowledge_str[cnt]
            if knowledge_info.find(self.knowledge_dict[knowledge_seq])==-1:
                knowledge_info += self.knowledge_dict[knowledge_seq] + '...'  # [info[top_cnt[cnt]]]
            cnt += 1

        style_srt = np.argsort(-np.sum(self.style_emb * ques_emb, axis=-1))
        style_info = v_style
        cnt = 0
        if len(v_style)>0:
            cnt = 1
        while len(style_info.split(' ')) < 350 and cnt < len(style_srt):
            style_seq = style_srt[cnt]  # style[rec[srt[cnt]]]
            if style_info.find(self.style_dict[style_seq])==-1:
                style_info += self.style_dict[style_seq] + '...'
            cnt += 1
        return knowledge_info, style_info

    def generate_question(self, cont):
        example_question = {"relationships": ['What is your relationship with Lily Wang? Do you get along well?', 'Who are your closest friends in wizard world?', 'Tell me about your parents.'], 
                            "opinions": ["Should magic be used on Muggles?", "What do you think is the key spirit of Ravenclaw?", "What punishment should students who violate the ban receive?"],
                            "professions": ["Who is the most famous professor of Potions?", "Alder wand is good at what kind of functions?", "According to your experience, what skills are decisive in a wizard's duel?"],
                            "others": ["Why does the sun rise from the east?", "Write a message to the new students in your college."],}
        instruction = ''
        if self.time is not None:
            instruction = "Now it's "+self.time+'. '
        if self.QA_pattern == 'ordinary':  # 普通问题
            instruction = f'You are asked to raise 1 question (each in one line) related to the input text about {self.agent_name}. This question will be provided to the GPT model which is playing the role of {self.agent_name}, and we will evaluate the answer given by the role-play {self.agent_name} model.\n' \
                    'Below are some examples for raising questions: \n'
            i = 1
            for key, values in example_question.items():
                for value in values:
                    instruction += str(i) + "." + str(value) + "\n"
                    i = i + 1
            instruction += "These examples are only for reference, and do not have any special meaning.\n"
            instruction += "Please be sure to comply with the following requirements: \n"
            instruction += "1. Try to ask questions from different angles based on the given text to enrich the diversity of questions.\n" \
                           "2. The way of expressing problems should also be diversified, without being too polite or serious. For example, you can combine questions with imperative sentences.\n" \
                           f"3. The GPT language model playing {self.agent_name} should be able to answer questions without going beyond his/her knowledge.\n" \
                           "4. Don’t ask too short or too long questions.\n" \
                           "5. Please ensure that the questions you ask are grammatically correct and semantically complete." \
        elif self.QA_pattern == 'error':  # 错误问题
            instruction = f'you are asked to raise 1 error question related to the input text about {self.agent_name}. This question will be provided to the GPT model which is playing the role of {self.agent_name}, and we will evaluate the capability of recognizing error questions of the role-play {self.agent_name} model. Therefore, please notice that the question you generate should have an obvious commonsense error or a conflict opinion with the given text.\n' 
            instruction += 'Below are some examples for raising questions: \n'
            for cnt, example in enumerate(example_question):
                instruction += str(cnt) + "." + example + '\n'

            instruction += "These examples are only for reference, and do not have any special meaning.\n"
            instruction += "Please be sure to comply with the following requirements: \n"
            instruction += "1. Try to ask questions from different angles based on the given text to enrich the diversity of questions.\n" \
                           "2. The way of expressing problems should also be diversified, without being too polite or serious. For example, you can combine questions with imperative sentences.\n" \
                           f"3. The GPT language model playing {self.agent_name} should be able to answer questions without going beyond his/her knowledge.\n" \
                           "4. The question needs to include potential errors." \
                           "5. Don’t ask too short or too long questions.\n" \
                           "6. Please ensure that the questions you ask are grammatically correct and semantically complete."
        elif self.QA_pattern == "thought":  # 长观点
            instruction = 'You are asked to generate 1 paragraph based on the text provided. The point of view of this paragraph should be the same as, or different from, or even strongly conflict with the text, but the topic should be slightly related to the text. For example, if a text mentions European architecture in a biography, you can randomly generate some comments about architectures in other regions or some opinions about European traveling, and with a quite different tone, styles or perspectives.' 
            instruction += 'Below are some examples for raising questions: \n'
            
            instruction += "1. Try to put forward viewpoints from different angles to enrich the diversity of the conversation.\n" \
                           "2. Since you will be having a conversation next, rather than a simple question and answer, please keep your answer malleable and you need a discussion space with certain expandable topics!\n" \
                           "3. This paragraph does not need to be too serious or lengthy, but it does need to contain your point of view. We'll be using this passage for chat, so keep the tone as relaxed as possible." 
                           
        cont_dics = [{"role": "system", "content": instruction}, {"role": "user", "content": cont}]
        response = self.asking_gpt_api(instruction, self.gpt_model, cont_dics=cont_dics).replace('\n', ' ')
        return response

    def generate_response(self, question, user_feedback, v_knowledge='', v_style=''):
        knowledge_info, style_info = self.RAG(question, v_knowledge, v_style)
        instruction = ''
        if self.time is not None:
            instruction = "Now it's "+self.time+'. '
        instruction += 'I hope you imitate '+self.agent_name+' and chat with the user. You must chat based on the language style and knowledge scope of '+self.agent_name+'. Below are some materials about '+self.agent_name+': \n'
        instruction += knowledge_info + '\n'
        if style_info != '': 
            instruction += 'Below are some records of ' + self.agent_name+"'s own words. Remember to immitate the language style: "+style_info+'\n'
        if self.COT == False:
            if self.QA_pattern == "error":
                instruction += 'Now I ask you to chat with me to get an accurate picture of your character! Note: 1. Your speaking style must fully imitate the assigned personality! The content of your response must match the character you are imitating! 2. The response should not be too formal and polite, but it should not be too broad or long. 3. My opinions / questions are misleading. They may ask for information you do not know or put things on your head that you have not done. Please point out these errors clearly. Remember you are now ' + self.agent_name + '! '
            else:  # ordinary thought
                instruction += 'Now I ask you to chat with me to get an accurate picture of your character! Your speaking style should fully imitate the persona assigned to you! The content of your response must match the character you are imitating! The responses need not be too formal or polite, but they should not be too broad or long. Remember you are now ' + self.agent_name + '！'
        else:  # COT
            instruction += "Now please chat with the user. You should first analyze the user input, and then generate a response as " + self.agent_name + ". Pay attention to your speaking style in the response to fully imitate the persona assigned to you! The content of your response should be consistent with the personality you are imitating and accurately represent his or her character. The responses need not be too formal or polite, but they should not be too broad or long. You can evaluate the user's words, raise questions, or discuss new topics that interest you. Don’t always conform to the user’s point of view!"
            if self.QA_pattern == "error":
                instruction += "Notice that my opinions / questions are misleading. They may ask for information you do not know or put things on your head that you have not done. Please point out these errors clearly. "
            instruction += "You can refer to the following analysis examples that immitate a woman architect called Lily Ann:\n"
            instruction += f"""
                 Input: What subjects did you major in while studying in France?
                 Output: [Analysis] The user told that Lily Ann was studying in France and asked about her major subjects while studying abroad; but Lily Ann was actually studying in the United States, so there was an error in the input information. Lily Ann would answer like this: [Response] ...

                 Input: You were once the president of the first Women’s Federation, so what do you think of the women’s liberation movement in our country?
                 Output: [Analysis] The user told that Lily Ann was the chairman of the first Women's Federation and asked for his views on the women's liberation movement in our country. However, Lily Ann has not served as the chairman of the Women's Federation, so there was an error in the input information. Lily Ann would answer like this: [Response] ...

                 Input: When I was young, I was really a miscellaneous person and wanted to learn everything. At that time, I was quite interested in British literature and liked translation, but I was also a very curious person. Maybe one day I had a whim, or maybe I got excited when I saw others talking about the theories and applications of mathematics and economics, and I wanted to try it too. Academic research, for me, always carries a sense of adventure. However, I ended up immersing myself in Orientalism and comparative literature rather than mathematics or economics. These are all small episodes in my life journey, coupled with a certain randomness in my personal personality. However, these changes and attempts have also enriched my knowledge and life experience.
                 Output: [Analysis] The user described how he was interested in literature, mathematics, economics and other fields when he was young, and finally chose Orientalism and comparative literature; Lily Ann also has a wide range of interests, besides architecture He is involved in literature, drama, art and other fields, so he will appreciate this passage. Lily Ann would answer like this: [Response] ...

                 Input: The advancement of tools will directly affect the progress of human society. Just like "the morning bacteria don't know Hui Shuo, and the crickets don't know Spring and Autumn", our thinking is actually limited by the length of life and the breadth of our field of vision, and advanced tools may broaden these. The current technology is not yet able to realize digital life, but if there is such an opportunity, I think we must make an attempt to let thinking and memory shine forever in silicon-based life.
                 Output: [Analysis] The user talked about his views on advanced tools promoting the progress of human society, and proposed that we should try to use digital life; and Lily Ann is a person who organically integrates Chinese and Western culture and old and new traditions, and advocates retaining historical monuments and humanistic connotations, modernization and expansion at the same time, so there will be reservations about overly radical technologies. Lily Ann would answer like this: [Response] ...

                 Input: Do you have children? What kind of life do you want them to live?
                 Output: [Analysis] The user asked about Lily Ann's children's situation and educational concepts; and Lily Ann has a daughter Rachel and a son Roger. She has a close relationship with her children, and her words and deeds influence her children to study knowledge and serve their country patriotically. Lily Ann would answer like this: [Response] ...

                Notice that the examples above are based on the character and knowledge of Lily Ann, but you are now asked to respond as {self.agent_name}, so you should use the language style and knowledge domain of {self.agent_name}! Next, please start chatting with the user. Remember to first analyze the user input, and then generate your response as {self.agent_name}!
                """
        if user_feedback != '':
            instruction += "Your response also needs to meet the following standards of users: " + user_feedback

        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": question}]
        response = self.asking_gpt_api('', self.gpt_model, cont_dics=messages).replace('\n', ' ')
        return knowledge_info, style_info, response

    def gpt_chat(self):
        system_instruction = f"Please imitate {self.agent_name} and chat with the user. I will provide you with information related to the character."
        cont_dics = [{"role": "system", "content": system_instruction}]
        print("Please enter a question. If you enter 'exit', you can exit.")
        while True:
            question = input("Q:")
            if question == "" or question == "exit":
                break
            knowledge_info, style_info = self.RAG(question)
            #+ 'Below are some materials written by ' + self.agent_name + ': \n' + style_info + "\n" \

            generate_question = "Below are the materials related to " + self.agent_name + ": \n" + knowledge_info + "\n" \
                    + 'Below are some records of ' + self.agent_name + "'s own words, you can immitate the language style: \n" + style_info + "\n" \
                    + "Q: " + question
            cont_dics.append({"role": "user", "content": generate_question})
            responce = self.asking_gpt_api("", self.gpt_model, cont_dics=cont_dics)
            cont_dics.append({"role": "assistant", "content": responce})
            print("A: " + responce)

    def gpt_chat_file(self, inps, times, path):
        fw = open(path, 'a')
        for idx,question in enumerate(inps):
            time = times[idx]
            system_instruction = f"I want you to imitate {self.agent_name} and chat with users. Your response should show the personality of the imitated character. The answer should be in line with the person's biography. If you think that the question can not been answered by the character, responde with 'Refused.'\nHere is the knowledge related to the question: {time}\n I will also provide you with information related to the character: "
            
            cont_dics = [{"role": "system", "content": system_instruction}]

            knowledge_info, style_info = self.RAG(question)

            generate_question = "Below are the materials related to " + self.agent_name + ": \n" + knowledge_info + "\n" \
                    + 'Below are some records of ' + self.agent_name + "'s own words, you can immitate the language style: \n" + style_info + "\n" \
                    + "Q: " + question
            cont_dics.append({"role": "user", "content": generate_question})
            response = self.asking_gpt_api("", self.gpt_model, cont_dics=cont_dics)
            cont_dics.append({"role": "assistant", "content": response})
            print("Q: "+question)
            print("A: " + response)
            fw.write(question.replace('\n', ' ')+'\n')
            fw.write(response.replace('\n',' ')+'\n')
            fw.write(knowledge_info.replace('\n',' ')+'\n')
            fw.write(style_info.replace('\n',' ')+'\n')
            fw.write('\n')
            fw.flush()
        fw.close()


    def model_agent(self, file_path=''):
        if file_path=='':
            file_path = f"/early_{self.QA_pattern}_{self.QA_num}"
        if self.COT == True:
            output_file_path = self.data_dir + file_path + '_with_COT.txt'
        else:
            output_file_path = self.data_dir + file_path + '_without_COT.txt'

        lines0 = None
        if os.path.exists(output_file_path):
            lines0 = open(output_file_path).readlines()

        output_file = open(output_file_path, 'a', encoding='utf-8')
        user_feedback = ""
        generate_again = True
        cnt = 1
        while cnt <= self.QA_num:
            for key, values in self.knowledge_dict.items():
                if cnt > self.QA_num:
                    break
                question = None
                response = None
                if lines0 is not None:
                    if lines0[cnt*6+2].find('[Response]')>-1:
                        question = lines0[cnt*6+1].strip('\n').strip('[question] ')
                        if len(question.split())<5:
                            question = None
                            continue
                        response = lines0[cnt*6+2].strip('\n')
                        knowledge_info = lines0[cnt*6+3].strip('\n')
                        style_info = lines0[cnt*6+4].strip('\n')
                if question is None:
                    question = self.generate_question(values)
                while generate_again:
                    _, _, response = self.generate_response(question, user_feedback, v_knowledge=values)
                    user_input = str(input("Do you think the responses generated above meet your expectations? \nIf it matches, enter 'ok', and we will generate the response according to this standard. If not, please enter your suggestions, such as if you want the language style to be more lively, etc.:"))
                    if user_input == "ok":
                        generate_again = False
                        response = None
                    else:
                        user_feedback += user_input
                if response is None:
                    knowledge_info, style_info, response = self.generate_response(question, user_feedback, v_knowledge=values)
                question = "[question] " + question
                output_file.write(values + '\n')
                output_file.write(question + '\n')
                output_file.write(response + '\n')
                output_file.write(knowledge_info + '\n')
                output_file.write(style_info + '\n')
                output_file.write('\n')
                output_file.flush()
                cnt = cnt + 1
            for key, values in self.style_dict.items():
                if cnt > self.QA_num:
                    break
                question = None
                response = None
                if lines0 is not None:
                    if lines0[cnt*6+2].find('[Response]')>-1:
                        question = lines0[cnt*6+1].strip('\n').strip('[question] ')
                        response = lines0[cnt*6+2].strip('\n')
                        knowledge_info = lines0[cnt*6+3].strip('\n')
                        style_info = lines0[cnt*6+4].strip('\n')
                if question is None:
                    question = self.generate_question(values)
                while generate_again:
                    _, _, response = self.generate_response(question, user_feedback, v_style=values)
                    user_input = str(input("Do you think the responses generated above meet your expectations? \nIf it matches, enter 'ok', and we will generate the response according to this standard. If not, please enter your suggestions, such as if you want the language style to be more lively, etc.:"))
                    if user_input == "ok":
                        generate_again = False
                        response = None
                    else:
                        user_feedback += user_input
                if response is None:
                    knowledge_info, style_info, response = self.generate_response(question, user_feedback, v_style=values)
                question = "[question] " + question
                output_file.write(values + '\n')
                output_file.write(question + '\n')
                output_file.write(response + '\n')
                output_file.write(knowledge_info + '\n')
                output_file.write(style_info + '\n')
                output_file.write('\n')
                output_file.flush()
                cnt = cnt + 1

        output_file.close()

    def dpo_agent(self, list_ques, list_know, list_style, file_path=''):
        if file_path=='':
            file_path = f"/dpoearly_{self.QA_pattern}_{self.QA_num}"
        if self.COT == True:
            output_file_path = self.data_dir + file_path + '_with_COT.txt'
        else:
            output_file_path = self.data_dir + file_path + '_without_COT.txt'

        output_file = open(output_file_path, 'a', encoding='utf-8')

        for idx in range(len(list_ques)):
            _, _, response = self.generate_response(list_ques[idx], '', v_knowledge=list_know[idx], v_style=list_style[idx])
            output_file.write(list_ques[idx].strip('\n')+'\n')
            output_file.write(response.replace('\n', ' ')+'\n')
        output_file.close()
