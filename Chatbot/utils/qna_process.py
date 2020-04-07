import numpy as np
import random
import bz2
import os

data_dir = os.curdir + '/reddit_parse/output/uncompressed'


with open(os.curdir + "/data/conversations.csv", "w", encoding='utf-8') as text_file:

    for file in os.listdir(data_dir):
        data = open(data_dir + '/' + file, mode='r', encoding='utf-8')
        print("Processing file: {}".format(file))
        question = data.readline()
        i = 1

        for line in data:
            if question != '\n':
                answer = line

                if answer != '\n':
                    question = question.rstrip()
                    answer = answer.rstrip()
                    output = question[2:] + '**|systemvaz|**<START>' + answer[2:] + '<END>\n'
                    text_file.write(output)
                    question = answer

                else:
                    try:
                        question = next(data)
                    except:
                        print('EOF')

            i = i+1
        
        print('Total lines processed: {}'.format(i))

text_file.close
