import csv
import numpy as np
import sentencepiece as spm
from nltk.translate.bleu_score import sentence_bleu
import random as rd
import time


def build_dataset(paths: list, output_path: str):
    data = []
    for path in paths :
        data = []
        with open(path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')  
            for row in reader:
                text = row[5]
                if not text.startswith('fr/lr'):
                    data.append(text)
                
        # Save the data in a text file
        with open(output_path, 'w', encoding='utf-8') as file:
            for item in data:
              file.write("%s\n" % item)

def train_model(data_path, vocab_size):
    spm.SentencePieceTrainer.train(input=data_path, model_prefix=f'./models/juritok_{vocab_size}', vocab_size=vocab_size)

def evaluate_model(model, data_path):
    with open(data_path, 'r', encoding='utf-8') as testing_data :
        data = testing_data.read()
        sp = spm.SentencePieceProcessor(model_file=model)
        t0 = time.time()
        encoded_data = sp.EncodeAsPieces(data.split('\n'))
        ti = time.time()
        decoded_data = sp.Decode(encoded_data)
        tf = time.time()

        with open(f"encoded_{model[17:-6]}.txt", 'w', encoding='utf-8') as file:
            file.write("%s\n" % encoded_data)
        with open(f"decoded_{model[17:-6]}.txt", 'w', encoding='utf-8') as file:
            for item in decoded_data:
                file.write("%s\n" % item)

        # with open(f"decoded_{model[17:-6]}.txt", 'r', encoding='utf-8') as file:
        #     decoded_data = file.read()
        
        return t0, ti, tf 
    


if __name__ == "__main__":

    paths = ["./data/jorf_2019.csv", "./data/jorf_2020.csv", "./data/jorf_2021.csv", "./data/jorf_2022.csv", "./data/jorf_2023.csv"]

    # paths_testing = rd.choices(paths, k=1)
    # paths_training = [path for path in paths if path not in paths_testing]
    # build_dataset(paths_training, "./data/training.txt")
    # build_dataset(paths_testing, "./data/testing.txt")
    
    vocab_sizes = [100, 500, 1000, 5000]
    # vocab_sizes = [1000]
    # for vocab_size in vocab_sizes :
    #     train_model("./data/training.txt", vocab_size)
    for vocab_size in vocab_sizes :
        t0, ti, tf = evaluate_model(f"./models/juritok_{vocab_size}.model", "./data/testing.txt")
        print(f'With vocab size = {vocab_size}, encoding duration = {ti - t0}, decoding duration = {tf - ti}')