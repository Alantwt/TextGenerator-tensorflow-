from keras.models import load_model
import os
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import text_generate_model as tgm
import numpy as np
from data import data


def main():
    while True:
        tgm.make_sequence()
        text,len_words = user_input()
        print(prediction(text=text,len_next_words=int(len_words)))



def user_input():
    while True:
        try:
            len_words = int(input("\nIntroduce cuantas palabras quieres de respuesta: "))
            break
        except:
            print("numero incorrecto!!!")

    text = input("\nEscribe...\n")
    if text == "exit":
        exit(0)
    return (text,len_words)

def prediction(text:str="Hola",len_next_words:int=10):
    model = load_model("TextGeneratorModel")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=data.corpus)
    for i in range(0,len_next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list],maxlen=tgm.max_sequence_len-1,padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        predicted_word = tokenizer.sequences_to_texts([[predicted_index]])[0]

        text += " " + predicted_word
    return text




if __name__ == "__main__":
    main()