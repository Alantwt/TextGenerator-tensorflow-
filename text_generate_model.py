import tensorflow as tf
import keras
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import data_utils
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt


from data import data

# Global variables
tokenizer = Tokenizer()
total_words = int()
input_sequences = list()
max_sequence_len = int()
xtraining_data = list()
ytraining_data = list()
model = Sequential()
adam = Adam()
max_sequence_len = int()
checkpoint_callback = None
history = None



def main():
    
    make_sequence()
    make_trainigdata()
    create_model()
    checkpoints()
    training()
    print(test("hola como estas",15))



def make_sequence():
    global tokenizer,total_words,max_sequence_len,input_sequences
    input_sequences = []
    data.data_validate()
    # Convertir los datos a una secuencia unica de enteros
    tokenizer.fit_on_texts(texts=data.corpus)

    # Calcular la cantidad de palabras unicas 
    total_words = len(tokenizer.word_index) + 1

    # Transformas la secuencia de palabras a una secuencia de enteros
    for line in data.corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[ : i+1]
            input_sequences.append(n_gram_sequence)
    
    # Obtener la secuencia con mayor longitud
    max_sequence_len = max([len(x) for x in input_sequences])

    # Rellenar los espacios vacios con ceros, y de esa forma tener todos los elementos de la misma longitud
    input_sequences = np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding="pre")) 



def make_trainigdata():
    global xtraining_data,ytraining_data

    # Tomar todos los datos desde el primero hasta el penultimo, el ultimo se usara de valor de salida
    xtraining_data = input_sequences[:,:-1]

    # Obtener solo el ultimo dato
    outputs = input_sequences[:,-1]

    # Cargar los datos de salida en categorias onehot
    ytraining_data = keras.utils.to_categorical(outputs,num_classes=total_words)



def create_model():
    global model,adam

    # Dimensiones de las capas
    embedding_dim = 250

    # Agregar las capas al modelo
    model.add(Embedding(total_words,embedding_dim,input_length=max_sequence_len-1)) # Capa de procesamiento
    model.add(Bidirectional(LSTM(240))) # capa oculta
    model.add(Dense(total_words,activation="softmax")) # capa de salida, con multiples categorias

    # Optimizador
    adam = Adam(learning_rate=0.0089)

    # Compilar el modelo
    model.compile(loss="categorical_crossentropy",optimizer=adam,metrics=["accuracy"])

    

def checkpoints():
    global checkpoint_callback

    # Guardar un checkpoint del entrenamiento
    checkpoint_dir = os.path.abspath("training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir,"checkpoint_{epoch}")
    checkpoint_callback = ModelCheckpoint(
                            filepath=checkpoint_prefix,
                            save_weights_only=True
                        )

def training():
    global history

    # Entrenar el modelo
    history = model.fit(xtraining_data,ytraining_data,epochs=20,verbose=1,callbacks=[checkpoint_callback])
    
    # Guardar el modelo
    model.save("TextGeneratorModel.h5")

    plt.figure()
    plt.subplot(1,2,1)
    data_graphs("loss")
    plt.subplot(1,2,2)
    data_graphs("accuracy")
    plt.show()



def data_graphs(metric:str="acurracy"):
    global history
    plt.plot(history.epoch, history.history[metric])
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend(metric)
    




def test(text:str="Hola",len_next_words:int=10):
    for i in range(0,len_next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        predicted_word = tokenizer.sequences_to_texts([[predicted_index]])[0]
        text += " " + predicted_word

    return text



if __name__ == "__main__":
    main()