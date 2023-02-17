import re

corpus = []

def data_validate():
    with open("..\TextGenerate\data\TrainingText.txt","r",encoding="utf-8") as f:
        for line in f:
            if len(line) > 1:
                line.replace(r"[^\w\s]","")
                line = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-záéíóúñ  \t])|(\w+:\/\/\S+|^rt|http.+?)","",line)
                corpus.append(line.strip().lower())
        f.close()

