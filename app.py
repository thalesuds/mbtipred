#######################################
#Bibliotecas da aplicação
from joblib import load
from flask import Flask, request, url_for, redirect, render_template, jsonify
import numpy as np
import pandas as pd
import json
import pickle

##########################################
#Bibliotecas das Funções Auxiliares de ML
import collections
import re
import os
import emoji
import spacy
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import opendatasets as od

#########################################
#Bibliotecas das Funções Auxiliares de DL
import keras
import keras_nlp.layers
from sklearn.preprocessing import OneHotEncoder as OHE
from keras_preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras_nlp.layers import FNetEncoder
from keras.layers import Dense, Flatten, Embedding, Dropout

classes = np.array(['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP','ENTJ', 'ISTJ', 'ENFJ', 'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ'])
OHE = OHE(handle_unknown='ignore')
OHE.fit(classes.reshape(-1,1))

tokenizador = ''
with open('Tokenizador.json') as f:
        data = json.load(f)
        tokenizador = tokenizer_from_json(data)

############################################

#Modelo de Machine Learning
dicionario = {}
with open('models/DicionarioCountVectorizer.json') as f:
    dicionario = json.load(f)    
vec = CountVectorizer(analyzer='word', strip_accents = 'unicode', lowercase = False, stop_words = ENGLISH_STOP_WORDS, min_df = 20, vocabulary = dicionario)
tfidf_transformer = TfidfTransformer()
model_svm_decisao = load('models/svm_decisao.joblib')
model_svm_energia = load('models/svm_energia.joblib')
model_svm_estilo = load('models/svm_estilo.joblib')
model_svm_informacao = load('models/svm_informacao.joblib')

############################################
#Funções Auxiliares
#Criando função de tratamento que contempla remoção de stop_words, stemming, menções, hashtags, links, emojis, etc.
def tratamento_de_texto(texto, remover_stop_words = True, fazer_stemming = True, remover_mencoes_hashtags = True,
                        remover_links = True, remover_emojis = True):

    #Remoção de classes do próprio texto
    
    texto = re.sub(r'ISTP\S+', '', texto)
    texto = re.sub(r'ISFP\S+', '', texto)
    texto = re.sub(r'ENTJ\S+', '', texto)
    texto = re.sub(r'ISTJ\S+', '', texto)
    texto = re.sub(r'ENFJ\S+', '', texto)
    texto = re.sub(r'ISFJ\S+', '', texto)
    texto = re.sub(r'ESTP\S+', '', texto)
    texto = re.sub(r'INFP\S+', '', texto)
    texto = re.sub(r'INFJ\S+', '', texto)
    texto = re.sub(r'INTP\S+', '', texto)
    texto = re.sub(r'INTJ\S+', '', texto)
    texto = re.sub(r'ESFP\S+', '', texto)
    texto = re.sub(r'ESFJ\S+', '', texto)
    texto = re.sub(r'ESTJ\S+', '', texto)
    texto = re.sub(r'ENTP\S+', '', texto)
    texto = re.sub(r'ENFP\S+', '', texto)

    # Remoção de Emojis
    if remover_emojis == True:
        emoji_pattern = re.compile("[" "\U0001F1E0-\U0001F6FF" "]+", flags=re.UNICODE)
        texto = emoji_pattern.sub(r"", texto)
        texto = "".join([x for x in texto if x not in emoji.EMOJI_DATA])
        texto = ''.join((x for x in texto if not x.isdigit()))

    if remover_mencoes_hashtags:
        texto = re.sub(r"@(\w+)", "", texto)
        texto = re.sub(r"#(\w+)", "", texto)
        texto = re.sub(r"__(\w+)", "", texto)

    if remover_links:
        texto = re.sub(r'http\S+', '', texto)

    #Remoção de caracteres diversos
    texto = re.sub(r"[^\x00-\x7F]+", " ", texto)
    pontuacoes_e_numeros = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = pontuacoes_e_numeros.sub(" ", texto.lower())
    palavras = (''.join(nopunct)).split()

    if(remover_stop_words):
        palavras = [p for p in palavras if p not in ENGLISH_STOP_WORDS]
        palavras = [p for p in palavras if len(p) > 2]  # remove a,an,of etc.

    if(fazer_stemming):
        stemmer = PorterStemmer()
        palavras = [stemmer.stem(p) for p in palavras]

    return texto

def MLPredictToString(energia, informacao, decisao, estilo):
    
    predicao = ''

    if energia == 0:
        predicao += 'E'
    else:
        predicao += 'I'
    
    if informacao == 0:
        predicao += 'S'
    else:
        predicao += 'N'
    
    if decisao == 0:
        predicao += 'F'
    else:
        predicao += 'T'
    
    if estilo == 0:
        predicao+='P'
    else:
        predicao+='J'
    
    return predicao
############################################################################
#Criar Modelo de DL
VocabSize = 10000
TamanhoMaximo = 1500
input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

x = Embedding(VocabSize, 256, input_length=TamanhoMaximo)(input_ids)

x = FNetEncoder(intermediate_dim=64)(inputs=x)
x = FNetEncoder(intermediate_dim=750)(inputs=x)
x = FNetEncoder(intermediate_dim=750)(inputs=x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(16, activation="softmax")(x)

fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")
fnet_classifier.load_weights('models/FNet_DLPesos.h5')

########################
#Criação do Aplicativo
app = Flask(__name__)

@app.route("/")
def home():
     return render_template("home.html")

@app.route("/MLHome")
def MLHome():
     return render_template("MLHome.html")

@app.route("/DLHome")
def DLHome():
     return render_template("DLHome.html")

@app.route("/MLPredict", methods = ['POST'])
def MLPredict():
    #request das entradas
    entrada = [x for x in request.form.values()]

    #processando os textos
    tratado = [tratamento_de_texto(entrada[0],  remover_stop_words = True, fazer_stemming = True)]

    X = vec.fit_transform(tratado)
    X = tfidf_transformer.fit_transform(X)
    
    #predict
    energia = model_svm_energia.predict(X)
    informacao = model_svm_informacao.predict(X)
    decisao = model_svm_decisao.predict(X)
    estilo = model_svm_estilo.predict(X)
    
    predicao = MLPredictToString(energia, informacao, decisao, estilo)
    return render_template("MLHome.html", pred = "Predição: {}".format(predicao))

@app.route('/DLPredict', methods = ['POST'])
def DLPredict():

    #request das entradas
    entrada = [x for x in request.form.values()]

    Sequencias = tokenizador.texts_to_sequences(entrada)
    Padded = pad_sequences(Sequencias, maxlen = 1500, truncating = "post", padding = "post")
    #predict
    predicao = fnet_classifier.predict(Padded)
    predicao = predicao[0]
    maximo = max(predicao)
    predicao = np.where(predicao == maximo, 1, 0)
    idx = np.where(predicao)
    return render_template("DLHome.html", pred = "Predição: {}".format(classes[idx]))

@app.route('/MetricaML', methods = ['GET'])
def MetricaML():
    
    metricas = {"decisao": 0, "estilo":0, "energia": 0, "informacao":0}

    for chave in metricas.keys():

        endereco = 'metricas/svm_'+str(chave)+'.json'

        with open(endereco) as f:
            modelo_metrica = json.load(f)    
            metricas[str(chave)] = modelo_metrica
    
    return metricas

@app.route('/MetricaDL', methods = ['GET'])
def MetricaDL():
    
    with open('metricas/MetricaDL_Fnet.json') as f:
        modelo_metrica = json.load(f)    
    
    return modelo_metrica

if __name__ == "__main__":
    app.run()