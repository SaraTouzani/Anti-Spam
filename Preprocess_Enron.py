# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import os
import nltk
import os
import codecs
import re
from pandas import DataFrame
import numpy
from pymongo import MongoClient
from random import randint
import tika
from tika import parser
from nltk import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
NEWLINE = '\n'
SKIP_FILES = {'cmds'}
email_patern="\w+([-+.']\\w+)*@\\w+([-.]\\w+)*\\.\\w+([-.]\\w+)*" 
subject_patern="Subject(.*)\r\n"
lien_patern='http[\s|:|/|s]*[\w|.|/|\d]*'
#subject_patern="Subject(\s*|:|:\s*|\s*:)(\w|\W)+\s"
#EMAIL_PATERN2="(<)?(\w+@\w+(?:\.\w+)+)(?(1)>)"
#Text Processing
def preprocess(raw) :
    stemmer = PorterStemmer()
    wordlist = word_tokenize(raw)
    returnlist = []
    for word in wordlist :
        if (word not in stopwords.words("english")) :
            if (word  not in (('1','2','3','4','5','6','7','8','9','0','&','(','-','_','ç','é','à',')','=','+','°','~','#',':','?','.',',','/'))) :
                returnlist.append(stemmer.stem(word))
    return list(set(returnlist))
    
def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    p=parser.from_file(file_path)
                    vocab=[]
                    email=" "
                    sujet=" "
                    lien=" "
                    text=str(p["content"])
                    indexlien = re.search(lien_patern, text)##detection de liens
                    indexemail=re.search(email_patern , text)##detection des emails
                    indexsujet=re.search(subject_patern,text)##detection de sujets
                    if indexlien!=None:
                        lien+=text[indexlien.start():indexlien.end()]
                    if indexemail!=None:
                        email+=text[indexemail.start():indexemail.end()]
                    if indexsujet!=None:
                        sujet+=text[indexsujet.start():indexsujet.end()]
                        sujet=re.sub("\s|\W+|Subject(\s*|:|:\s*|\s*:)" , " ",sujet)

                    yield file_path, p,sujet,lien,email


def build_data_frame(path, classification):
    rows = []
    index = []
        
    for path,p,sujet,lien,email in read_files(path):
        rows.append({'text': p["content"], 'class': classification,'Metadata':p["metadata"],'Sujet':sujet,'Lien':lien,'email':email })
        index.append(path)

    data_frame = DataFrame(rows, index=index)
    return data_frame
    
    
###############
HAM = 'ham'
SPAM = 'spam'
path_='/home/hadoopusr'
SOURCES = [
    (path_+'/enron1/spam',        SPAM),
    (path_+'/enron1/ham',        HAM)
]
vocab_msg=[]
data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))
for i in range(len(data)):
    vocab_msg.append(preprocess(str(data.iloc[i,5])))
    
data['vocab_msg']=vocab_msg
data = data.reindex(numpy.random.permutation(data.index))

###################    


#Step 1: Connection a MongoDB 
client = MongoClient()
db=client["SpamHam"]

mycol=db["Table_TextClass"]

for i in range(len(data)):
    Message = {
     'text' : data.iloc[i,5],
     'class' : data.iloc[i,3],
     'Metadata':data.iloc[i,1],
     'Subject':data.iloc[i,2],## 5162 subject
     'Lien':data.iloc[i,0],##617 lien detecté
     'email':data.iloc[i,4],##0 email
     'vocab':data.iloc[i,6]## vocab stemmer du message
    }   
     #Step 3: Inserer l'object Message directement dans MongoDB via isnert_one
    result=mycol.insert_one(Message)
     #Step 4: Print le ObjectID du nouveau message stocké
    print('Created {0} of 5171 as {1}'.format(i,result.inserted_id))
         
print('The End')

# Restoring elm in Db 
for x in mycol.find(): #pour restorer toute les occurences 
    print(x)

#### iNVERTED iNDEX
def low(z):
    return z.lower()
    
def first(z):
    return z[0]

def squish(k,x):
    s=set(x)
    return sorted([(i, (clean_doc(str(data.iloc[i,5]))).count(k) ) for i in s], key=first)
        
def clean_doc(doc) :
    wordlist = word_tokenize(low(doc))
    returnlist = []
    for word in wordlist :
        if (word not in stopwords.words("english")) :
            if (word  not in (('1','2','3','4','5','6','7','8','9','0','&','(','-','_','ç','é','à',')','=','+','°','~','#',':','?','.',',','/',"'s","'ve","'re","'ll"))) :
                returnlist.append((word))
    return (returnlist)

d={}    
for i in range(len(data)):
    tokens_message=clean_doc(str(data.iloc[i,5]))
    for word in tokens_message:
        if word not in d:
            d[word]=[]
            d[word].append(1)
            d[word].append([i])
        else:
            d[word][0]+=1
            d[word][1].append(i)
print("list of words:")
ld=list(d)
ld.sort(key=low)

mycol_index=db["TableIndexInverse"]
for k in ld:
    if re.sub("[a-z|A-Z]+\w*","",k)=='':#on construit la table 
        ## que pour les mots ou les codes qui commencent par 
        ##des lettres sinn ça sera tres couteux de le faire
        print(k,d[k][0])
        Index_inv={
        'Mot':k,
        'FrequenceTotal':d[k][0],##freq total d'app du mot k
        'FrequenceMotParDocument':squish(k,d[k][1])##freq pour chq doc du mot k
        }
        result=mycol_index.insert_one(Index_inv)
print('Fini pour la partie d enrichisement de metadata \n analysant nos donnés a travers un probleme de classification supervisé ')