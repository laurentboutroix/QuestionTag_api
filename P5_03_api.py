import pandas as pd
import re
import spacy
from bs4 import BeautifulSoup

import gradio as gr

import joblib

nlp = spacy.load('en_core_web_sm')
keep_postags = ['NOUN', 'ADJ', 'VERB', "ADV","ADP", 'X', 'PROPN']
little_words = ['c', 'r', 'x', 'on', 'qt']

vectorizer = "./le_tfidf_vectorizer.pkl"
loaded_vectorizer = joblib.load(vectorizer)

model = "./le_best_model.pkl"
loaded_model = joblib.load(model)

mbl = "./le_multilabel_binarizer.pkl"
mlb_load = joblib.load(mbl)

def tag_generator(text):
    text = text.lower()
    text = re.sub("node.js", "nodedotjs", text)
    text = re.sub("node js", "nodedotjs", text)
    text = re.sub("vue.js", "vuedotjs ", text)
    text = re.sub("asp.net-mvc","aspnetmvc", text)
    text = re.sub("model view controller","mvc", text)
    text = re.sub("model-view-controller","mvc", text)
    text = re.sub("asp.net","aspnet", text)
    text = re.sub("asp net", "aspnet", text)
    text = re.sub("c#","csharp", text)
    text = re.sub("\.net","dotnet", text)
    text = re.sub("\.js","dotjavascript", text)
    text = re.sub("js", "javascript", text)           
    text = re.sub("windows11","windowseleven", text)
    text = re.sub("windows10","windowsten", text)
    text = re.sub("office2000","officetwothousand", text)
    text = re.sub("windows7","windowsseven", text)
    text = re.sub("windows365","windowsthreehundredsixtyfive", text)
    text = re.sub("x11","xeleven", text)
    text = re.sub("office2003","officetwothousandthree", text)
    text = re.sub("office2007","officetwothousandseven", text)
    text = re.sub("office2010","officetwothousandten", text)
    text = re.sub("office2013","officetwothousandthirteen", text) 
    text = re.sub("office2016","officetwothousandsixteen", text)
    text = re.sub("office2019","officetwothousandnineteen", text)
    text = re.sub("ipv4","ipvfor", text)
    text = re.sub("ipv6","ipvsix", text) 
    text = re.sub("2d","twodimensions", text)
    text = re.sub("3d","threedimensions", text)
    text = re.sub("4d","fordimensions", text)
    text = re.sub("caff2","cafftwo", text) 
    text = re.sub("\.io","dotio", text)
    text = re.sub("\.com","dotcom", text)
    text = re.sub("objective-c","objectivec", text)
    text = re.sub("objective c", "objectivec", text)
    text = re.sub("f#","fsharp", text)
    text = re.sub("ms","microsoft", text)
    text = re.sub("for miscrosoft","formicrosoft", text) 
    text = re.sub("internet information services ", "iis", text)
    text = re.sub("language integrated query", "linq", text)
    text = re.sub("mac operating system", "macos", text)
    text = re.sub("mac os", "macos", text)    
    text = re.sub("object oriented programming", "oop", text)
    text = re.sub("object-oriented programming", "oop", text)
    text = re.sub("regular expression", "regex", text)
    text = re.sub("subversion", "svn", text)
    text = re.sub("visual basic for application", "vba", text)
    text = re.sub("visual basic .net", "vb.net", text)
    text = re.sub("visual basic", "vba", text)    
    text = re.sub("windows communication foundation", "wcf", text)
    text = re.sub("win for microsoft", "winformicrosoft", text)
    text = re.sub("windows presentation foundation", "wpf", text)
    text = re.sub("nodedotjavascript", "nodedotjs", text)
    text = re.sub("vuedotjavascript", "vuedotjs ", text)
    text = BeautifulSoup(text, "lxml").get_text()
    text = re.sub("[^a-zA-Z+]", " ", text)
    text = nlp(text) #Filter on POS tags and StopWords      
    text = [word.lemma_ for word in text if (word.pos_ in keep_postags)]  #Words to lemma 
    text = [w for w in text if len(w) > 2 or w in little_words] # filtre des mots comportant mois de 3 lettres mis à part ceux inclus dans la liste little_words
    text1 = pd.DataFrame(columns=["col"]) # création d'un dataframe dans lequel on insère en indice (0,0) l'ouput obtenu avec la fonction word_pos_lemma()
    text1.at[0, 'col'] = text
    text = loaded_vectorizer.transform(text1.col)
    y_pred = loaded_model.predict(text)
    tag_pred = mlb_load.inverse_transform(y_pred)
    return tag_pred
    
tagstack = gr.Interface(fn=tag_generator, title="Stack Overflow Tags generator", description="Hello you, to get a suggestion of the most frequently encountered tags on stackoverflow.com, please write in English the title of the question followed by the body of the question in an explicit and clear way specifying the technical environment of the subject.", inputs=["textbox"], outputs="textbox")
tagstack.launch(share=True)

