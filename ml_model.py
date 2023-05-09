import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import csv
import json
import pandas as pd
import os


def creater():
    

    # open the input JSON file
    with open('data/output.json') as f:

        data = json.load(f)

    # open the output CSV file and create a CSV writer object
    with open('data/output.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        # write the header row with the keys from the JSON data
        writer.writerow(data[0].keys())

        # write each row of data from the JSON file to the CSV file
        for row in data:
            writer.writerow(row.values())
        return (0)

# open the output CSV file and create a CSV writer object



def create_vdata():
    dbd=pd.read_csv("data/output.csv")
    dbd=dbd.drop(['speaker'], axis=1)

    polarity_score=[]
    subjectivity_score=[]

    for i in range(0,dbd.shape[0]):
        score=TextBlob(dbd.iloc[i][0])
        score1=score.sentiment[0]
        score2=score.sentiment[1]
        
        polarity_score.append(score1)
        subjectivity_score.append(score2)

    dbd=pd.concat([dbd,pd.Series(polarity_score)],axis=1)
    dbd=pd.concat([dbd,pd.Series(subjectivity_score)],axis=1)
    dbd.rename(columns={dbd.columns[5] :'Polarity_score'},inplace=True)
    dbd.rename(columns={dbd.columns[6]:'Subjectivity_score'},inplace=True)

    dbd.set_axis([ 'TEXT', 'start','end','Sentiment','Confidence','Polarity','Subjectivity'], axis='columns', inplace=True)
    datad=dbd


    convert_dict = {'Confidence': str,
                    'Polarity': str,
                    'TEXT':str,
                    'Subjectivity':str
                    }
    
    datad = datad.astype(convert_dict)
    vectorizer=pickle.load(open("vectorizer.pickle", 'rb')) 
    datad_concat = np.concatenate([datad['TEXT'].values.reshape(-1, 1),datad[['Confidence', 'Polarity', 'Subjectivity']].values],axis=1)
    datad_vec = vectorizer.transform(datad_concat.ravel())
    datad_vec = datad_vec.reshape(datad.shape[0], -1)


    with open('my_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    datadsentiment_pred_byloadedmodel = loaded_model.predict(datad_vec)
    print('Predicted sentiment by loaded model, labels:', datadsentiment_pred_byloadedmodel)
    datad['preds'] = datadsentiment_pred_byloadedmodel

    positives = []
    negatives = []
    neutrals = []
    for result in datad['preds']:
        
        if result == '2':
            positives.append(result)
        elif result == '0':
            negatives.append(result)
        else:
            neutrals.append(result)
            
    n_pos = len(positives)
    n_neg  = len(negatives)
    n_neut = len(neutrals)

    print("Num positives:", n_pos)
    print("Num negatives:", n_neg)
    print("Num neutrals:", n_neut)

    r_pos = (n_pos / (n_pos + n_neg+ n_neut))*100
    print(f"Positive ratio: {r_pos:.3f}")

    r_neg = (n_neg / (n_pos + n_neg+ n_neut))*100
    print(f"Negative ratio: {r_neg:.3f}")

    r_neut = (n_neut / (n_pos + n_neg+ n_neut))*100
    print(f"Neutral ratio: {r_neut:.3f}")

    polarity_score = (n_pos - n_neg) / (n_pos + n_neg)

    fig = plt.figure()
    labels = ['Positive [' + str(r_pos) + '%]',
           'Neutral [' + str(r_neut) + '%]',
                  'Negative [' + str(r_neg) + '%]'
                ]
    sizes = [r_pos, r_neut, r_neg]
    colors = [ 'lightgreen', 'gold', 'red' ]
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    strFile = r"static/images/plot1.png"
    if os.path.isfile(strFile):
        os.remove(strFile)  # Opt.: os.system("rm "+strFile)
    plt.savefig(strFile)
    



    
    os.remove("data/output.txt")
    os.remove("data/output.csv")
    os.remove("data/output.json")
    return (r_pos,r_neg,r_neut,n_pos,n_neg,n_neut,polarity_score)