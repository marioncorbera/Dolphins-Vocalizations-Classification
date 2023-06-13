# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:53:52 2022

@author: marion.corbera
"""


###############################################################################
### PROCESS 0 : Découpage en morceaux de x minutes
###############################################################################

###############################################################################
### SCRIPT PREPROCESS0 :
    
   # 1- Découpe les audio en échnatillons de x secondes.
   # 2- Preprocess echantillons
   # 3- Extrait features demander et renvoie un doctionnaire
   
###############################################################################


###############################################################################       
### IMPORT PACKAGES
###############################################################################
  
import os
import librosa
from tqdm import tqdm
import numpy as np

###############################################################################       


###############################################################################       
### CREATION FONCTIONS
###############################################################################


def initialisor(wanted_feature):
    # zone stockage données & leur noms
    dataset = {
        "mapping": [], # nom des classes
        "name": [],
        "label":[], # normalement numerique
        str(wanted_feature): []
    }
    
    return dataset



def normaliser(record):
    # norm MinMax
    norm_record = (record - record.min()) / (record.max() - record.min())
    
    return norm_record



def loader(file_path, sr, mono):
    # sr : freq echantillonnage
    # mono : nb de cannaux
    record = librosa.load(file_path, sr, mono)[0]   #duration=5
    
    # normalise
    norm_record = normaliser(record)
        
    return norm_record



def padder(sequence, sr, seq_time):
    # ajoute 0 si sequence trop courte
    expected_len = sr*seq_time
    if len(sequence)<expected_len:
        missing_len = int(expected_len - len(sequence))
        sequence =  np.pad(sequence, (0, missing_len), mode='constant')

    return sequence
     


def features_extractor(sample, sr, wanted_feature, frame_size, hop_length, n_mfcc):
    # extrait les features demandés
    
    if wanted_feature == 'spectrogramme':
        # short time fourier transform
        stft = librosa.core.stft(sample, frame_size, hop_length) # matrice complexe numbers
        features = np.abs(stft)**2
        
    elif wanted_feature == 'log_spectrogramme':
        # short time fourier transform
        stft = librosa.core.stft(sample, frame_size, hop_length) # matrice complexe numbers
        spectrogramme = np.abs(stft)**2        
        features = librosa.power_to_db(spectrogramme)
        
    elif wanted_feature == 'MFCCs':
        # Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(sample, sr, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length)
        #num_mfcc_vectors_per_segment = math.ceil(len(sample) / hop_length)
        #if len(mfccs) == num_mfcc_vectors_per_segment:
        features = mfccs.T
        
    elif wanted_feature == 'MFCCs_delta':
        mfccs = librosa.feature.mfcc(sample, sr, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        #features = np.concatenate((mfccs,delta_mfccs,delta2_mfccs))#.flatten()
        d_mfccs = np.concatenate((mfccs,delta_mfccs,delta2_mfccs))#.flatten()
        features = d_mfccs.T
    
    else:
        print('feature demandé non reconnue/codée pour le moment')
    
    
    return features.tolist()
       



def sequenceur(record, sr, seq_time, overlap, wanted_feature, frame_size, hop_length, n_mfcc, dataset, file_name, i):
    # seq_time : durée des sequences en secoondes
    # overlap : chevauchement entre sequence
    duration = len(record)/sr
      
    nb_segments = int(duration/seq_time)
        
    # boucle pour tout les segments d'un audio
    for seq in tqdm(range(nb_segments+1), desc= "slicing file in sequences"): 
        # on prend pas dernier segment (pas + 1)
        # pour si jamais < longueur sequence qui reste.
            
        # début séquence 'seq'
        start = int((seq_time*sr-sr*overlap)*seq)
        finish = int(start + seq_time*sr)
            
        # sauvegarde smaller sample
        sequence = record[start:finish]
        sample_name = file_name+'seq_'+str(seq+1)
        
        # pad si besoin
        sample = padder(sequence, sr, seq_time)
            
        # extrait features
        features = features_extractor(sample, sr, wanted_feature, frame_size, hop_length, n_mfcc)
        
        # mise en place de dataframe
        #dataset["label"].append(semantic_label)
        dataset["label"].append(i-1)  
        dataset["name"].append(sample_name)
        dataset[str(wanted_feature)].append(features)
        
    return dataset           



    
def Pipeline_preprocessing(datapath, sr=96000, mono=True, seq_time=0.5, overlap=0.1, wanted_feature='MFCCs_delta', frame_size=2048, hop_length=512, n_mfcc=40):
   
    # record path = chemin des recodings originaux
    # sr = sample rate
    # mono = nb de channel
    # seq_time = durée sous séquence => echantillon
    # over_lap = chevauchement sequence
    # wanted_feature = feature à extraire (spectro, log_spectro & mfccs)
    # frame_size = gde frame size : aumengte freq resolution / baisse temps resolution VS petite frame size : baisse freq resolution / aumengtes temps resolution 
    # hop_length = nb de pts on décalle vers droite pour changer de frame
    # n_mfcc = nb mel_band : hyperparametre à optimiser 40, 128... => nb de point mel echelle 


    # création dataframes stockages
    dataset = initialisor(wanted_feature)
    
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(datapath)): 
        # dirpath = dossier actuel
        # dirnames = sous-dossier
        # filesnames = tout les fichier dans dirpath
        pass
    
        # fonction os.walk va mettre dataset_path = dir_path donc on s'assure de bien être au noiveau sous-dossiers (dirnames)
        if dirpath is not datapath:
            
            # sauvegarde nom des labels 
            semantic_label = dirpath.split("\\")[-1] # F:\Marion\Recordings\Ventouse\echolocation => met dans liste chaque mot => on grade que dernier => 'echolocation' par exemple
            print("\nProcessing: {}".format(semantic_label))  
            
            # mapping pour les labels
            dataset["mapping"].append(semantic_label)
            
        
            for file in tqdm(os.listdir(dirpath), desc="processing files in the current path"): # juste nom du file => il nous faut le file path
                
                # get file name
                file_name = file.replace(".WAV", "")
                file_name = file.replace(".wav", "")

                # loading audio par audio du dossier
                file_path = os.path.join(dirpath, file)
                record = loader(file_path, sr=sr, mono=mono) 
        
                # normalisation, découpage, padding & extractaction des features
                dataset = sequenceur(record,
                                     sr,
                                     seq_time,
                                     overlap,
                                     wanted_feature,
                                     frame_size,
                                     hop_length,
                                     n_mfcc,
                                     dataset,
                                     file_name,
                                     i)

    
    return dataset

###############################################################################       


###############################################################################       
### PREPROCESS 0
###############################################################################       
        
DATAPATH = r'G:\Marion\data_training\dataset_mix96_192'
#DATAPATH = r'F:\Marion\vocalizations'
SR = 96000
MONO = True
SEQ_TIME = 0.5
OVER_LAP = 0.05
WANTED_FEATURE='MFCCs'
FRAME_SIZE = 1024  
HOP_LENGTH = 256
N_MFCCS = 20

if __name__ == "__main__":
    data = Pipeline_preprocessing(DATAPATH, sr=SR, wanted_feature=WANTED_FEATURE, n_mfcc=N_MFCCS)    
  
# si besoin save en json    
import json
with open(r'G:\Marion\Preparation\Supervised_testing\dta.json', "w") as fp: #'w' = write
    json.dump(data, fp, indent=4)

###############################################################################

x = np.array(data[str(WANTED_FEATURE)])
spectro = x
y = np.array(data['label'])
labels = y







# ajout autre possibilité features






### ici pb avec mfccs merdeux à partir de plus 20kHz => essayer autre filtres!
# recherche best filterbank
filter_bank = librosa.filters.mel(n_fft=2048, sr=192000, n_mels=13)
filter_bank.shape

plt.figure(figsize=[25,15])
librosa.display.specshow(filter_bank, sr=192000, x_axis='linear')
plt.colorbar(format='%+2.f')
plt.xlabel('Freq Hz')
plt.ylabel('bands')
plt.show()

