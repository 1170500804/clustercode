import pandas as pd
import os
import shutil

root_folder = '/home/saschaho/Simcenter/Floor_Elevation_Data'
img_folder = 'Streetview_Irma/Streetview_Irma/images'

train_folder = '/home/saschaho/Simcenter/copy_for_zhirong/train'
val_folder = '/home/saschaho/Simcenter/copy_for_zhirong/val'
test_folder = '/home/saschaho/Simcenter/copy_for_zhirong/test'

train_pd = pd.read_csv(os.path.join(root_folder,'elevation_train.csv'))
val_pd = pd.read_csv(os.path.join(root_folder,'elevation_val.csv'))
test_pd = pd.read_csv(os.path.join(root_folder,'elevation_test.csv'))

for row in train_pd.iterrows():

    filename = row[1]['filename']
    ffe = str(row[1]['first_floor_elevation_ft'])
    dest_folder = os.path.join(train_folder,ffe)
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    print(dest_folder)
    shutil.copy(os.path.join(root_folder,img_folder,filename),dest_folder)

for row in val_pd.iterrows():

    filename = row[1]['filename']
    ffe = str(row[1]['first_floor_elevation_ft'])
    dest_folder = os.path.join(val_folder,ffe)
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    print(dest_folder)
    shutil.copy(os.path.join(root_folder,img_folder,filename),dest_folder)

for row in test_pd.iterrows():

    filename = row[1]['filename']
    ffe = str(row[1]['first_floor_elevation_ft'])
    dest_folder = os.path.join(test_folder,ffe)
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    print(dest_folder)
    shutil.copy(os.path.join(root_folder,img_folder,filename),dest_folder)