#!/usr/bin/env python
# coding: utf-8


import os
import glob
import matplotlib.pyplot as plt
import zipfile 
import pickle



def save_img_to_zip(inpath, outpath):
    
    saveimgs = glob.glob(inpath)
    
    with zipfile.ZipFile(os.path.join(outpath, 'imgchart.zip'), 'w') as myzip:
        for f in saveimgs:  
            myzip.write(f)



def load_img(inpath, outpath):
    files = os.listdir(inpath)
    
    file_clean = []
    file_check = []

    for file in files:
        try:
            img = plt.imread(os.path.join(inpath, file))
            file_clean.append(file)
        except:
            file_check.append(file)
            continue
    
    with open(os.path.join(outpath, 'img_can_use.pkl'), 'wb') as f:
        pickle.dump(file_clean, f)
    
    with open(os.path.join(outpath, 'img_can_not_use.pkl'), 'wb') as f:
        pickle.dump(file_check, f)
    
    # return file_clean, file_check




def check_img_type(imgfiles):
    imgtypes = {}

    for imgfile in imgfiles:
        imgtype = imgfile.split('_')[0]
        if imgtype not in imgtypes:
            imgtypes[imgtype] = 1
        else:
            imgtypes[imgtype] += 1

    return imgtypes





## test

if __name__ == '__main__':
	inpath = './datasets'
	outpath = './processed_datasets'
	save_img_to_zip(inpath, outpath)
	load_img(inpath, outpath)
	with open(os.path.join(outpath, 'img_can_use.pkl'), 'rb') as f:
		imgfiles = pickle.load(f)
	imgtypes = check_img_type(imgfiles)
	print(imgtypes)

