import sys
import os
import multiprocessing as mp
import threading
import cv2
import time
from functools import reduce
from consts import *

input_dir = ""
output_dir = ""

def countPixels(img):
    n = 400 // 2
    h, w = img.shape
    split_img = []
    k = 0
    for y in range(0, h, n):
        for x in range(0, w, n):
            split_img.insert(k, img[y: y + n, x:x + n])
            k += 1
    black_pixels_avg = []
    for i in range(4):
        dark_pixels = n ** 2 - cv2.countNonZero(split_img[i])
        black_pixels_avg.insert(i, dark_pixels / (n ** 2))
    return black_pixels_avg

def patch_viability(patch):
    binarized = cv2.adaptiveThreshold(cv2.medianBlur(patch, 13), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    black_pixels_avg = countPixels(binarized)
    delta_black = 0.25
    delta_white = 0.10
    count = 0
    for i in range(len(black_pixels_avg)):
        if black_pixels_avg[i] < delta_white:
            count += 1
        if black_pixels_avg[i] > delta_black:
            count += 1
        if count >= 2:
            return False
    return True

def patchifier(img,path):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    counter = 0
    for r in range(0,img.shape[0]-400+1,400):
        for c in range(0,img.shape[1]-400+1,400):
            cropped = img[r:r+400,c:c+400]
            if patch_viability(cropped) == True:
                name = path.rsplit(os.path.sep,1)[1]
                for c in classes:
                    if c in name:
                        name = name[0:len(c)]+'_'+name[len(c):]
                        break
                for sc in subclasses:
                    if sc in name:
                        name = name[0:name.index(sc)+len(sc)]+'_'+name[name.index(sc)+len(sc):]
                        break
                name = name.rsplit(".",1)[0]+'_'+str(counter)+".jpg"
                cv2.imwrite(os.path.join(output_dir,name),cropped)
                counter+=1
  
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    h,w = img.shape[0],img.shape[1]
    img = img[h//10:9*h//10,w//10:9*w//10]
    #if 2-pages image, split into 2
    if img.shape[0]<img.shape[1]:
        left_img = img[0:img.shape[0],0:img.shape[1]//2]
        patchifier(left_img,image_path.rsplit(".",1)[0]+"_left.jpg")
        right_img = img[0:img.shape[0],img.shape[1]//2+1:img.shape[1]]
        patchifier(right_img,image_path.rsplit(".",1)[0]+"_right.jpg")
    else:
        patchifier(img,image_path)   
   

def preprocess(imagelist,in_dir,out_dir):
    global input_dir
    global output_dir
    input_dir = in_dir
    output_dir = out_dir
    for image in imagelist:
        threading.Thread(target=preprocess_image, args = (image,)).start()

def process_division():
    cpus = mp.cpu_count()
    images = reduce(lambda x,y:x+y,[[os.path.join(os.path.join(input_dir,folder),image) for image in os.listdir(os.path.join(input_dir,folder))] for folder in os.listdir(input_dir) if folder != "output"],[])
    for i in range(cpus):
        mp.Process(target = preprocess, args = (images[i::cpus],input_dir,output_dir,)).start()
        
def create_output_folder():
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

def main(args):
    if len(args) == 0:
        raise "No input directory"
    elif len(args) < 2:
        raise "No output directory"
    global input_dir
    global output_dir
    input_dir = args[0]
    output_dir = args[1]
    create_output_folder()
    process_division()

if __name__ == "__main__":
   main(sys.argv[1:])
