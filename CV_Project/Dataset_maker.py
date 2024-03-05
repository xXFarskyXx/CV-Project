import os 
import cv2
import numpy as np
import argparse

def grid_crop(paths, result_path):
    for i,path in enumerate(os.listdir(paths)):
        im_path = os.path.join(paths, path)
        im = cv2.imread(im_path)
        crop_im = im[50:450, 50:450]
        grid_list = []
        for i in range(8):
            for j in range(8):
                cv2.rectangle(crop_im, (i*50, j*50), (i*50+50, j*50+50), (0, 255, 0), 2)
                grid_list.append(crop_im[j*50:j*50+50, i*50:i*50+50])
                
    os.makedirs(result_path, exist_ok=True)
    for i, grid in enumerate(grid_list):
        out_path = f"Grid{i}.jpg"
        cv2.imwrite(os.path.join(result_path, out_path), grid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prints a chessboard-like representation with grid Image.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--result', type=str)
    args = parser.parse_args()
    grid_crop(args.dataset, args.result)