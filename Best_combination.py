import cv2
import numpy as np
import os
import imagenet0
import imagenet_1
import time


def best_combination(img,bestBefore,point,path_o):
    w, h, channel = img.shape
    y_deta = w/4
    x_deta = h/4
    miximg = np.zeros((w, h, 3), np.uint8)
    miximg.fill(0)
    poy = int(point/4)
    pox = point % 4
    if len(bestBefore) == 0:
        miximg[0:int(y_deta),0:int(x_deta)] = img[int(y_deta*poy):int(y_deta*(poy+1)),int(x_deta*pox):int(x_deta*(pox+1))]
        cv2.imwrite(path_o,miximg)
    else:
        for pj in range(len(bestBefore)):
            poyj = int(bestBefore[pj]/4)
            poxj = bestBefore[pj]%4
            miximg[int(y_deta * poyj):int(y_deta * (poyj + 1)),int(x_deta * poxj):int(x_deta * (poxj + 1))] = \
                img[int(y_deta * poyj):int(y_deta * (poyj + 1)),int(x_deta * poxj):int(x_deta * (poxj + 1))]
        qlast = len(bestBefore)
        miximg[int(y_deta * poy):int(y_deta * (poy + 1)), int(x_deta * pox):int(x_deta * (pox + 1))] = \
            img[int(y_deta * poy):int(y_deta * (poy + 1)), int(x_deta * pox):int(x_deta * (pox + 1))]
        cv2.imwrite(path_o, miximg)

def y_or_nPath(path):
    if not os.path.exists(path):
        os.makedirs(path)



def combine_point(point,aqc):

    data_n = ['train','val']
    data_class = ['Low level','Medium level','High level']
    data_out_path = '/home/qmz/Array optimization method/Best combination/'
    best_point = point[0:5]
    # best_point = [14]

    if aqc % 2 !=0:
        for n in data_n:
            for c in data_class:
                doc_ZSDCNN = '/home/qmz/Array optimization method/ALL_Food_s/%s/%s/' % (n, c)
                img_name = file_name_list = os.listdir(doc_ZSDCNN)
                path_out = data_out_path + '%s/%s/%s/' % ('6_point0', n, c)
                y_or_nPath(path_out)
                for img_n in img_name:
                    img = cv2.imread(doc_ZSDCNN + img_n, cv2.IMREAD_UNCHANGED)
                    best_combination(img, best_point, point[5], (path_out + img_n)) #point[5]
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        time.sleep(2)
        print('im0')
        loss, trainacc, valacc, abc = imagenet0.main()
        return loss, trainacc, valacc,abc
    else:
        for n in data_n:
            for c in data_class:
                doc_ZSDCNN = '/home/qmz/Array optimization method/ALL_Food_s/%s/%s/' % (n, c)
                img_name = file_name_list = os.listdir(doc_ZSDCNN)
                path_out = data_out_path + '%s/%s/%s/' % ('6_point1', n, c)
                y_or_nPath(path_out)
                for img_n in img_name:
                    img = cv2.imread(doc_ZSDCNN + img_n, cv2.IMREAD_UNCHANGED)
                    best_combination(img, best_point, point[5], (path_out + img_n))


        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        print('im1')
        loss, trainacc, valacc,abc = imagenet_1.main()
        return loss, trainacc, valacc,abc


