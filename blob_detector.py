
import cv2
from skimage.feature import blob_doh
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray , gray2rgb
def overlap_3_chanels(self,gt, pred,pred2,input):
        # print(input.shape, input.dtype, input.min(), input.max())
        # print(pred.shape, pred.dtype, pred.min(), pred.max())
        print(input.shape)
        pred = pred.astype(np.int32)
        gt = gt.astype(np.int32)

        tp = gt & pred & pred2
        fp = ~gt & pred & pred2
        fn = gt & ~pred & pred2
        tn = ~gt & ~pred & pred2

        print(tp.min(), tp.max(), fp.min(), fp.max(), fn.min(), fn.max())

        img = np.zeros((512, 512, 3), np.float32)
        img[:, :, 1] = tp
        img[:, :, 2] = fp
        img[:, :, 0] = fn

        input = gray2rgb(input)
        print(img.min(), img.max(), img.dtype, img.shape)
        print(input.min(), input.max(), input.dtype, input.shape)
        dst = cv2.addWeighted(input, 0.7, img, 0.3, 0)
        # dst = cv2.resize(dst,dsize,interpolation=cv2.INTER_AREA)
        # plt.imshow(img)
        # plt.show()

        return dst

def blob_detector(img,img2):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    plt.title('Determinant of Hessian')
    plt.imshow(img)
    #plt.tight_layout()
    plt.show()
    img = rgb2gray(img)
    img = img.astype(np.float64)
    img2 = rgb2gray(img2)
    img2 = img.astype(np.float64)
    blobs_doh = blob_doh(img, max_sigma=30, threshold=.01)
    blobs_doh2 = blob_doh(img2, max_sigma=30, threshold=.01)
    cords_list = {"x": [], "y": []}
    print (blobs_doh)
    #print (blobs_doh)
    fig, axes = plt.subplots()
    
    return cords_list


def main():
    img = cv2.imread(
        r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index12062023_1838\Test12102023_1447\Predictii_Overlap\OVERLAP_Colored_00e91486f2fd40f198e7e09216937155_48809182-1.png")
    list = blob_detector(img, img)


if __name__ == "__main__":
    main()
