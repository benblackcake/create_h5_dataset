import cv2
import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py


# Get the Image
def imread(path):
    img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img

def imsave(image, path, config):
    #checkimage(image)
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))

    # NOTE: because normial, we need mutlify 255 back    
    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)
    print("imshow")
    #checkimage(image)
    if __DEBUG__SHOW__IMAGE :
        imBGR2RGB = cv2.cvtColor(image.astype(np.float32),cv2.COLOR_BGR2RGB)
        plt.imshow(imBGR2RGB)
        plt.show()

    
def checkimage(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)

def downSample(image,scale=3):
    h,w,_ =image.shape
    h_n = h//scale
    w_n = w//scale
    img = np.full((h_n,w_n,_),0)
    
    for i in range(0,h):
        for j in range(0,w):
            if(i % scale==0 and j % scale==0):
                img[i//scale][j//scale] = image[i][j]
    return img



def shuffle_files(filenames):
    return random.shuffle(filenames)


def randomCrop(img_train, img_lable, width, height):
    # random.seed(1)

    assert img_train.shape[0] >= height
    assert img_train.shape[1] >= width    
    assert img_lable.shape[0] >= height
    assert img_lable.shape[1] >= width

    x = random.randint(0, img_train.shape[1] - width)
    y = random.randint(0, img_train.shape[0] - height)

    img_train = img_train[y:y+height, x:x+width]
    img_lable = img_lable[y:y+height, x:x+width]

    return img_train, img_lable
    
def crop_center(img,cropx,cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def process_sub_image(train_name, label_name, img_size=96, random_crop=False):
    img_train = train_name
    img_label = label_name

    h,w,_ = img_train.shape
    if h < img_size or w < img_size: 
        raise
    if random_crop:
        return randomCrop(img_train, img_label, img_size,img_size)

    else:
        return crop_center(img_train, img_label, img_size,img_size)

def fft_batch(batch_img):
    pass

def read_data(path, type_name):
    """
        Read h5 format data file

        Args:
            path: file path of desired file
            data: '.h5' file format that contains  input values
            label: '.h5' file format that contains label values 
    """
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get(type_name))
        return input_
        
        
def input_setup(args):
    """
    return file list
    """
    # random.seed(1)
    # if args.is_val:
    #     train_filenames = ''
    #     label_filenames = ''
    #     val_filenames = np.array(glob.glob(os.path.join(args.val_dir, '**', '*.*'), recursive=True))
    #     val_label_filenames = np.array(glob.glob(os.path.join(args.val_label_dir, '**', '*.*'), recursive=True))


    # if args.is_train:
    #     val_filenames = ''
    #     val_label_filenames = ''
    #     train_filenames = np.array(glob.glob(os.path.join(args.train_dir, '**', '*.*'), recursive=True))
    #     label_filenames = np.array(glob.glob(os.path.join(args.label_dir, '**', '*.*'), recursive=True))

    # if args.is_eval:
    train_filenames = np.array(glob.glob(os.path.join(args.train_dir, '**', '*.*'), recursive=True))
    label_filenames = np.array(glob.glob(os.path.join(args.label_dir, '**', '*.*'), recursive=True))

    val_filenames = np.array(glob.glob(os.path.join(args.val_dir, '**', '*.*'), recursive=True))
    val_label_filenames = np.array(glob.glob(os.path.join(args.val_label_dir, '**', '*.*'), recursive=True))

    eval_indices = np.random.randint(len(train_filenames), size=len(val_filenames))
    eval_filenames = train_filenames[eval_indices[:12]]
    eval_label_filenames = label_filenames[eval_indices[:12]]

    # print(len(train_filenames))
    # print((train_filenames[eval_indices[:12]]))
    # print((eval_label_filenames))

    # random.shuffle(train_filenames)
    # random.shuffle(val_filenames)gg
    # random.shuffle(eval_filenames)

    return train_filenames, label_filenames, val_filenames, val_label_filenames, eval_filenames, eval_label_filenames

