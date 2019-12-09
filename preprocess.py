import warnings
import os
import numpy as np
import glob
import skimage
from skimage import io, filters
import multiprocessing

def colorconstant(image,p,order=0,sigma=1):
    '''
    Corrects image for color constancy based on gray-edge algorithm (p = 6 case known as "Shades of Gray")

    References:
        "Edge-Based Color Constancy" by Joost van de Weijer, Theo Gevers, and Arjan Gijsenji
        "Improving Dermoscopy Image Classification Using Color Constancy" by Catarina Barata, M. Emre Celebi, and Jorge S. Marques
        "A Color Balancing Algorithm for Cameras" by Noy Cohen

    params:
        image - input image as a numerical array (numpy or output of io.imread)
        p - paramter for L^p (Minkowski) norm used for illumination estimation
        order - the order of the derivative filter used for gradient estimation
        sigma - the sigma used for Gaussian convolution on the input image

    output:
        Numpy array of pixel values for color-corrected image (values are float in [0,1])
    '''
    # Gaussian filter convolution (kernel size automatically determined based on sigma)
    filtered = filters.gaussian(image,sigma=sigma,multichannel=True)
    # Choosing nth-order derivative for filter and applying derivative filter
    if order == 0:
        derivative = [filtered[:,:,channel] for channel in range(3)]
    else:
        if order == 1:
            derive = filters.sobel
        else:
            derive = filters.laplace
        derivative = [np.abs(derive(filtered[:,:,channel])) for channel in range(3)]
    # Removing pixels that were saturated in original image
    for channel in range(3):
        derivative[channel][image[:,:,channel] >= 255] = 0.
    # Determining L^p norm used for illumination estimation (p = -1 denotes infinity)
    if p == -1:
        norm = np.max
    else:
        norm = lambda x: np.power(np.sum(np.power(x,p)),1 / p)
    # Estimating illumination
    illumination = [norm(channel) for channel in derivative]
    # Scaling each channel by respective illumination (using broadcasting)
    illumination = illumination / np.linalg.norm(illumination) * np.sqrt(3)
    # Returns image as ndarray of uint8
    return np.clip(image / illumination / 255., 0., 0.999999)

def fixborder(image):
    '''
    :param: image, a numpy array of shape (width,height,channels) or (height,width,channels)
    returns: image, the original image with the white border replaced by the nexxt nearest pixel color
    '''
    image[0,:,:] = image[1,:,:]
    image[:,0,:] = image[:,1,:]
    image[-1,:,:] = image[-2,:,:]
    image[:,-1,:] = image[:,-2,:]
    return image

trainimages = False

def processimage(filename):
    '''
    params:
        filename - string for input image filename

    output:
        None - replaces image at filename with processed version
    '''
    image = io.imread(filename)
    if trainimages:
        image = colorconstant(image,6)
    image = skimage.transform.resize(image,(150,200),anti_aliasing=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l = skimage.img_as_ubyte(image)
    out_file = "processed_data/" + filename.split("/", 1)[1]
    io.imsave(out_file, l, quality = 100)
    print("Processed image at: {}".format(out_file))
def process_image_with_exception(filename):
    try:
        processimage(filename)
        return True
    except Exception as e:
        print("\nFAILED TO PROCESS: {}\nException: {}\n".format(filename, str(e)))
        return False
def get_batches(imgs, batch_size=1024):
    i = 0
    while i < len(imgs):
        yield imgs[i:i+batch_size]
        i += batch_size
def main():
    # Reshape each image in train and test folder to 150x200 pixels
    # We know that most images are already in 3:4 aspect ratio
    with open("already_done.txt") as myfile:
        already_done = myfile.readlines()
    folders = ["processed_data/test/img", "processed_data/train/img"]
    for folder in folders:
        imgs = [img for img in glob.glob(folder + "/ISIC*") if img not in already_done]
        directory = "processed_data/" + folder.split("/", 1)[1]
        if not os.path.exists(directory):
            os.makedirs(directory)
        for batch_imgs in get_batches(imgs):
            batch_results = pool.map(process_image_with_exception, batch_imgs)
            successes = [filename for filename, succ in zip(batch_imgs, batch_results) if succ]
            failures = [filename for filename, succ in zip(batch_imgs, batch_results) if not succ]
            already_done = successes
            not_done = failures
            with open("already_done.txt", "a+") as myfile:
                myfile.write('\n'.join(already_done))
            with open('not_done.txt', mode='w+') as myfile:
                myfile.write('\n'.join(not_done))
        trainimages = True
    print("Finished processing all images")
if __name__ == "__main__":
    pool = multiprocessing.Pool(48)
    main()
