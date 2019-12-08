import glob
import os
from PIL import Image

# test
test = glob.glob("Data/test/img/ISIC*")
train = glob.glob("Data/train/img/ISIC*")

#counter = 0
#for image in test:
#  try:
#    im = Image.open(image).convert("RGB")
#    image_array = os.path.split(image)
#    print(image_array[1])
#    imResize = im.resize((400, 300), Image.ANTIALIAS)
#    imResize.save("Data/test/img/" + image_array[1], "JPEG", quality = 100)
#    print("{}/{} resizing completed".format(counter, len(test)))
#    counter = counter + 1
#  except:
#    print("error saving {} file".format(image))

counter = 0
for image in train:
  try:
    im = Image.open(image).convert("RGB")
    image_array = os.path.split(image)
    imResize = im.resize((400, 300), Image.ANTIALIAS)
    imResize.save("Data/train/img/" + image_array[1], "JPEG", quality = 100)
    print("{}/{} resizing completed".format(counter, len(train)))
    counter = counter + 1
  except:
    print("error saving {} file".format(image))

