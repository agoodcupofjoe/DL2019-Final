import glob
import os
from PIL import Image

# test
test = glob.glob("processed_data/test/img/ISIC*")
train = glob.glob("processed_data/train/img/ISIC*")

print(len(test))
counter = 0
for image in test:
  try:
    im = Image.open(image).convert("RGB")
    image_array = os.path.split(image)
    print(image_array[1])
    imResize = im.resize((200, 150), Image.ANTIALIAS)
    imResize.save("Data/test/img/" + image_array[1], "JPEG", quality = 100)
    print("{}/{} resizing completed".format(counter, len(test)))
    counter = counter + 1
  except:
    print("error saving {} file".format(image))

print(len(train))
counter = 0
for image in train:
  try:
    im = Image.open(image).convert("RGB")
    image_array = os.path.split(image)
    imResize = im.resize((200, 150), Image.ANTIALIAS)
    imResize.save("Data/train/img/" + image_array[1], "JPEG", quality = 100)
    print("{}/{} resizing completed".format(counter, len(train)))
    counter = counter + 1
  except:
    print("error saving {} file".format(image))

