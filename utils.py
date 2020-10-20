import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from ipywidgets import interact, fixed
import ipywidgets as widgets
import IPython.display as display


def imshow(x):
  '''
  Plots image x (in cwh order) proportional to its original size.
  '''
  x = np.squeeze(x).transpose((1, 2, 0)).astype(np.uint8)

  dpi = plt.rcParams['figure.dpi']
  height, width, depth = x.shape
  figsize = width / float(dpi), height / float(dpi)

  fig, ax = plt.subplots(figsize=figsize)
  ax.imshow(x)
  ax.set_axis_off()
  return ax


def get_image(url):
  '''
  Downlaods an image from url and returns np array in cwh order.
  '''
  im = Image.open(requests.get(url, stream=True).raw)
  im = np.array(im)
  return im.transpose(2, 0, 1)


def interct_imshow(img_list):
  '''
  Given a list of images in cwh order, creates an interactive slider and shows
  each list element.
  '''
  # img_list = np.squeeze(img_list).transpose(0, 2, 3, 1).astype(np.uint8)
  interact(
    # lambda img_list, idx: display.display(Image.fromarray(img_list[idx])),
    lambda idx: imshow(img_list[idx]),
    # img_list=fixed(img_list),
    idx=widgets.IntSlider(min=0, max=len(img_list)-1, step=1, value=0),
    continuous_update=False)


def get_vgg19_layer_names(model):
  start, end = 'conv1_1', 'fc8'
  attr_list = list(model.__dict__.keys())
  start_idx = attr_list.index(start)
  end_idx = attr_list.index(end)
  return attr_list[start_idx : (end_idx + 1)]

