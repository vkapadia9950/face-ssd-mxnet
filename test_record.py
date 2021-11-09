from gluoncv.data import RecordFileDetection
from gluoncv import utils
import numpy as np
import matplotlib.pyplot as plt

class_names = ['fire']

train_dataset = RecordFileDetection('valid_data.rec', coord_normalized=True)

print( 'no. samples:', len(train_dataset))

rnd_idx = np.random.randint(len(train_dataset))

image = train_dataset[rnd_idx][0]
label = train_dataset[rnd_idx][1]
bboxes = label[:, :4]
IDs = label[:, -1]

ax = utils.viz.plot_bbox(image, bboxes, 
                         labels=IDs,
                         class_names=class_names)
plt.show()