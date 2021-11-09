import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from altusi.utils.logger import *
import altusi.utils.visualizer as vis
import mxnet as mx
import gluoncv as gcv

ctx = mx.context.cpu()

LOG(INFO, 'Device in Use:', ctx)
classes = ['fire']
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, ctx=ctx)
net.load_parameters('ssd_512_mobilenet1.0_voc_best.params')

net.collect_params().reset_ctx(ctx)
LOG(INFO, 'SSDFace Model loading done')

def filter_bboxes(bboxes, scores, class_ids, thresh=0.8):
    ids = np.where(scores.asnumpy().reshape(-1) > thresh)[0]

    if len(ids):
        return bboxes[ids], scores[ids], class_ids[ids]
    else:
        return None, None, None


def ssd_predict(net, image, ctx, thresh=0.5):
    x, img = gcv.data.transforms.presets.ssd.transform_test(mx.nd.array(image), short=300)
    x = x.as_in_context(ctx)

    class_ids, scores, bboxes = net(x)

    if len(bboxes[0]) > 0:
        bboxes, scores, class_ids = filter_bboxes(bboxes[0], scores[0], class_ids[0], thresh)

        if bboxes is not None:
            classes = [net.classes[int(idx.asscalar())] for idx in class_ids]

    return class_ids, scores, bboxes, img

def rescale_bboxes(bboxes, dims, new_dims):
    H, W = dims
    _H, _W = new_dims
    
    _bboxes = []
    for bbox in bboxes:
        bbox = bbox.asnumpy()
        bbox = bbox / np.array([W, H, W, H]) * np.array([_W, _H, _W, _H])
        _bboxes.append(bbox)
        
    return _bboxes

start = time.time()
test_img_path = 'Data1'


for fname in os.listdir(test_img_path):
    image_fname = os.path.join(test_img_path,fname)
    # read and test
    image = cv.imread(image_fname)
    class_ids, scores, bboxes, ssd_image = ssd_predict(net, image, ctx)

    if bboxes is not None:
        bboxes = rescale_bboxes(bboxes, ssd_image.shape[:2], image.shape[:2])
        image = vis.drawObjects(image, bboxes, color=vis.COLOR_RED_LIGHT)
        cv.imwrite(f'results/{fname}', image)
    else:
        cv.imwrite(f'false/{fname}', image)
end = time.time()
print("Prediction Time:-",round(end-start,2))