import numpy as np
def process_nmstrt_result(num_detections,nmsed_boxes,nmsed_scores,nmsed_classes):
    nmsed_scores = np.expand_dims(nmsed_scores, axis=-1)
    boxes_and_scores = np.concatenate((nmsed_boxes, nmsed_scores), axis=-1)
    labels = nmsed_classes
    return labels, boxes_and_scores