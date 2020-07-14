import sys
from keras.models import load_model
from keras_retinanet import models as md
from keras import backend as K
# sys.path.append("D:/Suvajit/AICoE/P&G Germany/Phase 2/")

import cv2

import classification as cl
import objectdet as od

def load_model(model_type):
    """
    Returns the classification model / Object Detection model weights to be used for scoring

    :param model_type: Classification or Object_Detection

    """
    import sys
    from keras.models import load_model
    from keras_retinanet import models as md
    from keras import backend as K
    import cv2

    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    if model_type=="classification":
        model = load_model("classification_model_weights_v1.model",custom_objects={'f1': f1})
    else:
        model = md.load_model("retinanet_new_model_weights.h5", backbone_name='resnet50')

    return model



"""
Test Run ---

"""

# out1 = cl.classification("VISUCI1_S1_20200417_175457_071_1.jpg",load_model("Classification"),"")
# out2 = od.obj_detection("VISUCI1_S1_20200417_175457_071_1.jpg",load_model("Object_Detection"),"D:/Suvajit/AICoE/P&G Germany/Phase 2/")


# print(out1)

# print(out2[0])
#cv2.imshow('image',out[1])
#cv2.waitKey(0)