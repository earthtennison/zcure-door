"""
This module is used for feature extraction
"""
from keras_vggface.vggface import VGGFace
import time
from keras import backend as K
from numpy.core.arrayprint import dtype_is_implied
import tensorflow as tf
import cv2
import numpy as np

K.clear_session()
device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

class RecognitionModel():

    def __init__(self, model_path=None):
        self.vgg_features = None
        if model_path is None:
            #can select vgg16 resnet50 senet50
            self.vgg_features = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')
            print("pretrained RESNET50 initiated")
        else:
            self.vgg_features = feat_extract = tf.keras.models.load_model(model_path)
            print(f"trained {model_path} initiated")

        #open the model once for faster inference
        img = np.random.randint(0,255,size=(224,224,3))
        img = np.asarray((img,), dtype=np.float64)
        result = self.vgg_features.predict(img)

    def preprocess(self,img_raw):
        """
        :param img_raw: an image cropped from face detection model
        :return: array of preprocessed images, ready to be put in feature extractor model
        """
        img = np.copy(img_raw)
        
        #resize
        img = cv2.resize(img, (224,224))
        
        img = np.asarray(img, dtype=np.float64)

        #BGR -> RGB
        img = img[:,:,::-1] #faster than cv2

        #normalize the channel color to be 0 at mean
        #ref: github rcmalli/keras-vggface
        # img[:,:,0] -= -93.5940 
        # img[:,:,1] -= -103.8827
        # img[:,:,2] -= -129.1863

        img = np.asarray((img,))

        return img

    def extract(self,img):
        """
        preprocess and extract an image to feature

        :param img: an image array with shape (224,224,3)
        :return: numpy array with shape (,x)
        """
        img_pre = self.preprocess(img)
        result = self.vgg_features.predict(img_pre)
        #result = tf.math.l2_normalize(result)
        return result

    def recognize(self,feat, regis_feat, thresh=100, debug=False):
        """
        find the most match of a feature to all registered feature

        :param feat: extracted feature from vgg_face
        :type feat: numpy array in shape (1,x)
        :param regis_feat: list of registered features
        :type regis_feat: list
        :param thresh: threshold for comparing distance
        :type thresh: int

        :return: index of regis_feat that has minimum distance. If not found return None
        """
        regis_feat= np.asarray(regis_feat)
        distances = [np.linalg.norm(f) for f in regis_feat-feat]
        #debug
        print(f"distances debug : {distances}")
        if(np.min(distances) <= thresh):
            if debug:
                return np.argmin(distances), np.min(distances), distances
            else:
                return np.argmin(distances)
        else:
            return None

if __name__ == "__main__":

    model = RecognitionModel("model/model-2.h5")
    img = cv2.imread("sample_image/earth3_cropped.jpg")
    cv2.imshow("img",img)
    result = model.extract(img)
    print(type(result))
    print(result.shape)
    print(result)

    cv2.waitKey(0)

    #img_batch = np.asarray([img,img,img])
    #result_batch = extract(img_batch, batch=True)
    #print(result_batch.shape)
    #print(result_batch)



