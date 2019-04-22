# -*- coding: utf-8 -*-
"""
@author: lisssse14
"""
import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array, load_img, array_to_img
import numpy as np
import cv2
# 自動增加 GPU 記憶體用量
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
# 設定 Keras 使用的 Session
K.set_session(sess)
K.set_learning_phase(0)

class gradient_cams(object):
    def __init__(self, model, image_path, target_layer):
        self.model = model
        self.image = image_path
        self.target_layer = target_layer

    def preprocess_image(self,pre_process = True):
        x = img_to_array(self.image)
        x = np.expand_dims(x, axis=0)
        if pre_process:
            x = preprocess_input(x)
        return x
     
    def grad_cam(self, image):
        prediction = model.predict(image)
        pred_class = np.argmax(prediction[0])
        #pred_class_name = decode_predictions(prediction)[0][0][1]
        pred_output = model.output[:, pred_class]
        last_conv_output = model.get_layer(target_layer).output
        #梯度公式
        grads = K.gradients(pred_output, last_conv_output)[0]
        #定義計算函數
        gradient_function = K.function([model.input], [last_conv_output, grads])
        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]
        # 取得所有梯度的平均值(维度降低)
        weights = np.mean(grads_val, axis=(0, 1))
        gradcam = np.dot(output, weights)
        #梯度RGB化
        gradcam = cv2.resize(gradcam, (image.shape[1], image.shape[2]), cv2.INTER_LINEAR)
        gradcam = np.maximum(gradcam, 0)
        heatmap = gradcam / gradcam.max()
        # 上色
        jetcam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + image / 2)
        return jetcam

    
    def grad_cam_plus_plus(self, image):
        img = image /255.0
        predictions = model.predict(img)
        class_idx = np.argmax(predictions[0])
        class_output = model.layers[-1].output
        conv_output = model.get_layer(target_layer).output
        grads = K.gradients(class_output,conv_output)[0]
        # 一階微分
        first_derivative = K.exp(class_output)[0][class_idx] * grads
        #二階微分
        second_derivative = K.exp(class_output)[0][class_idx] * grads * grads
        # 三階微分
        third_derivative = K.exp(class_output)[0][class_idx] * grads * grads * grads
        gradient_function = K.function([model.input], [conv_output, first_derivative, second_derivative, third_derivative])
        conv_output, conv_first_grad, conv_second_grad, conv_third_grad = gradient_function([img])
        conv_output, conv_first_grad, conv_second_grad, conv_third_grad = conv_output[0], conv_first_grad[0], \
                                                                      conv_second_grad[0], conv_third_grad[0]
        weights = np.maximum(conv_first_grad, 0.0)
        global_sum = np.sum(conv_output.reshape((-1, conv_first_grad.shape[2])), axis=0)
        alpha_denom = conv_second_grad * 2.0 + conv_third_grad * global_sum.reshape((1, 1, conv_first_grad.shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alpha_num = conv_second_grad
        alphas = alpha_num / alpha_denom
        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
        alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant,
                                                      np.ones(alpha_normalization_constant.shape))
        alphas /= alpha_normalization_constant_processed.reshape((1, 1, conv_first_grad.shape[2]))
        
        deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad.shape[2])), axis=0)
        
        grad_CAM_map = np.sum(deep_linearization_weights * conv_output, axis=2)
        grad_CAM_map = np.maximum(grad_CAM_map, 0)
        grad_CAM_map = grad_CAM_map / np.max(grad_CAM_map)
        grad_CAM_map = cv2.resize(grad_CAM_map, (image.shape[1], image.shape[2]), cv2.INTER_LINEAR)
        jetcam = cv2.applyColorMap(np.uint8(255 * grad_CAM_map), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + image / 2)
        return jetcam
    
if __name__ == '__main__':
    model = ResNet50(weights='imagenet', include_top=True)
    image_path = 'boat.jpg'
    target_height, target_width = (224,224)
    image = load_img(image_path, color_mode='rgb')
    #original_height, original_width = (image.height, image.width)
    image = image.resize((target_height,target_width),3)
    target_layer = 'res5c_branch2c'
    vis = gradient_cams(model, image, target_layer)
# =============================================================================
#     grad_cam
# =============================================================================
    preprocess = vis.preprocess_image()
    grad = vis.grad_cam(preprocess)
    result = grad.reshape(target_height,target_height,3)
    cv2.imwrite('gcam.jpg', result)
# =============================================================================
#     grad_cam++
# =============================================================================
    preprocess = vis.preprocess_image(pre_process=False)
    grad_plus = vis.grad_cam_plus_plus(preprocess)
    result = grad_plus.reshape(target_height,target_height,3)
    cv2.imwrite('gcam++.jpg', result)