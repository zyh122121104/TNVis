import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import os
from ultralytics import YOLO
import PIL.Image as Image
import nibabel as nib
# from skimage import data, io, segmentation, color
# from scipy.optimize import minimize
# from scipy.spatial.distance import euclidean

cutrate = .3
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # image0 = image.copy()

    mod = YOLO('./best.pt')

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        xx1, yy1, xx2, yy2 = [], [], [], []
        roi = []
        for ind in range(image.shape[0]):
            result = mod(image)
            for res in result:
                boxes = res.boxes.xyxy.tolist()
                scores = res.boxes.conf.tolist()
                if len(scores) == 0:
                    roi.append([])
                    continue
                max_index = scores.index(max(scores))

                if max(scores) < 0.5:
                    roi.append([])
                    continue
                roi.append(boxes[max_index])

        roii = []
        p = 0
        for ind in range(image.shape[0]+1):
            if ((ind % 5 == 0) & (ind != 0)) | (ind == image.shape[0]):
                roii.append([np.median(xx1), np.median(yy1), np.median(xx2), np.median(yy2)])
                xx1, yy1, xx2, yy2 = [], [], [], []
                p = 0
            if ind == image.shape[0]:
                break
            if len(roi[ind]) == 0:
                p += 1
            else:
                xx1.append(roi[ind][0])
                yy1.append(roi[ind][1])
                xx2.append(roi[ind][2])
                yy2.append(roi[ind][3])

        for ind in range(image.shape[0]):
            slice0 = image[ind, :, :]
            pre = np.zeros_like(slice0)
            img = np.array(Image.open(img_path))
            img_h, img_w = np.shape(img)[0], np.shape(img)[1]
            idx = ind // 5
            x1, y1, x2, y2 = roii[idx]

            if np.isnan(x1):
                prediction[ind] = pre
                continue
            w = x2 - x1
            h = y2 - y1
            xw1 = max(int(x1 - cutrate * w), 0)
            yw1 = max(int(y1 - cutrate * h), 0)
            xw2 = min(int(x2 + cutrate * w), img_w - 1)
            yw2 = min(int(y2 + cutrate * h), img_h - 1)
            slice = slice0[yw1:yw2, xw1:xw2]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                pre[yw1:yw2, xw1:xw2] = pred
                prediction[ind] = pre
    else:
        image0 = image.copy()
        x, y = image.shape[0], image.shape[1]
        roi = [0, 0, x, y]
        result = mod(image0)
            for res in result:
                boxes = res.boxes.xyxy.tolist()
                scores = res.boxes.conf.tolist()
                if len(scores) == 0:
                    continue
                max_index = scores.index(max(scores))

                if max(scores) < 0.5:
                    continue
                roi = boxes[max_index]

        image = image[roi[1]:roi[3], roi[0]:roi[2]]

        img = image.copy()
        if x != patch_size[0] or y != patch_size[1]:
            img = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = torch.from_numpy(img).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                prediction = out
        if len(image0.shape) == 3:
            pred = np.zeros_like(image0[:,:,0])
        else:
            pred = np.zeros_like(image0)

        try:
            pred[roi[1]:roi[3], roi[0]:roi[2]] = prediction
        except:
            pred = prediction


    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(pred == i, label == i))

    p = True
    if p:
        img = image0.astype(np.float32)
        pre = pred.astype(np.uint8)*255
        lab = label.astype(np.float32)*255
        path = './test_result/model'
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(path + '/' + case + "_img.png", img)
        cv2.imwrite(path + '/' + case + "_pre.png", pre)
        cv2.imwrite(path + '/' + case + "_gt.png", lab)

    if test_save_path is not None:
        os.makedirs(test_save_path + '/' + case.split('/')[0] + '/' + case.split('/')[1], exist_ok=True)
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+ case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def find_w_h(img):
    ret, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    binary_image = binary_image.astype('uint8')
    
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_contour = max(contours, key=cv2.contourArea)
    
    (x, y), (width, height), angle = cv2.fitEllipse(max_contour)
    
    return height, width

def fill_holes(image):
    img = image.copy()

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(img, [contour], 0, 255, cv2.FILLED)

    return img

def fill_holes_3D(volume):
    vol = volume.copy()
    for i in range(vol.shape[0]):
        vol[i] = fill_holes(vol[i])
    return vol

def post_processing():
    nii_file1 = nib.load('./D_pred.nii.gz')
    img1 = nib.load('./D_img.nii.gz')
    img1 = img1.get_fdata()
    img1 = np.array(img1)
    
    data1 = nii_file1.get_fdata()
    
    a1 = np.array(data1)
    img = a1.copy()
    img[img>0] == 255
    img = img.astype(np.uint8)
    numpy_array1 = fill_holes_3D(img)
    numpy_array1 = numpy_array1 / 255
    
    
    w1 = []
    h1 = []
    ind1 = np.where(numpy_array1 == 1)
    minindex1 = np.min(ind1, axis=1)
    maxindex1 = np.max(ind1, axis=1)
    for i in range(numpy_array1.shape[2]):
        indices1 = numpy_array1[:,:,i]
        p = np.where(numpy_array1[:,:,i] == 1)
        if np.size(p) <= 5:
            continue
        width, height = find_w_h(indices1)
    
        w1.append(width)
        h1.append(height)
    width1 = max(w1)
    height1 = h1[w1.index(max(w1))]
    depth1 = maxindex1[2] - minindex1[2] + 1
    
    nii_file2 = nib.load('./L_pred.nii.gz')
    data2 = nii_file2.get_fdata()
    numpy_array2 = np.array(data2)
    w2 = []
    h2 = []
    ind2 = np.where(numpy_array2 == 1)
    minindex2 = np.min(ind2, axis=1)
    maxindex2 = np.max(ind2, axis=1)
    for i in range(numpy_array2.shape[2]):
        indices2 = numpy_array2[:,:,i]
        p = np.where(numpy_array2[:,:,i] == 1)
        if np.size(p) <= 5:
            continue
        width, height = find_w_h(indices2)
    
        w2.append(width)
        h2.append(height)
    width2 = max(w2)
    height2 = h2[w2.index(max(w2))]
    depth2 = maxindex2[2] - minindex2[2] + 1
    scale_factor = (depth1/(width2*height1/height2), depth1/(width2*height1/height2), 1.0)
    resized_data = zoom(numpy_array1, scale_factor, order=1).transpose(2,1,0)
    resized_img = zoom(img1, scale_factor, order=1).transpose(2,1,0)
    prd_itk = sitk.GetImageFromArray(resized_data.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 1))
    sitk.WriteImage(prd_itk, './pred.nii.gz')
    img_itk = sitk.GetImageFromArray(resized_img.astype(np.float32))
    img_itk.SetSpacing((1, 1, 1))
    sitk.WriteImage(img_itk, './img.nii.gz')

