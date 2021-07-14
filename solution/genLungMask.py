import os
from solution.external.lungmask import mask
from solution.dcm2png import get_pixels_hu_by_simpleitk
import SimpleITK as sitk
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2


class GenLungMask(object):
    def __init__(self, dcms_path):
        self.dcms_path = dcms_path

    def isDicomFile(self, filename):
        '''
           判断某文件是否是dicom格式的文件
        :param filename: dicom文件的路径
        :return:
        '''
        file_stream = open(filename, 'rb')
        file_stream.seek(128)
        data = file_stream.read(4)
        file_stream.close()
        if data == b'DICM':
            return True
        return False

    def loadPatient(self, src_dir):
        '''
            读取某文件夹内的所有dicom文件
        :param src_dir: dicom文件夹路径
        :return: dicom list, slice_location
        '''
        files = os.listdir(src_dir)
        slices = []
        slice_dict = {}
        for s in files:
            if self.isDicomFile(src_dir + os.sep + s):
                instance = pydicom.dcmread(src_dir + os.sep + s)
                slice_dict[s] = instance.SliceLocation # key: slice_name, value: slice_location
                slices.append(instance)
        slices.sort(key=lambda x: int(x.InstanceNumber))
        sorted(slice_dict, key=lambda kv: (kv[1], kv[0]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness
        return slice_dict

    def genLung(self):
        """
        dcms_path: dicoms path directory
        """
        # 读取dicom文件的元数据(dicom tags)
        slice_with_location = self.loadPatient(self.dcms_path)
        slice_with_location = sorted(slice_with_location.items(), key=lambda kv : kv[1], reverse=True) # descending order by SliceLocation
        dcm_list = [i[0] for i in slice_with_location] # os.listdir may not have the correct order,
                                                    # so we order it by SliceLocation
        for i in range(len(dcm_list)):
            dcm_list[i] = os.path.join(self.dcms_path, dcm_list[i])
        input_image = sitk.ReadImage(dcm_list)
        seg = mask.apply(input_image) # generate mask (number of slices, 512, 512)
        pngs = get_pixels_hu_by_simpleitk(dcm_list) # generate png
        print('The number of dicom files : ', len(slice_with_location))
        dataset = []
        for i in range(len(slice_with_location)):
            pngs[i][seg[i] == 0] = 0 # add mask
            tmp = np.concatenate([[pngs[i], pngs[i], pngs[i]]])# convert to 3 channels
            dataset.append(tmp)
        dataset = np.swapaxes(dataset, 1, 3)
        return dcm_list, dataset # return pngs to classification model


if __name__ == '__main__':
    glm = GenLungMask(r'C:\Users\DNY-004\Desktop\cases\T1-001', None)
    glm.genLung()