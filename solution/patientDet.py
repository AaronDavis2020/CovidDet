import os
import pandas as pd
import torch
from solution.inference import slicePredict

#load the result of the prediction of the slices in the result.csv, return a dic which the key represent the patient ID,
# the value is a str which represent the predic of all the slices of one patient
def readData(inference, case_path):
    label_data = inference
    row_amount = len(label_data.iloc[:,0])
    f_dir = {'T1': 'SPGC-Test1', 'T2': 'SPGC-Test2', 'T3': 'SPGC-Test3'}
    patient_dic = {}
    for i in range(row_amount):
        img_path=label_data.iloc[:,0][i] # 第0列
        label_pre = label_data.iloc[:,1][i] # 第1列
        dcm_name = img_path.split('.')[0]
        patient_dic[dcm_name] = label_pre
        # patient_num = img_path.split('.')[0].split('_')[0]
        # dcm = pydicom.dcmread(os.path.join(case_path, dcm_name + '.dcm'))
        # SliceLocation = dcm.SliceLocation
        # if patient_num in patient_dic.keys():
        #     patient_dic[patient_num].append([str(label_pre),float(SliceLocation)])
        # else:
        #     patient_dic[patient_num] = [[str(label_pre),float(SliceLocation),]]
    # for key in patient_dic.keys():
    #     patient_dic[key] = sorted(patient_dic[key],key=lambda x:x[1],reverse=True)
    #     patient_dic[key] = ''.join(patient_dic[key][i][0] for i in range(0,len(patient_dic[key])))
    return patient_dic

def classificationRule(a, b, c):
    max = 0
    if a > b:
        if b > c:
           max =a
        else:
            if a > c:
                max = a
            else:
                max = c
    else:
        if a > c:
            max = b
        else:
            if b > c:
                max = b
            else:
                max = c
    return max

def patientDet(patient_dic):
    predic_list = []
    length = len(patient_dic)
    normal_n = 0
    cap_n = 0
    covid_n = 0
    for j in patient_dic.values():
        if j == 0:
            normal_n += 1
        elif j == 1:
            cap_n += 1
        else:
            covid_n += 1
    max = classificationRule(a=normal_n, b=cap_n, c=covid_n)
    if max == covid_n:
        predic_list.append(['patient', 'covid'])
    elif max == cap_n:
        predic_list.append(['patient', 'cap'])
    else:
        person = float(normal_n/length)
        if person >= 0.95:
            predic_list.append(['patient', 'normal'])
        else:
            if cap_n > covid_n:
                predic_list.append(['patient', 'cap'])
            else:
                predic_list.append(['patient', 'covid'])
    return predic_list


if __name__ == '__main__':
    # Use CUDA
    torch.cuda.set_device(0)
    use_cuda = torch.cuda.is_available()
    inference = slicePredict(use_cuda)

    pat_dic = readData(inference)

    predic_list = patientDet(patient_dic=pat_dic)

    desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
    name = ['Patient','Class']
    test = pd.DataFrame(columns=name, data=predic_list)
    test.to_csv(os.path.join(desktop_path, "Covid19_detResult.csv"), encoding='gbk')

