import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm
from openpyxl import load_workbook
import pydicom

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

# 训练时运用的主要数据集
class CTReportDataset(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder  # 数据文件夹路径
        self.min_slices = min_slices  # 最小切片数
        self.accession_to_text = self.load_accession_text(csv_file)  
        # 读取CSV文件，获取AccessionNo和对应的文本信息
        self.paths=[]
        self.samples = self.prepare_samples()
        percent = 80
        num_files = int((len(self.samples) * percent) / 100)
        # num_files的值为样本总数的80%，用于训练集
        #num_files = 2286
        self.samples = self.samples[:num_files]
        print(len(self.samples))
        self.count = 0


        #self.resize_dim = resize_dim
        #self.resize_transform = transforms.Resize((resize_dim, resize_dim))
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        # 读取csv，将每个影像编号对应的英文描述和印象存入字典
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['AccessionNo']] = row["Findings_EN"],row['Impressions_EN']

        return accession_to_text

    # 准备样本  
    def prepare_samples(self):
        samples = []
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):

                for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
                    accession_number = nii_file.split("/")[-1]
                    #accession_number = accession_number.replace(".npz", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]

                    if impression_text == "Not given.":
                        impression_text=""

                    input_text_concat = ""
                    for text in impression_text:
                        input_text_concat = input_text_concat + str(text)
                    input_text_concat = impression_text[0]
                    input_text = f'{impression_text}'
                    samples.append((nii_file, input_text_concat))
                    self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)



    def nii_img_to_tensor(self, path, transform):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        df = pd.read_csv("train_metadata.csv") #select the metadata
        file_name = path.split("/")[-1]
        row = df[df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        img_data = slope * img_data + intercept

        img_data = img_data.transpose(2, 0, 1)

        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data ) / 1000)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        return tensor


    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return video_tensor, input_text
    # 返回的是原始文本（并没有经过分词）以及重采样后的3D影像。
  

# 自建数据集
# 根据需要写，直接修改Trainer文件即可
class VideoDatasetWithLabels(Dataset):
    def __init__(
            self,
            data_path, # 到姓名文件夹的路径
            csv_file_path,  # name or path? 
            min_slices=20,
            resize_dim=500, 
            force_num_frames=True):
        self.data_path = data_path  # 数据文件夹路径
        
        # 先读取csv文件，提取需要的信息，dataframe形式？numpy形式
        self.meta_csv_path = csv_file_path
        self.min_slices = min_slices  # 最小切片数
        self.data_table = pd.read_excel(self.meta_csv_path, sheet_name='Sheet2')  # 假设csv文件包含影像路径和文本信息

        # 按照目前的格式
        self.data_table = self.data_table[['姓名', '肺结节部位', 'CT上界', 'CT下界', 'PET上界', 'PET下界']]

    def __len__(self):
        return len(self.data_table)

    # CT影像处理
    def process_ct_slices(self, ct_array):
        pixel_array = [s.pixel_array for s in ct_array]
        slope = float(ct_array[0].RescaleSlope)
        intercept = float(ct_array[0].RescaleIntercept)
        pixel_array = [slope * p + intercept for p in pixel_array]

        hu_max = 1000
        hu_min = -1000
        hu_img_array = [np.clip(p, hu_min, hu_max) for p in pixel_array]
        img_uint8 = [np.uint8((p - hu_min) / (hu_max - hu_min) * 255) for p in hu_img_array]
        img_tensor = [torch.tensor(p) for p in img_uint8]
        img_tensor = torch.stack(img_tensor, dim=0)
        img_tensor = img_tensor.unsqueeze(0)  # 增加一个维

        return img_tensor
    
    # PET影像处理
    def process_pet_slices(self, pet_array):
        pixel_array = [s.pixel_array for s in pet_array]
        slope = float(pet_array[0].RescaleSlope)
        intercept = float(pet_array[0].RescaleIntercept)
        pixel_array = [slope * p + intercept for p in pixel_array]

        pet_min = np.min(pixel_array)
        pet_max = np.max(pixel_array)
        pet_img = [((p - pet_min) / (pet_max - pet_min) * 255).astype(np.uint8) for p in pixel_array]
        pet_tensor = [torch.tensor(p) for p in pet_img]
        pet_tensor = torch.stack(pet_tensor, dim=0)
        pet_tensor = pet_tensor.unsqueeze(0)  # 增加一个维度

        return pet_tensor


    def __getitem__(self, index):
        # 文本信息
        file_name = self.data_table.iloc[index]['姓名']  
        img_path = os.path.join(self.data_path, file_name, 'images')  
        text = self.data_table.iloc[index]['肺结节部位']

        # 如果是单次训练，直接使用文本信息
        text = [text]

        # 读取影像数据
        all_files = os.listdir(img_path)

        # CT
        ct_path = os.path.join(img_path, all_files[1])
        ct_slices_list = os.listdir(ct_path)
        ct_slices_list.sort(key=lambda x: int(x.split('.')[-2]))
        # 先排序，后选择
        
        ct_length = len(ct_slices_list)
        ct_up = int(ct_length - 1 - self.data_table.iloc[index]['CT上界'])
        ct_down = int(ct_length - 1 - self.data_table.iloc[index]['CT下界'])
        ct_slices_list_selected = ct_slices_list[ct_up:ct_down + 1]  # 选择指定范围的切片
        ct_slices_path = [os.path.join(ct_path, f) for f in ct_slices_list_selected] # 读取第一个切片的路径  
        
        ct_imgs = [pydicom.dcmread(f) for f in ct_slices_path]  # 读取指定范围的切片
        ct_tensor = self.process_ct_slices(ct_imgs)

        # PET
        pet_path = os.path.join(img_path, all_files[0])
        pet_slices_list = os.listdir(pet_path)
        pet_slices_path = [os.path.join(pet_path, f) for f in pet_slices_list]
        pet_imgs = [pydicom.dcmread(f) for f in pet_slices_path]
        pet_imgs.sort(key=lambda x: x.ImageIndex)

        pet_up = int(self.data_table.iloc[index]['PET上界'])
        pet_down = int(self.data_table.iloc[index]['PET下界'])

        pet_imgs_selected = pet_imgs[pet_down:pet_up + 1]  # 选择指定范围的切片

        pet_tensor = self.process_pet_slices(pet_imgs_selected)

        return text, ct_tensor, pet_tensor
    
'''
    # def __init__(
    #         self, 
    #         data_folder, 
    #         csv_file, 
    #         min_slices=20, 
    #         resize_dim=500, 
    #         force_num_frames=True):
    #     self.data_folder = data_folder  # 数据文件夹路径
    #     self.min_slices = min_slices  # 最小切片数
    #     self.accession_to_text = self.load_accession_text(csv_file)
    #     # 替换成读取文本标签  
    #     # 读取CSV文件，获取AccessionNo和对应的文本信息
    #     self.paths=[]
    #     self.samples = self.prepare_samples()
    #     # 构造文本标签与影像对，一对多？思考清楚逻辑
    #     percent = 80
    #     num_files = int((len(self.samples) * percent) / 100)
    #     # 选择样本总数的80%作为训练集
    #     #num_files = 2286
    #     self.samples = self.samples[:num_files]
    #     # 全部把数据加载到samples中？内存够吗？为什么不边训练边读取呢？

    #     print(len(self.samples))
    #     self.count = 0


    #     #self.resize_dim = resize_dim
    #     #self.resize_transform = transforms.Resize((resize_dim, resize_dim))
    #     self.transform = transforms.Compose([
    #         transforms.Resize((resize_dim,resize_dim)),
    #         transforms.ToTensor()
    #     ])
    #     self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)


        

    # def __getitem__(self, index):
    #     tensor = super().__getitem__(index)
    #     label = self.labels.get(self.paths[index].name, None)
    #     return tensor, label
'''


def main():
    # data_folder = 'path_to_data_folder'
    # csv_file = 'path_to_csv_file'
    # dataset = CTReportDataset(data_folder, csv_file)
    
    # for i in range(len(dataset)):
        # video_tensor, input_text = dataset[i]
        # print(f"Video Tensor Shape: {video_tensor.shape}, Input Text: {input_text}")
    # xlsx_file_path = '/data1/users/fangyifan/Works/Paper_Code/CT-CLIP/scripts/肺小结节诊断new.xlsx'
    # wb = load_workbook(filename=xlsx_file_path)
    # sheet = wb['人民2018-2020']

    # # 读取单元格内容
    # cell_value = sheet['A1'].value
    # print(f"Cell A1 value: {cell_value}")

    # # 读取所有行
    # row_values = [cell.value for cell in sheet['A']]  # 读取第一行
    # print(f"Row A values: {row_values}")

    # # 读取所有列
    # col_values = [cell.value for cell in sheet['1']]  # 读取第一
    # print(f"Column 1 values: {col_values}")

    # print('ok!')

    file_path = '/data1/users/fangyifan/Works/Paper_Code/CT-CLIP/scripts/肺小结节诊断new.xlsx'
    # file_path = '肺小结节诊断new.xlsx'

    # 读取xlsx文件
    data = pd.read_excel(file_path, sheet_name='人民2018-2020')

    # 显示数据
    print(data)

    print("Data loaded successfully!")




if __name__ == "__main__":
    main()
