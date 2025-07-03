import pydicom
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot
import pandas as pd


file_name = "IMG0002.dcm"
file_name_ct = "3.dcm"

def read_dcm():
    
    ds_ct = pydicom.dcmread(file_name_ct)
    ds_pixelarray = ds_ct.pixel_array
    slope = float(ds_ct.RescaleSlope) if 'RescaleSlope' in ds_ct else 1
    intercept = float(ds_ct.RescaleIntercept) 
    hu_image = ds_pixelarray * slope + intercept

    hu_min, hu_max = -1000, 1000 # ?肺部建议是-1000~400，原文用的是-1000~1000
    # 感觉这两个上限值都ok
    
    img = np.clip(hu_image, hu_min, hu_max)

    hu_image_uint8 = ((img + 1000) / 2000 * 255).astype(np.uint8)  # 归一化到0~255

    #保存图像
    img = Image.fromarray(hu_image_uint8)
    img.save('hu_image_1000.png')


    print('ok')
    

def read_excel():
    file_path = 'ceshi.xlsx'
    # 读取xlsx文件
    data = pd.read_excel(file_path, sheet_name='Sheet2')

    # 显示数据
    # print(data)

    # 取第2列和第7列
    # data_new = data.iloc[:, [2, 7]]

    # 取第16至28行
    data_new = data.iloc[14:26, :]
    # 这里的data_new是一个DataFrame对象
    # 如果需要转换为numpy数组，可以使用values属性

    # data_new_df = data_new.iloc[14:16, :]
    data_new_df = data_new[['姓名', '肺结节', '肺结节部位', 'CT上界', 'CT下界', 'PET上界', 'PET下界']]

    print(data_new_df)
    
    filename = data_new_df.at[14, '姓名']
    filepath = '26-40/26-40/'
    img_path = os.path.join(filepath, filename, 'images')

    all_files = os.listdir(img_path)
    
    print(all_files)

    ct_path = os.path.join(img_path, all_files[1])
    # 按照在文件夹中的顺序读取并排列
    ct_slices_files = os.listdir(ct_path)

    # 排序不对,按照文件名数字排序，而不是字母顺序，文件名只有.dcm前的数字不同，数字为?????.123.dcm的形式
    # 例如：x.1.dcm, x.2.dcm, x.3.dcm,....,x.204.dcmcvbnmkl;'] 
    ct_slices_files.sort(key=lambda x: int(x.split('.')[-2]))  # 按照文件名数字排序

    ct_slices = [os.path.join(ct_path, f) for f in ct_slices_files]
    ct_length = len(ct_slices)

    ct_up = int(ct_length - 1 - data_new_df.at[14, 'CT上界'])
    ct_down = int(ct_length - 1 - data_new_df.at[14, 'CT下界'])
    
    ct_slices_new = ct_slices[ct_up:ct_down + 1]  # 注意这里的切片是左闭右开区间

    ct_img = [pydicom.dcmread(f) for f in ct_slices_new]
    
    # 优化版代码，先确定区间，再读取数据
    # ct_slices_new = [pydicom.dcmread(f) for f in ct_slices[ct_up:ct_down + 1]]
    # ct_imgs = [s.PixelData for s in ct_slices_new]

    # 提取像素数据
    # ct_images = [s.pixel_array for s in ct_slices_new]
    ct_images = [s.pixel_array for s in ct_img]

    # 将像素数据转换为numpy数组
    ct_images_np = np.array(ct_images)
    print(ct_images_np.shape)

    print(data_new)

    print('!')


def read_excel2():
    file_path = 'ceshi.xlsx'
    # 读取xlsx文件
    data = pd.read_excel(file_path, sheet_name='Sheet2')

    # 显示数据
    # print(data)

    # 取第2列和第7列
    # data_new = data.iloc[:, [2, 7]]

    # 取第16至28行
    data_new = data.iloc[14:26, :]
    # 这里的data_new是一个DataFrame对象
    # 如果需要转换为numpy数组，可以使用values属性

    # data_new_df = data_new.iloc[14:16, :]
    data_new_df = data_new[['姓名', '肺结节', '肺结节部位', 'CT上界', 'CT下界', 'PET上界', 'PET下界']]

    print(data_new_df)
    
    filename = data_new_df.at[14, '姓名']
    filepath = '26-40/26-40/'
    img_path = os.path.join(filepath, filename, 'images')

    all_files = os.listdir(img_path)
    
    print(all_files)

    ct_path = os.path.join(img_path, all_files[0])
    # 按照在文件夹中的顺序读取并排列
    ct_slices_files = os.listdir(ct_path)

    # 排序不对,按照文件名数字排序，而不是字母顺序，文件名只有.dcm前的数字不同，数字为?????.123.dcm的形式
    # 例如：x.1.dcm, x.2.dcm, x.3.dcm,....,x.204.dcmcvbnmkl;'] 
    # ct_slices_files.sort(key=lambda x: int(x.split('.')[-2]))  # 按照文件名数字排序

    ct_slices = [os.path.join(ct_path, f) for f in ct_slices_files]

    ct_img = [pydicom.dcmread(f) for f in ct_slices]

    # 按照ct_img中的每个变量的属性ImageIndex排序从小到大
    ct_img.sort(key=lambda x: x.ImageIndex)
    # 这里的编号正好对应表格中的编号

    ct_up = int(data_new_df.at[14, 'PET上界'])
    ct_down = int(data_new_df.at[14, 'PET下界'])
    
    ct_img_new = ct_img[ct_down:ct_up + 1]  # 注意这里的切片是左闭右开区间

    # ct_img = [pydicom.dcmread(f) for f in ct_slices_new]
    
    # 优化版代码，先确定区间，再读取数据
    # ct_slices_new = [pydicom.dcmread(f) for f in ct_slices[ct_up:ct_down + 1]]
    # ct_imgs = [s.PixelData for s in ct_slices_new]

    # 提取像素数据
    # ct_images = [s.pixel_array for s in ct_slices_new]
    ct_images = [s.pixel_array for s in ct_img_new]

    # 将像素数据转换为numpy数组
    ct_images_np = np.array(ct_images)
    print(ct_images_np.shape)

    print(data_new)

    print('!')


def read_dcm2():

    ds_ct = pydicom.dcmread(file_name_ct)
    ds_pixelarray = ds_ct.pixel_array
    slope = float(ds_ct.RescaleSlope) if 'RescaleSlope' in ds_ct else 1
    intercept = float(ds_ct.RescaleIntercept) 
    pet_image = ds_pixelarray * slope + intercept

    # hu_min, hu_max = 0, 4000 # ?肺部建议是-1000~400，原文用的是-1000~1000 感觉这两个上限值都ok
    # img = np.clip(hu_image, hu_min, hu_max)
    # hu_image_uint8 = ((img + 1000) / 2000 * 255).astype(np.uint8)  # 归一化到0~255
    #     #保存图像
    # img = Image.fromarray(hu_image_uint8)
    # img.save('hu_image_1000.png')

    pet_min, pet_max = np.min(pet_image), np.max(pet_image)  # pet没有固定的物理单位窗口
    pet_img = ((pet_image - pet_min) / (pet_max - pet_min) * 255).astype(np.uint8)

    pyplot.imsave('pet_color.png', pet_img, cmap='hot')
    # # 保存图像
    # img = Image.fromarray(pet_img)
    # img.save('pet_image.png')
    
    print('PET图像已保存为 pet_image.png')

    print('ok')


    print('ok')

def main():
    read_excel()
    read_dcm()


if __name__ == "__main__":
    main()
