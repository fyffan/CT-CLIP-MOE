from pathlib import Path
from shutil import rmtree
from datetime import timedelta

from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from data import CTReportDataset, VideoDatasetWithLabels
from data_inference import CTReportDatasetinfer

import numpy as np
import pandas as pd
import tqdm


from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# helpers
def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array



def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): The path to save the NIfTI file.
        affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
    """

    tensor = tensor.cpu()

    if tensor.dim() == 4:
        # Assume single channel data if there are multiple channels
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))

        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma

class CTClipTrainer(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        num_train_steps,
        batch_size,
        data_train = "train",
        data_valid = "valid",
        reports_file_train = "data_reports.xslx",
        reports_file_valid = "data_reports.xslx",
        labels = "labels.csv",
        data_xlsx = "data.xlsx",
        tokenizer = None,
        lr = 1.25e-6,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 1000,
        save_model_every = 1000 ,
        results_folder = '/shares/menze.dqbm.uzh/ihamam/ctclip/',
        num_workers = 8,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))

        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        
        # print("Accelerator devices", self.accelerator.device)
        
        # 这是一个用于加速训练的类，
        # 它结合了分布式数据并行和进程组初始化的配置，以便在多GPU环境中高效地训练模型。
        self.CTClip = CTClip
        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            self.tokenizer=BertTokenizer.from_pretrained('/data1/users/fangyifan/Works/Paper_Code/CT-CLIP/pretrained/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
            # 这行代码的作用是初始化一个医学领域专用的BERT分词器，
            # 用于将原始文本转为模型可处理的token序列：
            # 能否用于描述肺部小结节位置的中文文本？
            # 是的，这个分词器可以用于处理描述肺部小结节位置的中文文本，
            # 但需要注意的是，它是基于英文医学文本训练的，可能对中文文本的处理效果不如专门针对中文训练的分词器。

        self.register_buffer('steps', torch.Tensor([0]))
        # 这行代码的作用是注册一个名为'steps'的缓冲区，该缓冲区是一个张量，初始值为0。
        # 这个缓冲区用于跟踪训练的步数，并且在模型保存和加载时会被保留。

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        # max_grad_norm是一个超参数，用于梯度裁剪，以防止梯度爆炸。

        self.lr=lr
        # # Load the pre-trained weights
        # self.ds = CTReportDataset(data_folder=data_train, csv_file=reports_file_train)
        # # 训练CT报告数据集
        # self.valid_ds = CTReportDatasetinfer(data_folder=data_valid, csv_file=reports_file_valid, labels = labels)
        # # CT报告测试数据集

        # 新的数据集
        self.pair_ds = VideoDatasetWithLabels(
            data_path=data_train,
            csv_file_path=data_xlsx
        )

        self.pair_dl = DataLoader(
            self.pair_ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )
        # suffle为True，即每次迭代都会打乱数据的顺序，
        # 这有助于提高模型的泛化能力，防止过拟合

        self.pair_dl_iter = cycle(self.pair_dl)

        # self.dl = DataLoader(
        #     self.ds,
        #     num_workers=num_workers,
        #     batch_size=self.batch_size,
        #     shuffle = True,
        # )

        # self.valid_dl = DataLoader(
        #     self.valid_ds,
        #     num_workers=num_workers,
        #     batch_size=1,
        #     shuffle = False,
        # )


        # # prepare with accelerator
        # self.dl_iter=cycle(self.dl)
        # 这行代码的作用是创建一个无限循环的迭代器，
        # 该迭代器会不断从数据加载器self.dl中获取数据，
        # 以便在训练过程中可以不断获取批次数据进行训练。

        # 为什么不用iter(self.dl)而是cycle(self.dl)？
        # 因为iter(self.dl)只会创建一个迭代器，
        # 当迭代器耗尽时会抛出StopIteration异常，而cycle(self.dl)会创建一个无限循环的迭代器，
        # 即使数据加载器耗尽也不会抛出异常，而是会重新开始迭代。

        # self.valid_dl_iter=cycle(self.valid_dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        # (
 		# 	self.dl_iter,
        #     self.valid_dl_iter,
        #     self.CTClip,
        #     self.optim,
        # ) = self.accelerator.prepare(
        #     self.dl_iter,
        #     self.valid_dl_iter,
        #     self.CTClip,
        #     self.optim,
        # )
        # # 这行代码的作用是使用Accelerator类的prepare方法将数据加载器、模型和优化器准备好，

        # 新代码部分
        (
            self.pair_dl_iter,
            self.CTClip,
            self.optim,
        ) = self.accelerator.prepare(
            self.pair_dl_iter,
            self.CTClip,
            self.optim,
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))
            # 这行代码的作用是删除指定路径下的所有文件和子目录，以清除之前的实验检查点和结果。

        self.results_folder.mkdir(parents=True, exist_ok=True)



    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.CTClip.train()  # set model to training mode

        # logs
        logs = {}

        # update CTClip model
        # video, text = next(self.dl_iter)
        text, ct_video, pet_video = next(self.pair_dl_iter)

        for video in [ct_video, pet_video]:

            print(video.shape)
            device=self.device
            video=video.to(device)
            mask = torch.ones((video.shape[0], video.shape[2])).bool().to(device)
            # 并创建一个与视频数据形状相同的布尔掩码
            # mask的作用是指示哪些时间步是有效的（即不被遮挡的），即那些slice是肺部部分
            # 在这里，mask的形状为(batch_size, num_frames)，
            #text = text.to(device)
            text = list(text)
            text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            # 这行代码的作用是将文本数据转换为模型可处理的token序列，

            #video = video
            with self.accelerator.autocast():
                loss = self.CTClip(text_tokens, video, return_loss=True, device=device)  # forward pass
                # 计算损失值

            self.accelerator.backward(loss)
            accum_log(logs, {'loss': loss.item()})
            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)
                # 这行代码的作用是对模型参数的梯度进行裁剪，以防止梯度爆炸。

            self.optim.step()
            self.optim.zero_grad()
            self.print(f"{steps}: loss: {logs['loss']}")


        # save results every so often
        if self.is_main and not (steps % self.save_results_every):
            # 只有主进程（多卡时）且到达指定步数才会执行评估。
            with torch.no_grad():

                models_to_evaluate = ((self.CTClip, str(steps)),)

                for model, filename in models_to_evaluate:
                    # 遍历模型，逐个评估
                    model.eval()
                    predictedall=[]
                    realall=[]

                    #Fast inference on 100 images
                    # 快速推理多张图像
                    for i in range(10):
                        print("test")
                        valid_data, text, onehotlabels, name_acc = next(self.valid_dl_iter)
                        valid_data = valid_data.to(device)

                        if "module" in model.__dict__:
                            model = model.module

                        pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
                        plotdir = str(self.results_folder / f'CTClip_{steps}' )
                        plotdir = plotdir + "/"

                        Path(plotdir).mkdir(parents=True, exist_ok=True)

                        predictedlabels=[]
                        for pathology in pathologies:
                            # 这里的pathology是一个字符串，表示病理类型
                            text = [f"There is {pathology}.", f"There is no {pathology}."]
                            # 构建正负prompt
                            text_tokens=self.tokenizer(
                                            text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                            output = model(text_tokens, valid_data,  device=device)

                            output = apply_softmax(output)

                            print(output)
                            append_out=output.detach().cpu().numpy()
                            print(output)
                            if output[0]>output[1]:
                                predictedlabels.append(append_out[0])
                            else:
                                predictedlabels.append(append_out[0])
                        predictedall.append(predictedlabels)
                        realall.append(onehotlabels.detach().cpu().numpy()[0])
                        # Print and save classification report
                    realall=np.array(realall)
                    predictedall=np.array(predictedall)

                    dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)
                    realall = np.rint(realall).astype(int)  
                    predictedall = np.rint(predictedall).astype(int)


                    print('Test F1 Accuracy: ', f1_score(realall, predictedall,average='micro'))
                    print('Test Flat Accuracy: ', accuracy_score(realall.flatten(), predictedall.flatten()),'\n')
 
                    writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

                    dfs.to_excel(writer, sheet_name='Sheet1', index=False)

                    writer.close()
                    del output


        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'CTClip.{steps}.pt')
            state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)

            self.accelerator.save(state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')


        self.steps += 1
        return logs



    def train(self, log_fn=noop):
        device = next(self.CTClip.parameters()).device
        device=torch.device('cuda')
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

