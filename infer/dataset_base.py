import random
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from decord import VideoReader
import pandas as pd
from typing import Union
from copy import deepcopy
import os


class DatasetBase(Dataset):
    """
    功能说明：
    数据集加载器, 每个视频加载中间帧
    适配分布式训练or推理
    适配当前所有视频数据信息
    """
    """
    参数说明：
    csv_path: 视频索引信息csv路径
    sample_size: 抽帧resize分辨率
    num_gpus: gpu数量
    local_rank: 节点gpu序号，用于多卡推理分割数据集
    csv_result_path: 已标注信息表，用于断点重新续
    s3_local_dir_replace: 路径清洗参数，path.replace(s3_dir, local_dir)
    video_col_indx: 视频索引信息列，包括视频唯一id，存储路径
    """
    def __init__(
            self,
            csv_path: Union[pd.DataFrame, str], 
            sample_size: Union[int, tuple] = 256, 
            num_gpus=1,
            local_rank=0,
            csv_result_path: Union[pd.DataFrame, str] = None,
            s3_local_dir_replace: tuple = None, # (s3_dir, local_dir)
            video_col_indx: tuple = None        # ('videoid', 'link') -- prime key and read_path
        ):
        assert local_rank < num_gpus, f'local_rank {local_rank} is over gpu number {num_gpus}, pleace check and adjuest args'
        super(DatasetBase, self).__init__()
        self.vid, self.vpath = self._setup_dataframe(csv_path, csv_result_path, s3_local_dir_replace, video_col_indx)
        # 分布式截段
        self.length = int(len(self.vid) / num_gpus)
        self.vid = self.vid[local_rank * self.length: (local_rank+1) * self.length]
        self.vpath = self.vpath[local_rank * self.length: (local_rank+1) * self.length]

        self.pixel_transforms = self._setup_transforms(sample_size)  # 初始化处理器
        
        
    def _setup_dataframe(self, csv_path, csv_result_path, s3_local_dir_replace, video_col_indx):
        """
        初始化csv数据集索引数据; 过滤已处理的数据条
        返回视频key 视频read_path
        """
        def _key_trans(k):
            try:
                return str(int(k))
            except:
                return str(k)
        def _read_csv(path):
            # 读取csv数据 TODO 更新encoder
            try:
                df = pd.read_csv(path)
            except:
                df = pd.read_excel(path)
            return df
    
        if video_col_indx:
            vid, vpath = video_col_indx
        else:
            vid, vpath = 'videoid', 'link' # 默认列索引
        
        # 数据加载
        vid_done = []
        if csv_result_path:
            if isinstance(csv_result_path, pd.DataFrame):
                df_done = deepcopy(csv_result_path)
            else:
                if os.path.exists(csv_result_path): 
                    df_done = _read_csv(csv_result_path)
                else:
                    df_done = pd.DataFrame()
            if not df_done.empty:
                vid_done = df_done[vid].apply(_key_trans)

        if isinstance(csv_path, str):
            df_info = _read_csv(csv_path)
        else:
            df_info = deepcopy(csv_path)
            del csv_path
        
        
        df_info = df_info[[vid, vpath]]
        df_info = df_info[~df_info[vid].apply(_key_trans).isin(vid_done)] # 过滤已完成视频信息
        
        if s3_local_dir_replace:
            df_info.loc[:, vpath] = df_info[vpath].apply(lambda x: x.replace(s3_local_dir_replace[0],
                                                                             s3_local_dir_replace[1]))
        return df_info[vid].to_list(), df_info[vpath].to_list()
    
    def _setup_transforms(self, sample_size):
        """
        初始化图像数据预处理模块
        数据预处理：随机翻转 resize  中心裁剪 通道标准化
        """
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.Resize(sample_size[0]),
                                   transforms.CenterCrop(sample_size),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),]) 
    
    def _get_video_process(self, idx):
        """
        根据索引抽取视频信息；
        返回视频图像数据、视频基本信息等
        """
        vid = self.vid[idx]
        vpath = self.vpath[idx]
        video_reader = VideoReader(vpath)
        video_length = len(video_reader)
        """
        按照sample_n_frames进行等距抽样 default 3: 取第一 中间 结束帧
        """
        # 取中间帧
        batch_index = int((video_length - 1) / 2)
        pixel_array = torch.from_numpy(video_reader.get_batch([batch_index]).asnumpy()[0, ]).permute(2, 0, 1).contiguous() # BGR2RGB
        # tensorsize --> [3, .., ..]
        # TODO 预留取若干帧
        # batch_index = list(set([int(v) for v in np.linspace(0, video_length-1, self.sample_n_frames)]))
        # pixel_array = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() # BGR2RGB

        del video_reader     
        return pixel_array, vid, vpath

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_array, vid, vpath = self._get_video_process(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length-1)
        sample = {'videoid': vid,
                  'pixel_array': pixel_array,
                  'link': vpath
                  }
        return sample


def collate_fn(batch):
    # padding batch for np.stack in Dataloader
    # return {key: list}
    from torchvision.transforms import CenterCrop
    pixel_array = [x['pixel_array'] for x in batch]
    max_size = [0, 0]
    for pv in pixel_array:
        if pv.shape[1]>max_size[0]:
            max_size[0] = pv.shape[1]
        if pv.shape[2]>max_size[1]:
            max_size[1] = pv.shape[2]
    transform = CenterCrop(max_size)
    padded_pixel_values = torch.stack([transform(p) for p in pixel_array])

    return {'pixel_array': padded_pixel_values,
            'videoid': [x['videoid'] for x in batch],
            'link': [x['link'] for x in batch],
            }

def get_dataloader(batch_size, num_workers, **args):
    # 获取dataloader
    # args dataset实例化参数
    return torch.utils.data.DataLoader(DatasetBase(**args), 
                                       batch_size=batch_size, 
                                       num_workers=num_workers,
                                       collate_fn=collate_fn)

if __name__ == "__main__":
    # artlist 数据测试
    csv_path='/highres_new/high_res/project/data_process/0_code_caption/artlist.csv'
    video_col_indx = ('videoid', 'link')
    s3_local_dir_replace = ('s3://morph-dataself-tag/download_artlist_1080p/artlist_20',
                            '/highres_new/high_res/project/data_process/0_code_caption/0_data')
    csv_result_path = None
    dataloader = get_dataloader(batch_size=4,
                                num_workers=6,

                                csv_path=csv_path, 
                                csv_result_path=csv_result_path,
                                video_col_indx=video_col_indx,
                                s3_local_dir_replace=s3_local_dir_replace)

    for idx, batch in enumerate(dataloader):
        print(len(batch['pixel_array']))
        print(batch['pixel_array'].shape)