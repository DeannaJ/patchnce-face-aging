import torch
import torch.distributed as dist
import torch.distributed.launch
# dist.init_process_group(backend='nccl', init_method='env://')
# torch.cuda.set_device(dist.get_rank())
from aging import model_dict
from aging.basic_template import TrainTask
import os
# torch.autograd.set_detect_anomaly(True) #检测模型中inplace报错的具体位置
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # self, For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
'''
export PYTHONPATH=$PYTHONPATH:/home/zzhuang/Codes
ssl export CUDA_VISIBLE_DEVICES=4,5
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4' #self


'''
python -m torch.distributed.launch --nproc_per_node=1 --master_port=7679 main.py --batch_size 32 \
    --index_file /home/duna/code/Archivenew/ContraAging/ContraAging/dataset/ffhq.txt --data_root /mnt/pami23/duna/dataset/FFHQ/images256x256 \
    --max_epochs 10 --save_freq 2000 --model_name ContraAging --age_group 4 --img_size 128 \
    --gan_loss_weight 75 --patch_nce_loss_weight 5. --pix_loss_weight 5. --init_lr 0.0002 --num_workers 32 --temperature 0.1 --num_patch 128

IPCGAN:2022.06.12 test
python -m torch.distributed.launch --nproc_per_node=4 --master_port=6999 main.py --batch_size 32 \
    --index_file /home/duna/code/Archivenew/ContraAging/ContraAging/dataset/ffhq.txt --data_root /mnt/pami23/duna/dataset/FFHQ/images256x256 \
    --max_epochs 10 --save_freq 2000 --model_name IPCGAN --age_group 4 --img_size 128 \
    --gan_loss_weight 75 --age_loss_weight 5. --pretrained_age_classifier 4_alexnet_morph_112_age_classifier.pth --init_lr 0.0002 --num_workers 32 --run_name ac612
'''

if __name__ == '__main__':
    # reference https://stackoverflow.com/questions/38050873/can-two-python-argparse-objects-be-combined/38053253
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args()
    MODEL = model_dict[default_opt.model_name]
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    print(opt)

    dist.init_process_group(backend='nccl', init_method='env://')#为了让每个进程的模型初始化完全相同，通常这N个进程都是由单个进程复制而来，                                                  
    torch.cuda.set_device(dist.get_rank()) #这时需要对分布式的进程初始化，建立相互通信的机制。Pytorch中使用distributed.init_process_group函数完成初始化
                                        #之后每个进程用唯一的编号rank进行区分，0-(N-1递增，一般将rank=0的进程作为主进程，其他rank的进程作为子进程，
    model = MODEL(opt)      #每个进程还要知道 world_size ，即分布式训练的总进程数 N。训练时，每个进程使用batch的一部分，互相不能重复，通过 nn.utils.data.DistributedSampler 来实现
    model.fit()
    # occupy_memory(batch_size=64)
