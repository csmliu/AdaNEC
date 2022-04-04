from os.path import join, basename
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util


opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = False   # True on SIR (wild, postcard, solid) dataset for speedup
opt.no_log =True
opt.display_id=0
opt.verbose = False

datadir = './datasets/eval'

# Define evaluation/test dataset

eval_dataset_real = datasets.CEILTestDataset(join(datadir, 'real20'), fns=read_fns(join(datadir, 'real20', 'data_list.txt')))
# eval_dataset_wild = datasets.CEILTestDataset(join(datadir, 'wild'), fns=read_fns(join(datadir, 'wild', 'data_list.txt')))
# eval_dataset_postcard = datasets.CEILTestDataset(join(datadir, 'postcard'), fns=read_fns(join(datadir, 'postcard', 'data_list.txt')))
# eval_dataset_solid = datasets.CEILTestDataset(join(datadir, 'solid'), fns=read_fns(join(datadir, 'solid', 'data_list.txt')))


eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

# eval_dataloader_wild = datasets.DataLoader(
#     eval_dataset_wild, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# eval_dataloader_solid = datasets.DataLoader(
#     eval_dataset_solid, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# eval_dataloader_postcard = datasets.DataLoader(
#     eval_dataset_postcard, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)


engine = Engine(opt)

"""Main Loop"""
result_dir = './results'

all_res = {}
res = engine.eval(eval_dataloader_real, dataset_name='testdata_real', savedir=join(result_dir, 'real20'))
all_res['real20'] = res
print('real20', res)
# res = engine.eval(eval_dataloader_wild, dataset_name='testdata_wild', savedir=join(result_dir, 'wild'))
# all_res['wild'] = res
# print('wild', res)
# res = engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=join(result_dir, 'postcard'))
# all_res['postcard'] = res
# print('postcard', res)
# res = engine.eval(eval_dataloader_solid, dataset_name='testdata_solid', savedir=join(result_dir, 'solid'))
# all_res['solid'] = res
# print('solid', res)


num = {
    'real20': 20,
    'wild': 50,
    'postcard': 199,
    'solid': 200,
}


avg_res = {}
cnt = 0
for d, res in all_res.items():
    for k in res.keys():
        avg_res[k] = avg_res.get(k, 0) + res[k] * num[d]
    cnt += num[d]
for k, v in avg_res.items():
    avg_res[k] = v / cnt
print('avg:', avg_res)
