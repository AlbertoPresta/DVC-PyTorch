import os
import argparse
import torch
import cv2
import logging
import numpy as np
#from net_dvc import *
from models.DVC.net_dvc import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import sys
import math
import json
from dataset import DataSet,  HEVCDataSet
from tensorboardX import SummaryWriter
#from drawuvg import uvgdrawplt
torch.backends.cudnn.enabled = True
# gpu_num = 4
os.environ['CUDA_VISIBLE_DEVICES']= '1'
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 2048
print_step = 1000
cal_step = 10
# print_step = 10
warmup_step = 0#  // gpu_num
gpu_per_batch = 4
test_step = 10000#  // gpu_num
tot_epoch = 1000000
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ref_i_dir = geti(train_lambda)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



parser = argparse.ArgumentParser(description='DVC reimplement')

parser.add_argument('-l', '--log', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--testhevc', action='store_true')
parser.add_argument('--testvtl', action='store_true')
parser.add_argument('--testmcl', action='store_true')
parser.add_argument('--testauc', action='store_true')
parser.add_argument('--rerank', action='store_true')
parser.add_argument('--allpick', action='store_true')
parser.add_argument('--config', dest='config', default="DVC_1024.json",
        help = 'hyperparameter of Reid in json format')

import wandb
def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, test_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, ref_i_dir
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'test_step' in config:
        test_step = config['test_step']
        print('teststep : ', test_step)
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        ref_i_dir = geti(train_lambda)
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']

def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    global warmup_step
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:#  // gpu_num:
        lr = base_lr
    else:
        lr = base_lr * (lr_decay ** (global_step // decay_interval))
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Var(x):
    return Variable(x.cuda())

def testhevc(global_step, testfull=False):
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print("testing : %d/%d"% (batch_idx, len(test_loader)))
            input_images = input[0]
            ref_image = input[1]
            ref_bpp = input[2]
            ref_psnr = input[3]
            ref_msssim = input[4]
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach().numpy()
            sumpsnr += torch.mean(ref_psnr).detach().numpy()
            summsssim += torch.mean(ref_msssim).detach().numpy()
            cnt += 1
            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]
                inputframe, refframe = Var(input_image), Var(ref_image)
                strings_and_shape = net.compress(inputframe, refframe)
                strings, shape = strings_and_shape["strings"], strings_and_shape["shape"]
                reconframe = net.decompress(refframe, strings, shape)["x_hat"]

                num_pixels = input_image.size()[2] * input_image.size()[3]
                num_pixels = torch.tensor(num_pixels).float()
                bpp = (len(strings[0][0])+ len(strings[1][0]) + len(strings[2][0]) + len(strings[3][0])) * 8.0 / num_pixels
                #bpp = torch.sum(len(s[0]) for s in strings) * 8.0 / num_pixels
                #print(bpp)
                mse_loss = torch.mean((reconframe - inputframe).pow(2))
                #print(torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy())
                sumbpp += torch.mean(bpp).cpu().detach().numpy()
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()
                summsssim += ms_ssim(reconframe.cpu().detach(), input_image, data_range=1.0, size_average=True).numpy()
                cnt += 1
                ref_image = reconframe
        log = "global step %d : " % (global_step) + "\n"
        logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "HEVCdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)
        #uvgdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=testfull)

def train(epoch, global_step):

    print ("epoch", epoch)
    global gpu_per_batch
    train_loader = DataLoader(dataset = train_dataset, shuffle=True, num_workers=gpu_num, batch_size=gpu_per_batch, pin_memory=True)
    net.train()


    total_loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss_av = AverageMeter()
    psnr_loss = AverageMeter()
    inter_psnr_loss = AverageMeter()
    warp_psnr_loss = AverageMeter()
    bpp_feature_loss = AverageMeter()
    bpp_mv_loss = AverageMeter()
    bpp_z_loss = AverageMeter()
    dist_loss = AverageMeter()



    global optimizer
    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    sumpsnr = 0
    suminterpsnr = 0
    sumwarppsnr = 0
    sumbpp = 0
    sumbpp_feature = 0
    sumbpp_mv = 0
    sumbpp_z = 0
    tot_iter = len(train_loader)
    t0 = datetime.datetime.now()
    for batch_idx, input in enumerate(train_loader):
        global_step += 1
        bat_cnt += 1
        input_image, ref_image = Var(input[0]), Var(input[1])
        # ta = datetime.datetime.now()
        clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(input_image, ref_image)
        
        # tb = datetime.datetime.now()
        mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = \
                                        torch.mean(mse_loss), torch.mean(warploss), torch.mean(interloss),\
                                        torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp_mv),\
                                         torch.mean(bpp)
        
        distribution_loss = bpp
        if global_step < 500000: #orignal: global_step<500000
            warp_weight = 0.1
        else:
            warp_weight = 0
        distortion = mse_loss + warp_weight * (warploss + interloss)
        rd_loss = train_lambda * distortion + distribution_loss
        # tc = datetime.datetime.now()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        rd_loss.backward()


        # tf = datetime.datetime.now()
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 0.5)
        optimizer.step()

        aux_loss = net.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if batch_idx % 1000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{batch_idx*len(input)}/{len(train_loader)}"
                f" ({100. * batch_idx / len(train_loader):.1f}%)]"
                f'\tLoss: {rd_loss:.3f} |'
                f'\tMSE loss: {distortion:.3f} |'
                f'\tBpp loss: {distribution_loss:.2f} |'
               f"\tAux loss: {0.000:.2f}"
            )

        if global_step % 1 == 0:
            cal_cnt += 1
            if mse_loss > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy()
            else:
                psnr = 100
            if warploss > 0:
                warppsnr = 10 * (torch.log(1 * 1 / warploss) / np.log(10)).cpu().detach().numpy()
            else:
                warppsnr = 100
            if interloss > 0:
                interpsnr = 10 * (torch.log(1 * 1 / interloss) / np.log(10)).cpu().detach().numpy()
            else:
                interpsnr = 100

            loss_ = rd_loss.cpu().detach().numpy()

            sumloss += loss_
            sumpsnr += psnr
            suminterpsnr += interpsnr
            sumwarppsnr += warppsnr
            sumbpp += bpp.cpu().detach()
            sumbpp_feature += bpp_feature.cpu().detach()
            sumbpp_mv += bpp_mv.cpu().detach()
            sumbpp_z += bpp_z.cpu().detach()

            total_loss.update(loss_)
            mse_loss_av.update(mse_loss.cpu().detach())
            dist_loss.update(distortion.cpu().detach())
            bpp_loss.update(bpp.cpu().detach())
            bpp_feature_loss.update(bpp_feature.cpu().detach())
            bpp_mv_loss.update(bpp_mv.cpu().detach())
            bpp_z_loss.update(bpp_z.cpu().detach())
            inter_psnr_loss.update(interpsnr)
            warp_psnr_loss.update(warppsnr)
            wand_dict = {
                    "train_batch": global_step,
                    "train_batch/loss": rd_loss.cpu().detach().item(),
                    "train_batch/bpp_total": bpp.cpu().detach().item(),
                    "train_batch/bpp_feature":bpp_z.cpu().detach().item(),
                    "train_batch/bpp_z":bpp_z.cpu().detach().item(),
                    "train_batch/bpp_mv":bpp_mv.cpu().detach().item(),
                    "train_batch/mse":mse_loss.cpu().detach().item(),
                    "train_batch/distorsion":distortion.cpu().detach().item(),
                    "train_batch/inter_psnr":interpsnr,
                    "train_batch/warp_psnr":warppsnr
                }
            wandb.log(wand_dict)
    net.update()

    wand_dict = {
            "train": epoch,
            "train/loss": total_loss.avg,
            "train/bpp_total": bpp_loss.avg,
            "train/bpp_feature":bpp_feature_loss.avg,
            "train/bpp_z":bpp_z_loss.avg,
            "train/bpp_mv":bpp_mv_loss.avg,
            "train/mse":mse_loss_av.avg,
            "train/distorsion":dist_loss.avg,
            "train/warp_psnr_loss":warp_psnr_loss.avg, 
            "train/inter_psnr_loss":inter_psnr_loss.avg
        }
    wandb.log(wand_dict)
    return  global_step




def test(epoch, global_step,mode = "valid"):

    print ("epoch", epoch)
    global gpu_per_batch
    valid_loader = DataLoader(dataset = valid_dataset if mode == "valid" else test_dataset, 
                    shuffle=False,
                     num_workers=gpu_num,
                      batch_size=gpu_per_batch if mode == "valid" else 1,
                       pin_memory=True)
    net.eval()


    total_loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss_av = AverageMeter()
    psnr_loss = AverageMeter()
    inter_psnr_loss = AverageMeter()
    warp_psnr_loss = AverageMeter()
    bpp_feature_loss = AverageMeter()
    bpp_mv_loss = AverageMeter()
    bpp_z_loss = AverageMeter()
    dist_loss = AverageMeter()


    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    sumpsnr = 0
    suminterpsnr = 0
    sumwarppsnr = 0
    sumbpp = 0
    sumbpp_feature = 0
    sumbpp_mv = 0
    sumbpp_z = 0
    tot_iter = len(valid_loader)
    with torch.no_grad():
        for batch_idx, input in enumerate(valid_loader):
            global_step += 1
            bat_cnt += 1
            input_image, ref_image = Var(input[0]), Var(input[1])
            # ta = datetime.datetime.now()
            clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(input_image, ref_image)
            
            # tb = datetime.datetime.now()
            mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = \
                                            torch.mean(mse_loss), torch.mean(warploss), torch.mean(interloss),\
                                            torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp_mv),\
                                            torch.mean(bpp)
            
            distribution_loss = bpp
            warp_weight = 0
            distortion = mse_loss + warp_weight * (warploss + interloss)
            rd_loss = train_lambda * distortion + distribution_loss

            psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy()
            warppsnr = 10 * (torch.log(1 * 1 / warploss) / np.log(10)).cpu().detach().numpy()
            interpsnr = 10 * (torch.log(1 * 1 / interloss) / np.log(10)).cpu().detach().numpy()


            loss_ = rd_loss.cpu().detach().numpy()

            sumloss += loss_
            sumpsnr += psnr
            suminterpsnr += interpsnr
            sumwarppsnr += warppsnr
            sumbpp += bpp.cpu().detach()
            sumbpp_feature += bpp_feature.cpu().detach()
            sumbpp_mv += bpp_mv.cpu().detach()
            sumbpp_z += bpp_z.cpu().detach()

            total_loss.update(loss_)
            mse_loss_av.update(mse_loss.cpu().detach())
            dist_loss.update(distortion.cpu().detach())
            bpp_loss.update(bpp.cpu().detach())
            bpp_feature_loss.update(bpp_feature.cpu().detach())
            bpp_mv_loss.update(bpp_mv.cpu().detach())
            bpp_z_loss.update(bpp_z.cpu().detach())
            inter_psnr_loss.update(interpsnr)
            warp_psnr_loss.update(warppsnr)

            if mode == "valid":
                wand_dict = {
                        "valid_batch": global_step,
                        "valid_batch/loss": rd_loss.cpu().detach().item(),
                        "valid_batch/bpp_total": bpp.cpu().detach().item(),
                        "valid_batch/bpp_feature":bpp_z.cpu().detach().item(),
                        "valid_batch/bpp_z":bpp_z.cpu().detach().item(),
                        "valid_batch/bpp_mv":bpp_mv.cpu().detach().item(),
                        "valid_batch/mse":mse_loss.cpu().detach().item(),
                        "valid_batch/distorsion":distortion.cpu().detach().item(),
                        "valid_batch/inter_psnr":interpsnr,
                        "valid_batch/warp_psnr":warppsnr
                    }
                wandb.log(wand_dict)
            else: 
                wand_dict = {
                        "test_batch": global_step,
                        "test_batch/loss": rd_loss.cpu().detach().item(),
                        "test_batch/bpp_total": bpp.cpu().detach().item(),
                        "test_batch/bpp_feature":bpp_z.cpu().detach().item(),
                        "test_batch/bpp_z":bpp_z.cpu().detach().item(),
                        "test_batch/bpp_mv":bpp_mv.cpu().detach().item(),
                        "test_batch/mse":mse_loss.cpu().detach().item(),
                        "test_batch/distorsion":distortion.cpu().detach().item(),
                        "test_batch/inter_psnr":interpsnr,
                        "test_batch/warp_psnr":warppsnr
                    }
                wandb.log(wand_dict)            

            
        if mode == "valid":    
            wand_dict = {
                "valid": epoch,
                "valid/loss": total_loss.avg,
                "valid/bpp_total": bpp_loss.avg,
                "valid/bpp_feature":bpp_feature_loss.avg,
                "valid/bpp_z":bpp_z_loss.avg,
                    "valid/bpp_mv":bpp_mv_loss.avg,
                    "valid/mse":mse_loss_av.avg,
                    "valid/distorsion":dist_loss.avg,
                    "valid/warp_psnr_loss":warp_psnr_loss.avg, 
                    "valid/inter_psnr_loss":inter_psnr_loss.avg
                }
        else:
            wand_dict = {
                    "test": epoch,
                    "test/loss": total_loss.avg,
                    "test/bpp_total": bpp_loss.avg,
                    "test/bpp_feature":bpp_feature_loss.avg,
                    "test/bpp_z":bpp_z_loss.avg,
                    "test/bpp_mv":bpp_mv_loss.avg,
                    "test/mse":mse_loss_av.avg,
                    "test/distorsion":dist_loss.avg,
                    "test/warp_psnr_loss":warp_psnr_loss.avg, 
                    "test/inter_psnr_loss":inter_psnr_loss.avg
                }
        
        wandb.log(wand_dict)

    return total_loss.avg,global_step


if __name__ == "__main__":
    args = parser.parse_args()
    wandb.init( config= args, project="classic_dvc", entity="albipresta") 

    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.log != '':
        filehandler = logging.FileHandler(args.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("DVC training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)

    model = VideoCompressor()
    if args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        global_step = load_model(model, args.pretrain)
    net = model.cuda()
    #net = torch.nn.DataParallel(net, list(range(gpu_num)))
    bp_parameters = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
    aux_parameters = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
    optimizer = optim.Adam(bp_parameters, lr=base_lr)
    aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)
    # save_model(model, 0)
    global train_dataset, test_dataset, valid_dataset
    if args.testhevc:
        net.update(force=True)
        test_dataset = HEVCDataSet(testfull=True)
        print('testing HEVC')
        testhevc(0, testfull=True)
        exit(0)
    
    save_name = args.config.split('.')[0]
    if not os.path.isdir('./events/{}'.format(save_name)):
        os.mkdir('./events/{}'.format(save_name))

    tb_logger = SummaryWriter('./events/{}'.format(save_name))
    train_dataset = DataSet("/scratch/dataset/vimeo_septuplet/",mode = "train")
    valid_dataset = DataSet("/scratch/dataset/vimeo_septuplet/",mode = "valid")
    test_dataset = DataSet("/scratch/dataset/vimeo_septuplet/",mode = "test")
    # test_dataset = UVGDataSet(refdir=ref_i_dir)

    stepoch = global_step // (train_dataset.__len__() // (gpu_per_batch))# * gpu_num))
    best_lss = 1000000000
    global_step_val = global_step 
    global_step_test = global_step
    for epoch in range(stepoch, tot_epoch):
        print("************************************ ",epoch)
        adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:  
            save_model(model, best)
            break
        global_step = train(epoch, global_step)

        lss, global_step_val = test(epoch,global_step_val, mode = "valid")
        _,global_step_test = test(epoch,global_step_test, mode = "test")
        if lss < best_lss:
            best = True 
            best_lss = lss 
        else:
            best = False


        save_model(model, best)
    
    



""""
        if mode == "valid":
            wand_dict = {
                    "valid_batch": global_step,
                    "valid_batch/loss": rd_loss.cpu().detach().item(),
                    "valid_batch/bpp_total": bpp.cpu().detach().item(),
                    "valid_batch/bpp_feature":bpp_z.cpu().detach().item(),
                    "valid_batch/bpp_z":bpp_z.cpu().detach().item(),
                    "valid_batch/bpp_mv":bpp_mv.cpu().detach().item(),
                    "valid_batch/mse":mse_loss.cpu().detach().item(),
                    "valid_batch/distorsion":distortion.cpu().detach().item(),
                    "valid_batch/inter_psnr":interpsnr,
                    "valid_batch/warp_psnr":warppsnr
                }
            wandb.log(wand_dict)
        else: 
            wand_dict = {
                    "test_batch": global_step,
                    "test_batch/loss": rd_loss.cpu().detach().item(),
                    "test_batch/bpp_total": bpp.cpu().detach().item(),
                    "test_batch/bpp_feature":bpp_z.cpu().detach().item(),
                    "test_batch/bpp_z":bpp_z.cpu().detach().item(),
                    "test_batch/bpp_mv":bpp_mv.cpu().detach().item(),
                    "test_batch/mse":mse_loss.cpu().detach().item(),
                    "test_batch/distorsion":distortion.cpu().detach().item(),
                    "test_batch/inter_psnr":interpsnr,
                    "test_batch/warp_psnr":warppsnr
                }
            wandb.log(wand_dict)            

        
    if mode == "valid":    
        wand_dict = {
            "valid": epoch,
            "valid/loss": total_loss.avg,
            "valid/bpp_total": bpp_loss.avg,
            "valid/bpp_feature":bpp_feature_loss.avg,
            "valid/bpp_z":bpp_z_loss.avg,
                "valid/bpp_mv":bpp_mv_loss.avg,
                "valid/mse":mse_loss_av.avg,
                "valid/distorsion":dist_loss.avg,
                "valid/warp_psnr_loss":warp_psnr_loss.avg, 
                "valid/inter_psnr_loss":inter_psnr_loss.avg
            }
    else:
        wand_dict = {
                "test": epoch,
                "test/loss": total_loss.avg,
                "test/bpp_total": bpp_loss.avg,
                "test/bpp_feature":bpp_feature_loss.avg,
                "test/bpp_z":bpp_z_loss.avg,
                "test/bpp_mv":bpp_mv_loss.avg,
                "test/mse":mse_loss_av.avg,
                "test/distorsion":dist_loss.avg,
                "test/warp_psnr_loss":warp_psnr_loss.avg, 
                "test/inter_psnr_loss":inter_psnr_loss.avg
            }
    
    wandb.log(wand_dict)

"""