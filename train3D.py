import os

from torch.nn.functional import threshold
fold_num = 0
gpu_bias = 2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = f'2, 5, 6'
# os.environ['CUDA_VISIBLE_DEVICES'] = f'{fold_num*2+gpu_bias}, {fold_num*2+1+gpu_bias}'

import time
import json
import argparse
from numpy import Inf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.CT_pancreas_ids import PanCTDataset, EvaPanCTDataset, CachePanDataset
from loss.criterions import get_criterions
# from utils.utils import train_on_epoch, eval_on_epoch, save_model
# from utils.utils_3D_2 import train_on_epoch, eval_on_epoch, save_model
from utils.utils_3D_embed_full import train_on_epoch, eval_on_epoch, save_model, get_weight
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from tqdm import tqdm

def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_data', type=str,
                        default='data/Cirrhosis_T1_3D',
                        help='direction for the dataset')
    parser.add_argument('--is_transform', type=bool,
                        default=True, help='apply transform or not')
    parser.add_argument('--split_ratio', type=float,
                        default=0.9, help='split ratio for training')
    parser.add_argument('--is_pretrained', type=bool,
                        default=True, help='pretained or not')
    parser.add_argument('--pretrained_dir', type=str,
                        default='./out/log/20220125-12_2', help='pretrained dir')
    parser.add_argument('--model_name', type=str,
                        default='MaskTransUnet', help='model name for training')
    parser.add_argument('--batch_size', type=int,
                        default=3, help='patient batch size')
    parser.add_argument('--depth_size', type=int,
                        default=32, help='patient depth size')
    parser.add_argument('--num_samples', type=int,
                        default=6, help='num samples')

    # num layers [32, 32, 64, 64, 128]
    # previous [16, 32, 64, 128, 256]
    # [32, 64, 64, 128, 256]
    parser.add_argument('--num_layers', type=list, 
                        default=[16, 32, 64, 128, 256], help='number of layer for each layer')
    # 320-160-80-40-20: 160-80-40-20-10
    # 256-128-64-32-16:  80-40-20-10-5
    parser.add_argument('--roi_size_list', type=list, 
                        default=[100, 65, 40, 25, 10], help='size of roi for each layer')
    parser.add_argument('--is_roi_list', type=list, 
                        default=[False, True, True, True, True], help='using roi for each layer')

    '''
    parser.add_argument('--num_layers', type=list, 
                        default=[16, 32, 32, 64], help='number of layer for each layer')
    '''
    parser.add_argument('--dim_input', type=int,
                        default=1, help='input dimension or modality')
    parser.add_argument('--dim_output', type=int,
                        default=2, help='output dimension or classes')
    parser.add_argument('--kernel_size', type=int,
                        default=3, help='kernel_size for convolution')

    parser.add_argument('--device', type=str,
                        default='cuda', help='device for training')
    parser.add_argument('--epochs', type=int,
                        default=800, help='epochs for training')

    parser.add_argument('--eval_epoch', type=int,
                        default=5, help='the interval epoch for eval')
    parser.add_argument('--log_dir', type=str,
                        default='./runs/log', help='device for training')
    parser.add_argument('--model_dir', type=str,
                        default='./out/log', help='device for training')
    parser.add_argument('--criterion_list', type=list,
                        default=['CrossEntroLoss', 'DiceClassLoss'],
                        help='device for training')
    parser.add_argument('--criterion_weight', type=list,
                        default=[1, 1],
                        help='device for training')
    parser.add_argument('--weight_list', type=list,
                        default=[0.05, 0.05, 0.1, 0.1, 1.0],
                        help='weight list for training')
    parser.add_argument('--final_weight', type=list,
                        default=[2., 1.5, 1.0, 1., 1.0],
                        help='weight list for training')
    parser.add_argument('--initial_weight', type=list,
                        default=[0.1, 0.2, 0.3, 0.4, 1.0],
                        help='device for training')

    args = parser.parse_args()
    return args

def get_model(args, fold_num, device):
    model_fn = get_model_dict(args.model_name)
    model = model_fn(num_layers=args.num_layers,
                     roi_size_list=args.roi_size_list,
                     is_roi_list=args.is_roi_list,
                     dim_input=args.dim_input,
                     dim_output=args.dim_output,
                     kernel_size=args.kernel_size)

    if args.is_pretrained:
        pretrained_dir = os.path.join(args.pretrained_dir, f'fold_{fold_num}', 'temp_model.pt')
        # state_dict = torch.load(pretrained_dir).state_dict()
        # model.load_state_dict(state_dict)
        model.load_state_dict(torch.load(pretrained_dir))

    model = nn.DataParallel(model.to(device))
    return model

def get_dynamic_weight(args, T, warmup_step):
    weight_list = args.weight_list
    initial_weight = args.initial_weight
    final_weight = args.final_weight
    out_list = []

    for i in range(len(weight_list)):
        y = [get_weight(j-warmup_step, T=T, 
                        default_weight=weight_list[i], 
                        initial_weight=initial_weight[i],
                        final_weight=final_weight[i])
                        for j in range(args.epochs)]
        out_list.append(y)

    out_list = list(zip(*out_list))
    return out_list

def get_criterion_list(args):
    criterions = []
    criterion_name = args.criterion_list
    temp_list = ['CrossEntroLoss', 'BalanceDiceLoss']
    temp_list2 = ['CrossEntroLoss', 'DiceClassLoss']
    eval_list = ['BalanceDiceLoss', 'DiceClassLoss', 'RecallLoss', 'PrecisionLoss','LocalizationLoss']

    for i in range(len(args.num_layers)):
        if i < (len(args.num_layers)-2):
            criterions.append(get_criterions(temp_list))
        elif i == (len(args.num_layers)-2):
            criterions.append(get_criterions(temp_list2))
        else:
            criterions.append(get_criterions(criterion_name))
    
    eval_criterions = get_criterions(eval_list)
    return criterions, eval_criterions

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(  # noqa: B038
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best,
                                                                                                    dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


def train_fn(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0.
    for i, data in enumerate(loader):
        x = data[0].to(device)
        y = data[1].to(device)

        optimizer.zero_grad()
        y_hat, _ = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def main(args):
    # torch.autograd.set_detect_anomaly(True)
    num_device = torch.cuda.device_count()

    root = args.dir_data
    is_transform = args.is_transform
    depth_size = args.depth_size
    num_samples = args.num_samples
    batch_size = args.batch_size
    step_times = num_samples // 3

    train_pandataset = CachePanDataset(root=root,
                                       depth_size=depth_size,
                                       num_samples=num_samples,
                                       ids=[0,1,2,3])
    
    test_pandataset = EvaPanCTDataset(root=root,
                                      depth_size=depth_size,
                                      ids=[1,2,3])

    train_panDl = DataLoader(dataset=train_pandataset, batch_size=batch_size,
                             num_workers=12, shuffle=True, pin_memory=True)
    test_panDl = DataLoader(dataset=test_pandataset, batch_size=1,
                            shuffle=False, pin_memory=True)

    img = train_pandataset[0]
    label = train_pandataset[0]
    img = img[0]["image"]
    label = label[0]["label"]
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :, 10].detach().cpu(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, 10].detach().cpu())
    plt.show()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # model = get_model(args, fold_num, device=device)

    model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(512, 512, 32),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    loss_function = DiceCELoss()
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(20):

        loss = train_fn(model, optimizer, loss_function, train_panDl, device)

        print(f"Epoch {epoch + 1} loss: ", loss)
        if epoch % 10 == 0:
            checkpoint = {
                "model":model.state_dict()
            }

    #
    # warmup_step = 10
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                                       mode='min',
    #                                                       factor=0.8,
    #                                                       patience=5,
    #                                                       threshold=1e-2,
    #                                                       cooldown=1,
    #                                                       min_lr=1e-7)
    # '''
    # sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250,
    #                                            gamma=0.5)
    #
    # sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                       T_max=100,
    #                                                       eta_min=1e-6)
    # '''
    # epochs = args.epochs
    # patient_batchsize = batch_size
    # patient_epochs = num_samples
    #
    # criterions, eval_criterions = get_criterion_list(args)
    # # print(criterions)
    # criterion_weight = args.criterion_weight
    # # writer = SummaryWriter(os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M")))
    # writer = SummaryWriter(os.path.join(args.log_dir, time.strftime("%Y%m%d-%H_2"), f'fold_{fold_num}'))
    # # writer = SummaryWriter(os.path.join(args.log_dir, '20211109-1112'))
    # # model_dir = os.path.join(args.model_dir, time.strftime("%Y%m%d-%H%M"))
    # model_dir = os.path.join(args.model_dir, time.strftime("%Y%m%d-%H_2"), f'fold_{fold_num}')
    #
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    #
    # global_step = 0
    # smooth_ratio = 0
    # best_eval_loss = Inf
    # eval_loss = 50
    # best_train_loss = Inf
    # train_loss = Inf
    # smooth_eval_loss = Inf
    # smooth_train_loss = Inf
    # T = 12
    #
    # dynamic_weight_list = get_dynamic_weight(args, T, warmup_step)
    #
    # for i in tqdm(range(epochs)):
    #     dynamic_weight =  dynamic_weight_list[i]
    #     # dynamic_weight =  args.weight_list
    #     if i % args.eval_epoch == 0:
    #         eval_loss, global_step = eval_on_epoch(model=model,
    #                                                dataloader=test_panDl,
    #                                                criterions=eval_criterions,
    #                                                device=device,
    #                                                writer=writer,
    #                                                patient_epochs=patient_epochs,
    #                                                patient_batchsize=patient_batchsize,
    #                                                global_step=global_step)
    #
    #         sheduler.step(eval_loss)
    #
    #         if i != 0:
    #             smooth_eval_loss = eval_loss
    #             smooth_train_loss = train_loss
    #         else:
    #             smooth_eval_loss = (1-smooth_ratio)*eval_loss + \
    #                                     smooth_ratio*smooth_eval_loss
    #             smooth_train_loss = (1-smooth_ratio)*train_loss + \
    #                                     smooth_ratio*smooth_train_loss
    #
    #         if smooth_eval_loss <= best_eval_loss:
    #             best_eval_loss = smooth_eval_loss
    #             best_train_loss = smooth_train_loss
    #             print('Best train_loss:', best_train_loss)
    #             print('Best eval loss', eval_loss)
    #
    #             save_model(model.module.state_dict(),
    #                     os.path.join(model_dir, f'temp_model.pt'))
    #
    #     if i < warmup_step:
    #         dynamic_weight =  dynamic_weight_list[0]
    #
    #     train_loss, global_step = train_on_epoch(model=model,
    #                                              dataloader=train_panDl,
    #                                              optimizer=optimizer,
    #                                              criterions=criterions,
    #                                              step_times=step_times,
    #                                              device=device,
    #                                              writer=writer,
    #                                              patient_epochs=patient_epochs,
    #                                              patient_batchsize=patient_batchsize,
    #                                              global_step=global_step,
    #                                              dynamic_weight=dynamic_weight)
    #
    #     # sheduler.step()
    #
    # print('Best train_loss:', best_train_loss)
    # print('Best eval loss', eval_loss)
    # writer.close()
    # save_model(model.module, os.path.join(model_dir, 'model.pt'))



if __name__ == '__main__':
    args = get_parse()
    main(args)
