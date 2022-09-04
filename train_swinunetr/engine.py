import os

import pandas as pd
import torch
from monai import data
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm

from config import config
from model import create_model
from transforms import train_transform, val_transform
from utils import create_datalist, set_seed


def train(
    model,
    optimizer,
    loss_function,
    global_step,
    train_loader,
    val_loader,
    dice_val_best,
    global_step_best,
    scaler,
    device,
    max_iterations,
    eval_num,
    dice_metric,
    post_label,
    post_pred,
    epoch_loss_values,
    metric_values,
    ):
    
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["petct"].to(device), batch["mask"].to(device))
        
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        
        #loss.backward()
        scaler.scale(loss).backward()  # <=============================

        epoch_loss += loss.item()

        #optimizer.step()
        scaler.step(optimizer)  # <=============================

        scaler.update()  # <=============================

        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, max_iterations, loss)
        )
        if ((
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations) and (global_step >= 70_000):
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )

            dice_val = validation(
                model=model,
                device=device,
                dice_metric=dice_metric,
                post_label=post_label,
                post_pred=post_pred,
                global_step=global_step,
                epoch_iterator_val=epoch_iterator_val
                )

            epoch_loss /= step

            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                        model.state_dict(), os.path.join('path_to_models_dir', "best_metric_model_fold4_final.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best

def validation(
    model,
    device,
    dice_metric,
    post_label,
    post_pred,
    global_step,
    epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["petct"].to(device), batch["mask"].to(device))
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 16, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate at step (%d)" % (global_step)
            )
        mean_dice_val = dice_metric.aggregate().item()

        dice_metric.reset()
    return mean_dice_val

def run_fold(df, fold, config):

    if config['include_healthy']:
        df_train = df[(df['kfold'] != fold)]
        df_valid = df[(df['kfold'] == fold)]
    else:
        df_train = df[(df['kfold'] != fold) & (df['diagnosis'] != 'NEGATIVE')]
        df_valid = df[(df['kfold'] == fold) & (df['diagnosis'] != 'NEGATIVE')]

    if config['debug']:
        df_train = df_train[:8]
        df_valid = df_valid[:1]

    train_files = create_datalist(df=df_train, col='study_location')
    validation_files = create_datalist(df=df_valid, col='study_location')

    print(f'Length datalist_train: {len(train_files)}, length datalist_valid: {len(validation_files)}')


    if config['cache_dataset']:
        train_ds = data.CacheDataset(
            data=train_files,
            transform=train_transform,
            cache_rate=1.0,
            num_workers=16
        )
        
        val_ds = data.CacheDataset(
            data=validation_files,
            transform=val_transform,
            cache_rate=1.0,
            num_workers=16
            )
    elif config['persistent_dataset']:
        train_ds = data.PersistentDataset(
            data=train_files,
            transform=train_transform,
            cache_dir=config['persistent_dir']
        )
        
        val_ds = data.PersistentDataset(
            data=validation_files,
            transform=val_transform,
            cache_dir=config['persistent_dir']
            )
    else:
        train_ds = data.Dataset(
            data=train_files,
            transform=train_transform
        )

        val_ds = data.Dataset(
            data=validation_files,
            transform=val_transform
        )
    
    train_loader = data.DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = create_model(config=config)
    model.to(config['device'])

    torch.backends.cudnn.benchmark = True

    criterion = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr= config['lr'],
        weight_decay=config['wd'],
    )

    scaler = torch.cuda.amp.GradScaler()

    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    
    dice_metric = DiceMetric(include_background=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    
    epoch_loss_values = []
    metric_values = []

    while global_step < config['max_iterations']:
        global_step, dice_val_best, global_step_best = train(
            model=model,
            optimizer=optimizer,
            loss_function=criterion,
            global_step=global_step,
            train_loader=train_loader,
            val_loader=val_loader,
            dice_val_best=dice_val_best,
            global_step_best=global_step_best,
            scaler=scaler,
            device=config['device'],
            max_iterations=config['max_iterations'],
            eval_num=config['eval_num'],
            dice_metric=dice_metric,
            post_label=post_label,
            post_pred=post_pred,
            epoch_loss_values=epoch_loss_values,
            metric_values=metric_values
            )

    return dice_val_best, global_step_best
        
if __name__=='__main__':
    config = config

    df = pd.read_csv(config['folds_path'])
    fold = config['fold']
    
    set_seed()
    dice_val_best, global_step_best = run_fold(
        df=df,
        fold=config['fold'],
        config=config
    )

    print(f'Best Dice Score of {dice_val_best} at global step {global_step_best}.')