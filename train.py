from dataset import MVTecADDataset, CATEGORIES, OBJECTS, TEXTILES, Resize, RandomCrop, RandomTranslation, \
    RandomRotation, ToTensor
from models.originalAE import OriginalAE
from models.largeAE import LargeAE

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import pytorch_ssim
import pytorch_msssim
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, choices=['original', 'large'], default='large')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--latent_space_dim', type=int, default=2048)
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'ssim', 'mssim'], default='l2')
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=1600)

    parser.add_argument('--dataset_directory', type=str, default='D:/mvtec_anomaly_detection')
    parser.add_argument('--category', type=str, choices=CATEGORIES, default='bottle')
    parser.add_argument('--color_mode', type=str, choices=['rgb', 'grayscale'], default='rgb')
    parser.add_argument('--augmentation', type=bool, default=True)

    parser.add_argument('--tensorboard_directory', type=str, default='runs')
    parser.add_argument('--load_model_directory', type=str)
    parser.add_argument('--save_model_directory', type=str, default='D:/saved_AE_models')

    return parser.parse_args()


args = parse_args()

if args.color_mode == 'rgb' and (args.category == 'grid' or args.category == 'screw' or args.category == 'zipper'):
    print("Only grayscale images are supported for {}. Continuing with grayscale images".format(args.category))

if args.category in TEXTILES:
    if args.augmentation:
        transform_list = [Resize(256), ToTensor()]
    else:
        transform_list = [Resize(256), ToTensor()]
elif args.category in OBJECTS:
    if args.augmentation:
        transform_list = [RandomTranslation(40), RandomRotation(30), Resize(256), ToTensor()]
    else:
        transform_list = [Resize(256), ToTensor()]

dataset_directory = args.dataset_directory
train_dataset = MVTecADDataset(dataset_path=dataset_directory, mode="train", category=args.category,
                               color_mode=args.color_mode, transform=transforms.Compose(transform_list))
validate_dataset = MVTecADDataset(dataset_path=dataset_directory, mode="validate", category=args.category,
                                  color_mode=args.color_mode, transform=transforms.Compose(transform_list))

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                 pin_memory=True)

# build initial model
if args.model == 'original':
    model = OriginalAE(args.color_mode, args.latent_space_dim).to(device)
elif args.model == 'large':
    model = LargeAE(args.color_mode, args.latent_space_dim).to(device)

# select type of loss
if args.loss == 'l1':
    criterion = nn.L1Loss().to(device)
elif args.loss == 'l2':
    criterion = nn.MSELoss().to(device)
elif args.loss == 'ssim':
    criterion = pytorch_ssim.SSIM(window_size=11).to(device)
elif args.loss == 'mssim':
    criterion = pytorch_msssim.MSSSIM(window_size=11).to(device)

# initialize optimizer and writer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
writer = SummaryWriter(os.path.join(args.tensorboard_directory,
                                    "{}_{}_{}_b{}_lr-{}".format(args.model, args.latent_space_dim, args.loss,
                                                                args.batch_size,
                                                                args.learning_rate)))

load_model_path = args.load_model_directory
save_model_path = args.save_model_directory

# load trained model to continue training
last_epoch = 0
if load_model_path is not None:
    load = torch.load(load_model_path)
    model.load_state_dict(load['model'])
    optimizer.load_state_dict(load['optimizer'])
    last_epoch = load['epoch']
    loss = load['loss']

os.makedirs(os.path.join(save_model_path,
                         "{}_{}_{}_b{}_lr-{}".format(args.model, args.latent_space_dim, args.loss, args.batch_size,
                                                     args.learning_rate)), exist_ok=True)
for epoch in range(args.num_epochs):
    # train
    train_loss = 0.0
    model.train()
    for _, train_data in enumerate(train_dataloader):
        x = train_data[0].to(device)
        x = x.float().to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss = train_loss / len(train_dataloader)

    # validate
    validate_loss = 0.0
    model.eval()
    with torch.no_grad():
        for _, validate_data in enumerate(validate_dataloader):
            x = validate_data[0].to(device)
            x = x.float().to(device)
            output = model(x)
            loss = criterion(output, x)
            validate_loss += loss.item() * x.size(0)
        validate_loss = validate_loss / len(validate_dataloader)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(last_epoch + epoch + 1, train_loss,
                                                                               validate_loss))

    # tensorboard
    writer.add_scalar('training loss', train_loss, last_epoch + epoch)
    writer.add_scalar('validation loss', validate_loss, last_epoch + epoch)

    # save model
    if save_model_path is not None and (last_epoch + epoch + 1) % 200 == 0:
        torch.save({
            'epoch': last_epoch + epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss
        }, os.path.join(save_model_path,
                        "{}_{}_{}_b{}_lr-{}".format(args.model, args.latent_space_dim, args.loss, args.batch_size,
                                                    args.learning_rate), "epoch_{}.pt".format(last_epoch + epoch + 1)))

        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)

    if not torch.isfinite(loss) or train_loss > 500: break
