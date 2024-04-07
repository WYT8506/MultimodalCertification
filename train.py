import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.util import confusion_matrix, getScores, tensor2labelim, tensor2im, print_current_losses
import numpy as np
import random
import torch
import cv2
from tensorboardX import SummaryWriter
from torch.utils.data import random_split

if __name__ == '__main__':
    opt = TrainOptions().parse()

    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    #train_data_loader = CreateDataLoader(train_opt)
    all_dataset = create_dataset(opt)#train_data_loader.load_data()
    all_dataset_size = len(all_dataset)
    print('#all images = %d' % all_dataset_size)
    train_length = int(0.8 * all_dataset_size)  # 80% for training
    val_length = all_dataset_size - train_length  # 20% for validation
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(all_dataset, [train_length, val_length])
    print(len(train_dataset),len(val_dataset))
    
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=not opt.serial_batches,
    num_workers=0,
    worker_init_fn=lambda worker_id: numpy.random.seed(opt.seed + worker_id))
    

    val_dataset.phase = "test"
    val_dataset_size = len(val_dataset)
    print('#validation images = %d' % val_dataset_size)
    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=not opt.serial_batches,
    num_workers=0,
    worker_init_fn=lambda worker_id: numpy.random.seed(opt.seed + worker_id))

    writer = SummaryWriter()

    model = create_model(opt, train_dataset.dataset)
    model.setup(opt)
    total_steps = 0
    tfcount = 0
    F_score_max = 0
    for epoch in range(opt.epoch_count, opt.nepoch + 1):
        ### Training on the training set ###
        model.train()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        train_loss_iter = []
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                tfcount = tfcount + 1
                losses = model.get_current_losses()
                train_loss_iter.append(losses["segmentation"])
                t = (time.time() - iter_start_time) / opt.batch_size
                print_current_losses(epoch, epoch_iter, losses, t, t_data)
                # There are several whole_loss values shown in tensorboard in one epoch,
                # to help better see the optimization phase
                writer.add_scalar('train/whole_loss', losses["segmentation"], tfcount)

            iter_data_time = time.time()

        mean_loss = np.mean(train_loss_iter)
        # One average training loss value in tensorboard in one epoch
        writer.add_scalar('train/mean_loss', mean_loss, epoch)

        palet_file = 'datasets/palette.txt'
        impalette = list(np.genfromtxt(palet_file,dtype=np.uint8).reshape(3*256))
        tempDict = model.get_current_visuals()
        rgb = tensor2im(tempDict['rgb_image'])
        if opt.use_sne:
            another = tensor2im((tempDict['another_image']+1)/2)    # color normal images
        else:
            another = tensor2im(tempDict['another_image'])
        label = tensor2labelim(tempDict['label'], impalette)
        output = tensor2labelim(tempDict['output'], impalette)
        image_numpy = np.concatenate((rgb, another, label, output), axis=1)
        image_numpy = image_numpy.astype(np.float64) / 255
        writer.add_image('Epoch' + str(epoch), image_numpy, dataformats='HWC')  # show training images in tensorboard

        print('End of epoch %d / %d \t Time Taken: %d sec' %   (epoch, opt.nepoch, time.time() - epoch_start_time))
        model.update_learning_rate()

        ### Evaluation on the validation set ###
        model.eval()
        valid_loss_iter = []
        epoch_iter = 0
        conf_mat = np.zeros((val_dataset.dataset.num_labels, val_dataset.dataset.num_labels), dtype=np.float64)
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                model.set_input(data)
                model.forward()
                model.get_loss()
                epoch_iter += 1
                gt = model.label.cpu().int().numpy()
                _, pred = torch.max(model.output.data.cpu(), 1)
                pred = pred.float().detach().int().numpy()

                # Resize images to the original size for evaluation
                image_size = model.get_image_oriSize()
                oriSize = (image_size[0].item(), image_size[1].item())
                gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
                pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)

                conf_mat += confusion_matrix(gt, pred, val_dataset.dataset.num_labels)
                losses = model.get_current_losses()
                valid_loss_iter.append(model.loss_segmentation)
                print('valid epoch {0:}, iters: {1:}/{2:} '.format(epoch, epoch_iter, len(val_dataset)), end='\r')

        avg_valid_loss = torch.mean(torch.stack(valid_loss_iter))
        globalacc, pre, recall, F_score, iou = getScores(conf_mat)

        # Record performance on the validation set
        writer.add_scalar('valid/loss', avg_valid_loss, epoch)
        writer.add_scalar('valid/global_acc', globalacc, epoch)
        writer.add_scalar('valid/pre', pre, epoch)
        writer.add_scalar('valid/recall', recall, epoch)
        writer.add_scalar('valid/F_score', F_score, epoch)
        writer.add_scalar('valid/iou', iou, epoch)
        print('valid/loss', avg_valid_loss, epoch)
        print('valid/global_acc', globalacc, epoch)
        print('valid/pre', pre, epoch)
        print('valid/recall', recall, epoch)
        print('valid/F_score', F_score, epoch)
        print('valid/iou', iou, epoch)
        # Save the last model
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save_networks(opt.certification_method)
        F_score_max = F_score
        writer.add_text('last model', str(epoch))
