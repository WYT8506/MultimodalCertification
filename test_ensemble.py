
from data import create_dataset
from models import create_model
import numpy as np
import random
import torch
import cv2
from tensorboardX import SummaryWriter
from torch.utils.data import random_split
from options.test_options import TestOptions
from PIL import Image
from tqdm import tqdm
if __name__ == '__main__':

    test_opt = TestOptions().parse()
    np.random.seed(test_opt.seed)
    random.seed(test_opt.seed)
    torch.manual_seed(test_opt.seed)
    torch.cuda.manual_seed(test_opt.seed)
    
    all_dataset = create_dataset(test_opt)
    all_dataset_size = len(all_dataset)
    print('#all images = %d' % all_dataset_size)
    train_length = int(0.8 * all_dataset_size)  # 80% for training
    test_length = all_dataset_size - train_length  # 20% for testing
    #fix seed such that train and test division is unchanged
    torch.manual_seed(42)
    _, test_dataset = random_split(all_dataset, [train_length, test_length])
    print("Length of the test dataset is:",len(test_dataset))
    test_dataset.phase = "test"
    
    test_opt.isTrain = False

    test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    worker_init_fn=lambda worker_id: numpy.random.seed(test_opt.seed + worker_id))
    writer = SummaryWriter()

    model = create_model(test_opt, test_dataset.dataset)
    model.load_networks(test_opt)
    total_steps = 0
    tfcount = 0
    F_score_max = 0
 
    model.eval()
    epoch_iter = 0
    
    print("num_labels:", test_dataset.dataset.num_labels)
    all_pred =[]
    all_gt =[]
    for i in tqdm(range(test_opt.N)): #subsample each testing input N times, default value of N is 100
        preds =[]
        gts = []
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                model.set_input(data)
                model.forward()
                model.get_loss()
                epoch_iter += test_opt.batch_size
                gt = model.label.cpu().int().numpy()
                _, pred = torch.max(model.output.data.cpu(), 1)
                pred = pred.float().detach().int().numpy()
                
                oriSize = [1242,375]
                gt = torch.tensor(np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)).int()
                pred = torch.tensor(np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)).int()
                preds.append(torch.tensor(pred[0]).int())
                gts.append(torch.tensor(gt[0]).int())

        preds = torch.stack(preds)
        gts = torch.stack(gts)
        all_pred.append(preds)
        if i == 0:
            all_gt.append(gts)
    all_pred = torch.stack(all_pred)
    all_gt = torch.stack(all_gt)
    dict_ = {"all_pred":all_pred, "all_gt":all_gt}
    if test_opt.certification_method == "randomized_ablation":
        torch.save(dict_, 'output/'+test_opt.certification_method+"_ablation-ratio-test="+str(test_opt.ablation_ratio_test)+'_all_outputs.pth')
    else:
        torch.save(dict_, 'output/'+test_opt.certification_method+"_ablation-ratio-test1="+str(test_opt.ablation_ratio_test1)+"_ablation-ratio-test2="+str(test_opt.ablation_ratio_test2)+'_all_outputs.pth')