import os

import torch as t
from torch.utils.data import DataLoader

from models import Yolov1_vgg16bn

from config import BestConfig
from datasets import AerialImages
from utils import AverageMeter, Visualizer, NMS

import fire

device = t.device("cuda")

def evaluate():
    
    from hw2_evaluation_task import readdet, voc_eval
    
    detpath = os.path.join('predictTxt/best/', '{:s}.txt')
    annopath = os.path.join('hw2_train_val/val1500/labelTxt_hbb/', '{:s}.txt')
    imagenames = [x.split('.')[0] for x in os.listdir('hw2_train_val/val1500/labelTxt_hbb/') if x.endswith('.txt')]

    '''
    # read list of images
    imagesetfile = sys.argv[3]
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    '''

    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    ##############################################
    #classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #            'storage-tank',  'soccer-ball-field', 'harbor', 'swimming-pool']
    ##############################################

    det = readdet(detpath, imagenames, classnames)
    del_list = []
    for key in det:
        #print('%s: %d' % (key, len(det[key])))
        if len(det[key]) == 0:
            del_list.append(key)
    #classnames = [x for x in classnames if x not in del_list]

    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(det[classname],
             annopath,
             imagenames,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        ## uncomment to plot p-r curve for each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
        # plt.show()
    map = map/len(classnames)
    print('map:', map)
    #classaps = 100*np.array(classaps)
    #print('classaps: ', classaps)
    
    return map  

def yololoss(predicts, labels, lambda_coord=5, lambda_noobj=0.5):
    
    loss = 0
    
    predicts_clone = predicts.detach().clone()
    labels_clone = labels.clone()[:, 0:4, :, :]
    
    obj_mask = labels[:, 4:5, :, :] > 0.1
    obj_mask = obj_mask.float().expand_as(labels)
    
    
    weights = t.zeros(16).to(device)
    
    n = 0
    n_classes = 0
    
    for i in range(16):
        
        count = (((t.max(labels[:, 10:, :, :], dim=1, keepdim=True)[1].int()+(labels[:, 4:5, :, :] < 0.1).int()*-16) == i)).float().sum().item()
        n = n + count
        
        if count > 0:
            
            weights[i] = 1 / count
            n_classes = n_classes + 1
    
    if n > 0:
    
        mul = t.norm(weights, p=1).item() * n / n_classes
        weights = weights / t.norm(weights, p=1).item()
        weights = weights * mul
        
    weights = weights[t.max(labels[:, 10:, :, :], dim=1, keepdim=True)[1]]
    weights = weights.expand_as(labels)
    
    
    loss += lambda_noobj * (((predicts*(1-obj_mask))[:, 4:10:5, :, :]) ** 2).sum()
    
    
    loss += ((((predicts*obj_mask)[:, 10:, :, :] - (labels*obj_mask)[:, 10:, :, :])**2)*weights[:, 10:, :, :]).sum()
    
    
    predicts_clone[:, 0:2, :, :] = predicts_clone[:, 0:2, :, :] / 7.0
    predicts_clone[:, 5:7, :, :] = predicts_clone[:, 5:7, :, :] / 7.0
    
    predicts_clone[:, 0, :, :] = predicts_clone[:, 0, :, :] - predicts_clone[:, 2, :, :]/2.0
    predicts_clone[:, 1, :, :] = predicts_clone[:, 1, :, :] - predicts_clone[:, 3, :, :]/2.0
    predicts_clone[:, 2, :, :] = predicts_clone[:, 2, :, :] + predicts_clone[:, 0, :, :]
    predicts_clone[:, 3, :, :] = predicts_clone[:, 3, :, :] + predicts_clone[:, 1, :, :]
    
    predicts_clone[:, 5, :, :] = predicts_clone[:, 5, :, :] - predicts_clone[:, 7, :, :]/2.0
    predicts_clone[:, 6, :, :] = predicts_clone[:, 6, :, :] - predicts_clone[:, 8, :, :]/2.0
    predicts_clone[:, 7, :, :] = predicts_clone[:, 7, :, :] + predicts_clone[:, 5, :, :]
    predicts_clone[:, 8, :, :] = predicts_clone[:, 8, :, :] + predicts_clone[:, 6, :, :]

    labels_clone[:, 0:2, :, :] = labels_clone[:, 0:2, :, :] / 7.0
    
    labels_clone[:, 0, :, :] = labels_clone[:, 0, :, :] - labels_clone[:, 2, :, :]/2.0
    labels_clone[:, 1, :, :] = labels_clone[:, 1, :, :] - labels_clone[:, 3, :, :]/2.0
    labels_clone[:, 2, :, :] = labels_clone[:, 2, :, :] + labels_clone[:, 0, :, :]
    labels_clone[:, 3, :, :] = labels_clone[:, 3, :, :] + labels_clone[:, 1, :, :]
    
    inter1_xmin = t.max(predicts_clone[:, 0:1, :, :], labels_clone[:, 0:1, :, :])
    inter1_ymin = t.max(predicts_clone[:, 1:2, :, :], labels_clone[:, 1:2, :, :])
    inter1_xmax = t.min(predicts_clone[:, 2:3, :, :], labels_clone[:, 2:3, :, :])
    inter1_ymax = t.min(predicts_clone[:, 3:4, :, :], labels_clone[:, 3:4, :, :])
    
    inter1 = t.max(inter1_xmax-inter1_xmin, t.tensor([0.0]).to(device)) * t.max(inter1_ymax-inter1_ymin, t.tensor([0.0]).to(device))
    union1 = (predicts_clone[:, 2:3, :, :]-predicts_clone[:, 0:1, :, :]) * (predicts_clone[:, 3:4, :, :]-predicts_clone[:, 1:2, :, :])
    union1 += (labels_clone[:, 2:3, :, :]-labels_clone[:, 0:1, :, :]) * (labels_clone[:, 3:4, :, :]-labels_clone[:, 1:2, :, :])
    union1 -= inter1
    
    iou1 = inter1 / (union1+0.000001)
    
    inter2_xmin = t.max(predicts_clone[:, 5:6, :, :], labels_clone[:, 0:1, :, :])
    inter2_ymin = t.max(predicts_clone[:, 6:7, :, :], labels_clone[:, 1:2, :, :])
    inter2_xmax = t.min(predicts_clone[:, 7:8, :, :], labels_clone[:, 2:3, :, :])
    inter2_ymax = t.min(predicts_clone[:, 8:9, :, :], labels_clone[:, 3:4, :, :])
    
    inter2 = t.max(inter2_xmax-inter2_xmin, t.tensor([0.0]).to(device)) * t.max(inter2_ymax-inter2_ymin, t.tensor([0.0]).to(device))
    union2 = (predicts_clone[:, 7:8, :, :]-predicts_clone[:, 5:6, :, :]) * (predicts_clone[:, 8:9, :, :]-predicts_clone[:, 6:7, :, :])
    union2 += (labels_clone[:, 2:3, :, :]-labels_clone[:, 0:1, :, :]) * (labels_clone[:, 3:4, :, :]-labels_clone[:, 1:2, :, :])
    union2 -= inter2
    
    iou2 = inter2 / (union2+0.000001)
    
    iou = t.cat((iou1, iou2), dim=1)
    max_iou, indices = t.max(iou, dim=1, keepdim=True)
    
    bbx0_mask = (indices.float()<0.1).float().expand_as(labels)
    
    
    loss += lambda_coord * ((((predicts*obj_mask*bbx0_mask)[:, 0:2, :, :] - (labels*obj_mask*bbx0_mask)[:, 0:2, :, :]) ** 2)*weights[:, 0:2, :, :]).sum()
    loss += lambda_coord * (((((predicts*obj_mask*bbx0_mask)[:, 2:4, :, :]+0.000001)**0.5 - ((labels*obj_mask*bbx0_mask)[:, 2:4, :, :]+0.000001)**0.5) ** 2)*weights[:, 2:4, :, :]).sum()
    
    loss += lambda_coord * ((((predicts*obj_mask*(1-bbx0_mask))[:, 5:7, :, :] - (labels*obj_mask*(1-bbx0_mask))[:, 0:2, :, :]) ** 2)*weights[:, 0:2, :, :]).sum()
    loss += lambda_coord * (((((predicts*obj_mask*(1-bbx0_mask))[:, 7:9, :, :]+0.000001)**0.5 - ((labels*obj_mask*(1-bbx0_mask))[:, 2:4, :, :]+0.000001)**0.5) ** 2)*weights[:, 2:4, :, :]).sum()
    
    
    loss += ((((predicts*obj_mask*bbx0_mask)[:, 4:5, :, :] - max_iou*obj_mask[:, 0:1, :, :]*bbx0_mask[:, 0:1, :, :]) ** 2)*weights[:, 0:1, :, :]).sum()
    loss += lambda_noobj * ((((predicts*obj_mask*bbx0_mask)[:, 9, :, :]) ** 2)*weights[:, 0, :, :]).sum()

    loss += ((((predicts*obj_mask*(1-bbx0_mask))[:, 9:10, :, :] - max_iou*obj_mask[:, 0:1, :, :]*(1-bbx0_mask[:, 0:1, :, :])) ** 2)*weights[:, 0:1, :, :]).sum()
    loss += lambda_noobj * ((((predicts*obj_mask*(1-bbx0_mask))[:, 4, :, :]) ** 2)*weights[:, 0, :, :]).sum()

    loss = loss / predicts.size()[0]
    
    return loss, (max_iou.sum()/obj_mask[:, 0, :, :].sum()).item()

def train(**kwargs):
    
    opt = BestConfig()
    opt.parse(kwargs)
    
    train_root = 'hw2_train_val/train15000/'
    valid_root = 'hw2_train_val/val1500/'
    
    train_dataset = AerialImages(train_root)
    valid_dataset = AerialImages(valid_root)
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, opt.batch_size, shuffle=False)

    yolov1 = Yolov1_vgg16bn(pretrained=True).to(device)
    
    optimizer = t.optim.SGD(yolov1.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    
    if not os.path.isdir(opt.ckpts_root):
        os.makedirs(opt.ckpts_root)
        
    if not os.path.isdir(opt.predicts_root):
        os.makedirs(opt.predicts_root)
        
    vis = Visualizer(opt.env, opt.port)
    vis.add_names('loss', 'iou', 'conf', 'map')
    
    for epoch in range(opt.n_epoch):
        
        if epoch == 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr*0.1

        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        conf_meter = AverageMeter()
        
        yolov1.train()
             
        for i, (I, labels, img_names, (H, W)) in enumerate(train_dataloader):
            
            I = I.to(device)
            labels = labels.to(device)
            
            predicts = yolov1(I)         
            loss, iou = yololoss(predicts, labels)
            
            print(i, loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), I.size()[0])
            iou_meter.update(iou, I.size()[0])
            conf_meter.update(t.max(predicts[:, 4:5, :, :]*t.max(predicts[:, 10:, :, :], dim=1, keepdim=True)[0]).item(), I.size()[0])
            
        t.save(yolov1.state_dict(), os.path.join(opt.ckpts_root, 'e{}.ckpt').format(epoch))
        
        
        yolov1.eval()
        
        for I, labels, img_names, (H, W) in valid_dataloader:
            
            I = I.to(device)
            
            predicts = yolov1(I)
            predicts = predicts.detach()
            
            for n in range(predicts.size()[0]):
                
                final_indices, final_classes, final_scores = NMS(predicts[n], H[n], W[n], opt.thres, valid_dataset.classes)
                
                with open(os.path.join(opt.predicts_root, img_names[n].split('.')[0]+'.txt'), 'w') as predicttxt:
                    
                    for index, class_, score in zip(final_indices, final_classes, final_scores):
                        
                        x = index[0]
                        y = index[1]
                        w = index[2]
                        h = index[3]
                        
                        
                        predicttxt.write(str(x-w/2) + ' ')
                        predicttxt.write(str(y-h/2) + ' ')
                        
                        predicttxt.write(str(x+w/2) + ' ')
                        predicttxt.write(str(y-h/2) + ' ')
                        
                        predicttxt.write(str(x+w/2) + ' ')
                        predicttxt.write(str(y+h/2) + ' ')
                        
                        predicttxt.write(str(x-w/2) + ' ')
                        predicttxt.write(str(y+h/2) + ' ')
                        
                        predicttxt.write(class_ + ' ')
                        predicttxt.write(str(score) + '\n')
        
        
        vis.plot('loss', epoch, loss_meter.avg)
        vis.plot('iou', epoch, iou_meter.avg)
        vis.plot('conf', epoch, conf_meter.avg)
        vis.plot('map', epoch, evaluate())
        
    return

if __name__ == '__main__':
    
    fire.Fire()