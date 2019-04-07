import numpy as np

import torch as t

import visdom

class AverageMeter():

    def __init__(self):
        
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
        return

    def reset(self):
        
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
        return

    def update(self, val, n=1):
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        return

class Visualizer():
    
    def __init__(self, env='YOLOv1', port=8888):
        
        self.vis = visdom.Visdom(env=env, port=port)
        self.names = set()
        self.log_text = ''
        
        return
    
    def add_names(self, *args):
        
        for name in args:
            self.names.add(name)
            
        return
    
    def plot(self, name, epoch, value):
        
        if not name in self.names:
            print('Unknown name for plotting!')
            print('Use add_names to add a new name.')
            
        else:
            opts = {'xlabel':'epoch', 'ylabel':name} 
            self.vis.line(Y=np.array([value]), X=np.array([epoch]), win=name,
                          opts=opts, update=None if epoch == 0 else 'append')
            
        return
    
    def imgs(self, name, I):
        
        self.vis.images(I, nrow=4, win=name)
        
        return
        
    def log(self, info, win='log'):
        
        self.log_text += '{} <br>'.format(info)
        self.vis.text(self.log_text, win)
        
        return
    
def NMS(predict, H, W, thres, class_list):
    
    indices = []
    classes = []
    scores = []
    
    for i in range(7):
        for j in range(7):
            
            x = predict[0, i, j].item()
            y = predict[1, i, j].item()
            w = predict[2, i, j].item()
            h = predict[3, i, j].item()
            c = predict[4, i, j].item() * t.max(predict[10:, i, j]).item()
            class_ = class_list[(t.max(predict[10:, i, j], dim=0)[1]).item()]
            
            x = x*64.0 + j*64.0
            y = y*64.0 + i*64.0
            w = w * 447.0
            h = h * 447.0
            
            x = x * (W.item()-1.0) / 447.0
            y = y * (H.item()-1.0) / 447.0
            w = w * (W.item()-1.0) / 447.0
            h = h * (H.item()-1.0) / 447.0
            
            indices.append([x, y, w, h])
            classes.append(class_)
            scores.append(c)
            
            
            x = predict[5, i, j].item()
            y = predict[6, i, j].item()
            w = predict[7, i, j].item()
            h = predict[8, i, j].item()
            c = predict[9, i, j].item() * t.max(predict[10:, i, j]).item()
            
            x = x*64.0 + j*64.0
            y = y*64.0 + i*64.0
            w = w * 447.0
            h = h * 447.0
            
            x = x * (W.item()-1.0) / 447.0
            y = y * (H.item()-1.0) / 447.0
            w = w * (W.item()-1.0) / 447.0
            h = h * (H.item()-1.0) / 447.0
            
            indices.append([x, y, w, h])
            classes.append(class_)
            scores.append(c)
            
            
    indices = np.array(indices)
    classes = np.array(classes)
    scores = np.array(scores)
    
    indices = indices[np.argsort(scores)[::-1]]
    classes = classes[np.argsort(scores)[::-1]]
    scores = scores[np.argsort(scores)[::-1]]
    
    indices = indices[scores>thres]
    classes = classes[scores>thres]
    scores = scores[scores>thres]
    
    
    final_indices= []
    final_classes = []
    final_scores = []
    
    for index, class_, score in zip(indices, classes, scores):
        
        x11 = index[0] - index[2]/2.0
        y11 = index[1] - index[3]/2.0
        x12 = index[0] + index[2]/2.0
        y12 = index[1] + index[3]/2.0
        
        
        non_max = False
        
        for final_index in final_indices:
            
            x21 = final_index[0] - final_index[2]/2.0
            y21 = final_index[1] - final_index[3]/2.0
            x22 = final_index[0] + final_index[2]/2.0
            y22 = final_index[1] + final_index[3]/2.0
            
            
            x1 = max(x11, x21)
            y1 = max(y11, y21)
            x2 = min(x12, x22)
            y2 = min(y12, y22)
            
            inter = max((x2-x1), 0) * max((y2-y1), 0) 
            union = (x12-x11)*(y12-y11) + (x22-x21)*(y22-y21) - inter
            iou = inter / (union+0.000001)
            
            if iou > 0.5:
                
                non_max = True
                
                break
            
        if not non_max:
            
            final_indices.append(index)
            final_classes.append(class_)
            final_scores.append(score)
            
    return final_indices, final_classes, final_scores