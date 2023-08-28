from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
from PIL import Image
import cv2
from collections import OrderedDict
import torchvision
from . import flow_viz
from torch.optim import lr_scheduler
import torch.nn.init as init
from torch.utils.data.sampler import WeightedRandomSampler



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array

def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[0]
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path)
    # image_pil.save(image_path.replace('.jpg', '.png'))

def save_torch_img(img, save_path):
    image_numpy = tensor2im(img,tile=False)
    save_image(image_numpy, save_path, create_dir=True)
    return image_numpy

def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            if len(images_np.shape) == 4 and images_np.shape[0] == 1:
                images_np = images_np[0]
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)



# def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
#     if isinstance(image_tensor, list):
#         image_numpy = []
#         for i in range(len(image_tensor)):
#             image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
#         return image_numpy

#     if isinstance(image_tensor, torch.autograd.Variable):
#         image_tensor = image_tensor.data
#     if len(image_tensor.size()) == 5:
#         image_tensor = image_tensor[0, -1]
#     if len(image_tensor.size()) == 4:
#         image_tensor = image_tensor[0]
#     image_tensor = image_tensor[:3]
#     image_numpy = image_tensor.cpu().float().numpy()
#     if normalize:
#         image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#     else:
#         image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
#     #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0        
#     image_numpy = np.clip(image_numpy, 0, 255)
#     if image_numpy.shape[2] == 1:        
#         image_numpy = image_numpy[:,:,0]
#     return image_numpy.astype(imtype)

def write_image(image_outputs, display_image_num=2, image_directory=None, postfix=None):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]
    #image_outputs = [images for images in image_outputs]
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = torchvision.utils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    torchvision.utils.save_image(image_grid, fp='%s/gen_%s.jpg'%(image_directory, postfix), nrow=1)


def get_sampler(img_path, flist):
    images = sorted(os.listdir(img_path))
    large_pose_list = read_flist(flist)
    label_list = [1 if image in large_pose_list else 0 for image in images]
    class_counts = np.bincount(label_list)
    print('Class Counts is:', class_counts)
    class_weights = 1. / class_counts
    weights = class_weights[label_list]
    return WeightedRandomSampler(weights, len(weights))

def read_flist(flist):
    f = open(flist, 'r')
    lines = sorted(f.readlines())
    list = []
    print('Loading labels from', flist)
    for line in lines:
        list.append(line.strip())
    return list

def seq2batch(input, batch_size, heat_num=7, multi=False, heatmap=False):
    if heatmap:
        if multi:
            inter = heat_num
        else:
            inter = 1
    else:
        inter = 3
    #print('internal is:', inter)
    tensor_list =  []
    for i in range(batch_size):
        tensor_list.append(input[:, i*inter:(i+1)*inter, :, :])
    return torch.cat(tensor_list, 0)

def pc2batch(input, batch_size, name='face'):
    if name == 'contour':
        inter = 17
    else:
        inter = 51
    tensor_list =  []
    for i in range(batch_size):
        tensor_list.append(input[:, i*inter:(i+1)*inter, :])
    return torch.cat(tensor_list, 0)

def batch2seq(input, batch_size, infer=False):
    tensor_list =  []
    for i in range(batch_size):
        tensor_list.append(input[i:(i+1), :, :, :])
    if not infer:
        return torch.cat(tensor_list, 1)
    else:
        return tensor_list

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".ckpt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

def dict_unite(pretrained_state_dict, model_state_dict):
    for key in pretrained_state_dict:
        if 'module.' in key:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]
    return model_state_dict


def tensor2flow(flo, imtype=np.uint8):
    flo = flo[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_viz.flow_to_image(flo)
    return flo


def add_dummy_to_tensor(tensors, add_size=0):
    if add_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        dummy = torch.zeros_like(tensors)[:add_size]
        tensors = torch.cat([dummy, tensors])
    return tensors

def remove_dummy_from_tensor(tensors, remove_size=0):
    if remove_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        tensors = tensors[remove_size:]
    return tensors

# def save_image(image_numpy, image_path):
#     image_pil = Image.fromarray(image_numpy)
#     image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert(tuple):
    return (int(tuple[0]), int(tuple[1])) 

def draw_heatmap_from_68_landmark(lmk, heatmap=None, width=512, height=512, draw_mouth=True):
    if heatmap is None:                                                                                              
        heatmap = np.zeros((width, height), dtype=np.uint8)

    def draw_line(list):                                                                                                                              
        for i in range(len(list)-1):                                                                                                                   
            #print(lmk[list[i]])                                                                                                                        
            cv2.line(heatmap, convert(lmk[list[i]]), convert(lmk[list[i+1]]), thickness=2, color=255)                                                                                                                                                       
    def draw_circle(list):                                                                                                                                                                                                                                    
        for i in range(len(list)):
            if i != len(list)-1:
                cv2.line(heatmap, convert(lmk[list[i]]), convert(lmk[list[i+1]]), thickness=2, color=255)
            else:
                cv2.line(heatmap, convert(lmk[list[i]]), convert(lmk[list[0]]), thickness=2, color=255)
    if draw_mouth:
        '''draw mouth outter'''                                                                                                                      
        mo_list = list(range(48, 60))                                                                             
        draw_circle(mo_list)

        '''draw mouth inner'''                                                                                                                       
        mi_list = list(range(60, 68))                                                                          
        draw_circle(mi_list)                                                                                                                                 

    '''draw left eye'''                                                                                                                      
    le_list = list(range(36, 42))                                                                             
    draw_circle(le_list)

    '''draw right eye'''                                                                                                                      
    re_list = list(range(42, 48))                                                                             
    draw_circle(re_list)

    '''draw left eye brow'''                                                                                                                      
    leb_list = list(range(17,22))                                                                             
    draw_line(leb_list)

    '''draw right eye brow'''                                                                                                                      
    reb_list = list(range(22,27))                                                                             
    draw_line(reb_list)

    '''draw nose'''                                                                                                                      
    ns_list1 = list(range(27, 31))                                                                            
    draw_line(ns_list1)
    ns_list2 = list(range(31, 36))                                                                            
    draw_line(ns_list2)

    '''draw jaw'''                                                                                                                      
    jaw_list = list(range(0,17))                                                                             
    draw_line(jaw_list)
                                                                                                                                      
    #heatmap = cv2.GaussianBlur(heatmap, ksize=(5, 5), sigmaX=1, sigmaY=1)
    #print('heatmap size:', heatmap.shape)                                                                              
    return heatmap 

def draw_depth(pc, cam, width=512, height=512, color='w'):

    def get_color(c, depth):
        scale_c = (int(c[0]*depth), int(c[1]*depth), int(c[2]*depth))
        return scale_c

    heatmap = np.zeros((width, height), dtype=np.uint8)

    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    elif color == 'w':
        c = (255, 255, 255)

    depth = depth_norm(pc, cam)
    for i in range(pc.shape[0]):
        st = pc[i, :2]
        heatmap = cv2.circle(heatmap,(int(st[0]), int(st[1])), 3, get_color(c, depth[i]), 3)
    print(heatmap.shape)
    return heatmap

def depth_norm(pc, cam):
    scale = cam[0]
    z = pc[:, -1] * (-1) / scale
    z_max = np.max(z)
    z_min = np.min(z)
    z = (z - z_min) / (z_max - z_min) + 0.1
    return z

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def draw_heatmap_from_86_landmark(lmk, heatmap=None, width=512, height=512, draw_mouth=True):
    if heatmap is None:                                                                                              
        heatmap = np.zeros((width, height), dtype=np.uint8)

    def draw_line(list):                                                                                                                              
        for i in range(len(list)-1):                                                                                                                   
            #print(lmk[list[i]])                                                                                                                        
            cv2.line(heatmap, convert(lmk[list[i]]), convert(lmk[list[i+1]]), thickness=2, color=255)                                                                                                                                                       
    def draw_circle(list):                                                                                                                                                                                                                                    
        for i in range(len(list)):
            if i != len(list)-1:
                cv2.line(heatmap, convert(lmk[list[i]]), convert(lmk[list[i+1]]), thickness=2, color=255)
            else:
                cv2.line(heatmap, convert(lmk[list[i]]), convert(lmk[list[0]]), thickness=2, color=255)
    if draw_mouth:
        '''draw mouth outter'''                                                                                                                      
        mo_list = [66, 72, 67, 68, 69, 73, 70, 74, 85, 71, 84, 75]                                                                             
        draw_circle(mo_list)

        '''draw mouth inner'''                                                                                                                       
        mi_list = [76, 78, 79, 80, 77, 83, 82, 81]                                                                          
        draw_circle(mi_list)                                                                                                                                 

    '''draw left eye'''                                                                                                                      
    le_list = [35, 36, 41, 37, 38, 39, 42, 40]                                                                             
    draw_circle(le_list)

    '''draw right eye'''                                                                                                                      
    re_list = [43, 44, 49, 45, 46, 47, 50, 48]                                                                             
    draw_circle(re_list)

    '''draw left eye brow'''                                                                                                                      
    leb_list = [17, 18,19, 20, 21, 25, 24, 23, 22]                                                                             
    draw_circle(leb_list)

    '''draw right eye brow'''                                                                                                                      
    reb_list = [26, 27, 28, 29, 30, 34, 33, 32, 31]                                                                             
    draw_circle(reb_list)

    '''draw nose'''                                                                                                                      
    ns_list1 = [51, 52, 53, 54]                                                                           
    draw_line(ns_list1)
    ns_list2 = [57, 59, 61, 62, 63, 64, 65, 60, 58]                                                                            
    draw_line(ns_list2)

    '''draw jaw'''                                                                                                                      
    jaw_list = list(range(0,17))                                                                             
    draw_line(jaw_list)
                                                                                                                                      
    #heatmap = cv2.GaussianBlur(heatmap, ksize=(5, 5), sigmaX=1, sigmaY=1)
    #print('heatmap size:', heatmap.shape)                                                                              
    return heatmap