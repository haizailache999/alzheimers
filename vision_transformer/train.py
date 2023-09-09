import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer
from lion_pytorch import Lion
#from torch.utils.tensorboard import SummaryWriter
#from torchvision import transforms
import os,gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
#device_ids = [4,5,6]
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
#del variable
gc.collect()
torch.cuda.empty_cache()
#from my_dataset import MRIDataset
from DataGenerator import MRIDataGenerator
from vit_model import own_model as create_model
from utils import read_split_data, train_one_epoch, evaluate, test,read_split_data1,test_model
#import torchvision.models as modelss


def main(args):
    #device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("/hy-tmp/weights_20_11") is False:
        os.makedirs("/hy-tmp/weights_20_11")

    #tb_writer = SummaryWriter()

    '''train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    test_images_path,test_images_label=read_split_data1(args.test_data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    data_transform1={"test": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}'''
    # 实例化训练数据集
    '''train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    
    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform1["test"])'''
    train_dataset=MRIDataGenerator('/hy-tmp/ADNI_CAPS',
                                     batchSize=args.batch_size* torch.cuda.device_count(),
                                     idx_fold=0,
                                     split='train')
    val_dataset=MRIDataGenerator('/hy-tmp/ADNI_CAPS',
                                     batchSize=args.batch_size* torch.cuda.device_count(),
                                     idx_fold=0,
                                     split='val')
    test_dataset=MRIDataGenerator('/hy-tmp/ADNI_CAPS',
                                     batchSize=args.batch_size* torch.cuda.device_count(),
                                     idx_fold=0,
                                     split='test')
    print(args.batch_size*torch.cuda.device_count())
    '''train_dataset=MRIDataset('C:/Users/user/uiuc_data/data1/ADNI_CAPS', 'train', None)
    val_dataset=MRIDataset('C:/Users/user/uiuc_data/data1/ADNI_CAPS', 'val', None)
    test_dataset=MRIDataset('C:/Users/user/uiuc_data/data1/ADNI_CAPS', 'test', None)'''
    
    #batch_size = args.batch_size* len(device_ids)
    batch_size=args.batch_size*torch.cuda.device_count()
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 32])  # number of workers
    nw=56
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
                                               #collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
                                             #collate_fn=val_dataset.collate_fn)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
                                             #collate_fn=val_dataset.collate_fn)
    model = create_model(num_classes=args.num_classes, has_logits=False)
    #model=torch.nn.DataParallel(model, device_ids=device_ids)
    #model=torch.nn.DataParallel(model)
    #model=model.to('cuda')
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location='cpu')
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias','pos_embed','norm.weight','norm.bias'] if model.has_logits \
            else ['head.weight', 'head.bias','pos_embed','norm.weight','norm.bias','patch_embed.proj.weight','patch_embed.proj.bias']
            #'pre_logits.fc.weight', 'pre_logits.fc.bias',
        for k in del_keys:
            del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)
        del weights_dict
        gc.collect()
        torch.cuda.empty_cache()
        model=torch.nn.DataParallel(model)
        model=model.to('cuda')
        #print(next(model.parameters()).device)
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.24) #5E-5
    #optimizer = Lion(pg, lr=args.lr)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    total_step_train = math.ceil(len(train_dataset) / batch_size)
    total_step_val=math.ceil(len(val_dataset) / batch_size)
    total_step_test=math.ceil(len(test_dataset) / batch_size)
    #print(total_step_train)

    for epoch in range(args.epochs):
        # train
        file = open("./accuracy_20_11.txt", 'a')
        '''train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data=train_dataset,
                                                device=next(model.parameters()).device,
                                                epoch=epoch,
                                                round=total_step_train)'''
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=next(model.parameters()).device,
                                                epoch=epoch)
        scheduler.step()

        # validate
        '''val_loss, val_acc = evaluate(model=model,
                                     data=val_dataset,
                                     device=next(model.parameters()).device,
                                     epoch=epoch,
                                     round=total_step_val)'''

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=next(model.parameters()).device,
                                     epoch=epoch
                                    )
        
        '''test_acc=test(model=model,data=test_dataset,device=next(model.parameters()).device,epoch=epoch,round=total_step_test)'''
        test_acc=test(model=model,data_loader=test_loader,device=next(model.parameters()).device,epoch=epoch)
        #acccc=torch.load('G:/uiuc/deep-learning-for-image-processing-master/pytorch_classification/vision_transformer/model_best.pth.tar')
        #model_new=modelss.vgg16()
        #test_acc1=test_model(model=model_new.load_state_dict(torch.load('G:/uiuc/deep-learning-for-image-processing-master/pytorch_classification/vision_transformer/model_best.pth.tar')['model']),data=test_dataset,device=device,epoch=epoch,round=total_step_test)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate","test_acc"]
        #tb_writer.add_scalar(tags[0], train_loss, epoch)
        #tb_writer.add_scalar(tags[1], train_acc, epoch)
        #tb_writer.add_scalar(tags[2], val_loss, epoch)
        #tb_writer.add_scalar(tags[3], val_acc, epoch)
        #tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        #tb_writer.add_scalar(tags[5], test_acc, epoch)

        torch.save(model.state_dict(), "/hy-tmp/weights_20_11/model-{}.pth".format(epoch))
        file.write("epoch is: ")
        file.write(str(epoch))
        file.write("train acc: ")
        file.write(str(train_acc))
        file.write("train loss: ")
        file.write(str(train_loss))
        file.write(", valid acc: ")
        file.write(str(val_acc))
        file.write(", valid loss: ")
        file.write(str(val_loss))
        file.write(", test accuracy is: ")
        file.write(str(test_acc))
        file.write("\n")
        file.close()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--lrf', type=float, default=0.5)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    '''parser.add_argument('--data-path', type=str,
                        default="G:/uiuc/deep-learning-for-image-processing-master/pytorch_classification/vision_transformer/flower_photos")
    parser.add_argument('--test-data-path', type=str,
                        default="G:/uiuc/deep-learning-for-image-processing-master/pytorch_classification/vision_transformer/flower_photos/test")'''
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/hy-tmp/pretrain_weight/base_32_224_21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    #parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument("--local_rank", type=int)

    opt = parser.parse_args()

    main(opt)
