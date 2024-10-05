import os.path as osp
import os
import numpy as np
import torch
import gc
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from utils import level2_mapping
from dataloader.samplers import CategoriesSampler
from utils import ensure_path, Averager, compute_confidence_interval, create_mapping
from types import SimpleNamespace
from torch import nn
from models.protonet import ProtoNet
from dataloader.hfc100 import FC100 as Dataset

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

random.seed(0)
torch.manual_seed(0)

args = SimpleNamespace(
    max_epoch=1,
    way=5,
    test_way=5,
    shot=1,
    query=15,
    lr=0.00001,
    lr_mul=100,
    step_size=5,
    gamma=0.5,
    model_type='small',
    dataset='FC100',
    init_weights='./initialization/fc100/checkpoint1600.pth',
    gpu='0',
    exp='hfs'
)
save_path = '-'.join([args.exp, args.dataset, args.model_type])
args.save_path = osp.join('./results', save_path)
ensure_path(args.save_path)

torch.cuda.empty_cache()
gc.collect()


def save_model(name):
    torch.save(dict(params=backbone.state_dict()), osp.join(args.save_path, name + '.pth'))
    torch.save(dict(params=classifier.state_dict()),
               osp.join(args.save_path, name + '_classifier.pth'))


if __name__ == '__main__':
    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, 10, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 10, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label, 10, args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=8, pin_memory=True)

    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Flatten()

    classifier = ProtoNet(backbone)

    backbone_optimizer = torch.optim.Adam([{'params': backbone.parameters()}], lr=args.lr, weight_decay=0.001)
    backbone_scheduler = torch.optim.lr_scheduler.StepLR(backbone_optimizer, step_size=args.step_size, gamma=args.gamma)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr * args.lr_mul, weight_decay=0.001)
    classifier_scheduler = torch.optim.lr_scheduler.StepLR(classifier_optimizer, step_size=args.step_size,
                                                           gamma=args.gamma)

    global_count = 0
    writer = SummaryWriter(comment=args.save_path)
    trlog = {'args': vars(args), 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'max_acc': 0.0,
             'max_acc_epoch': 0}

    loss_fn = nn.CrossEntropyLoss()

    # TRAINING
    for epoch in range(1, args.max_epoch + 1):
        print(f"epoch {epoch}")
        backbone.train()
        classifier.train()
        tl = Averager()
        ta = Averager()
        loss = 0

        for i, batch in enumerate(train_loader, 1):
            backbone.zero_grad()
            classifier.zero_grad()
            global_count = global_count + 1

            p = args.shot * args.test_way

            data, labels, parent_names = batch
            parent_labels = [level2_mapping[label_name] for label_name in parent_names[0]]

            support_data, query_data = data[:p], data[p:]
            support_labels, query_labels = labels[:p], labels[p:]
            parent_support_labels, parent_query_labels = parent_labels[:p], parent_labels[p:]
            parent_labels_support_mapping = create_mapping(parent_support_labels)
            mapped_parent_support_labels = torch.Tensor(
                [parent_labels_support_mapping[label] for label in np.array(parent_support_labels)])
            mapped_parent_query_labels = torch.Tensor(
                [parent_labels_support_mapping[label] for label in np.array(parent_query_labels)])

            n_way = len(support_labels.unique())
            mapped_query_labels = torch.arange(args.test_way).repeat(args.query).long()
            scores = classifier(support_data, support_labels, query_data, n_way)
            loss = loss_fn(scores, mapped_query_labels)

            pred = torch.argmax(scores, dim=1)
            acc = (pred == mapped_query_labels).type(torch.FloatTensor).mean().item()
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            loss.backward()
            backbone_optimizer.step()
            classifier_optimizer.step()

        backbone_scheduler.step()
        classifier_scheduler.step()

        tl = tl.item()
        ta = ta.item()

        backbone.eval()
        classifier.eval()

        vl = Averager()
        va = Averager()

        # VALIDATION
        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                data, labels, parent_names = batch
                p = args.shot * args.test_way
                support_data, query_data = data[:p], data[p:]
                support_labels, query_labels = labels[:p], labels[p:]
                mapped_query_labels = torch.arange(args.test_way).repeat(args.query).long()

                n_way = len(support_labels.unique())
                scores = classifier(support_data, support_labels, query_data, n_way)
                loss = loss_fn(scores, mapped_query_labels)
                pred = torch.argmax(scores, dim=1)
                acc = (pred == mapped_query_labels).type(torch.FloatTensor).mean().item()
                vl.add(loss.item())
                va.add(acc)

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), epoch)
        writer.add_scalar('data/val_acc', float(va), epoch)
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va >= trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        save_model('epoch-last')

    trlog = torch.load(osp.join(args.save_path, 'trlog'))

    backbone.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    backbone.eval()

    classifier.load_state_dict(
        torch.load(osp.join(args.save_path, 'max_acc' + '_classifier.pth'))['params'])
    classifier.eval()

    # TESTING
    loss_fn = nn.CrossEntropyLoss()
    test_acc_record = np.zeros((10,))
    ave_acc = Averager()

    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            data, labels, parent_labels = batch
            p = args.shot * args.test_way
            support_data, query_data = data[:p], data[p:]
            support_labels, query_labels = labels[:p], labels[p:]
            ls = torch.arange(args.test_way).repeat(args.query).long()
            n_way = len(support_labels.unique())

            scores = classifier(support_data, support_labels, query_data, n_way)
            loss = loss_fn(scores, ls)
            pred = torch.argmax(scores, dim=1)
            acc = (pred == ls).type(torch.FloatTensor).mean().item()

            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            print('batch {}: loss {:.2f} acc {:.2f}({:.2f})'.format(i, loss, ave_acc.item() * 100, acc * 100))

    m, pm = compute_confidence_interval(test_acc_record)
    print('Val Best Epoch {}, Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print('Test Acc {:.4f} + {:.4f}'.format(m * 100, pm * 100))
