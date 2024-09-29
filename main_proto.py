import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader
from CPEA.utils import level2_mapping

from dataloader.samplers import CategoriesSampler
from utils import ensure_path, Averager, compute_confidence_interval
from tensorboardX import SummaryWriter
from types import SimpleNamespace
import gc

args = SimpleNamespace(
    max_epoch=2,
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
    exp='CPEA'
)
save_path = '-'.join([args.exp, args.dataset, args.model_type])
args.save_path = osp.join('./results', save_path)
ensure_path(args.save_path)

torch.cuda.empty_cache()
gc.collect()

from dataloader.hfc100 import FC100 as Dataset

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


    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights
    from models.protonet import ProtoNet

    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Flatten()

    classifier = ProtoNet(backbone)

    backbone_optimizer = torch.optim.Adam([{'params': backbone.parameters()}], lr=args.lr, weight_decay=0.001)
    backbone_scheduler = torch.optim.lr_scheduler.StepLR(backbone_optimizer, step_size=args.step_size, gamma=args.gamma)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr * args.lr_mul, weight_decay=0.001)
    classifier_scheduler = torch.optim.lr_scheduler.StepLR(classifier_optimizer, step_size=args.step_size, gamma=args.gamma)


    def save_model(name):
        torch.save(dict(params=backbone.state_dict()), osp.join(args.save_path, name + '.pth'))
        torch.save(dict(params=classifier.state_dict()),
                   osp.join(args.save_path, name + '_classifier.pth'))



    global_count = 0
    writer = SummaryWriter(comment=args.save_path)
    torch.cuda.empty_cache()

    trlog = {'args': vars(args), 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'max_acc': 0.0,
             'max_acc_epoch': 0}

    torch.cuda.empty_cache()
    gc.collect()


    def create_mapping(labels):
        labels_mapping = {}
        current_index = 0

        for label in labels:
            if label not in labels_mapping.keys():
                labels_mapping[label] = current_index
                current_index += 1

        return labels_mapping


    backbone_dict = {}
    classifier_dict = {}

    for i in range(1 + len(level2_mapping)):
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Flatten()
        backbone_dict[i] = backbone
        classifier_dict[i] = ProtoNet(backbone)

    import random

    loss_fn = nn.CrossEntropyLoss()
    random.seed(0)
    torch.manual_seed(0)

    for epoch in range(1, args.max_epoch + 1):
        print(f"epoch {epoch}")
        # get parent level model
        backbone = backbone_dict[0]
        classifier = classifier_dict[0]

        backbone.train()
        classifier.train()
        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            print(f"batch {i}")
            loss = 0
            backbone.zero_grad()
            classifier.zero_grad()
            global_count = global_count + 1

            data, labels, parent_names = batch
            parent_labels = [level2_mapping[label_name] for label_name in parent_names[0]]

            p = args.shot * args.test_way
            support_data, query_data = data[:p], data[p:]
            support_labels, query_labels = labels[:p], labels[p:]
            parent_support_labels, parent_query_labels = parent_labels[:p], parent_labels[p:]

            parent_labels_support_mapping = create_mapping(parent_support_labels)

            mapped_parent_support_labels = torch.Tensor(
                [parent_labels_support_mapping[label] for label in np.array(parent_support_labels)])
            mapped_parent_query_labels = torch.Tensor(
                [parent_labels_support_mapping[label] for label in np.array(parent_query_labels)])

            # mapped_query_labels = torch.arange(args.test_way).repeat(args.query).long()
            # scores = classifier(support_data, support_labels, query_data, n_way)
            # loss = loss_fn(scores, mapped_query_labels)

            n_way_parent = len(mapped_parent_support_labels.unique())
            scores_parent = classifier(support_data, mapped_parent_support_labels, query_data, n_way_parent)
            loss_parent = loss_fn(scores_parent, mapped_parent_query_labels.long())
            pred_parent = torch.argmax(scores_parent, dim=1)

            loss += loss_parent

            for ind, query_image in enumerate(query_data):
                print(ind)

                true_class = parent_query_labels[ind]
                true_class_mapped = int(mapped_parent_query_labels[ind].item())
                true_subclasses = class_to_subclass_dict[true_class]

                sub_backbone = backbone_dict[ind]
                sub_backbone.train()
                sub_backbone.zero_grad()

                sub_classifier = classifier_dict[ind]
                sub_classifier.train()
                sub_classifier.zero_grad()

                subclass_support_images = support_data[mapped_parent_support_labels == true_class_mapped]
                subclass_support_labels = support_labels[mapped_parent_support_labels == true_class_mapped]

                subclass_support_mapping = create_mapping(np.array(subclass_support_labels))
                subclass_query_mapping = create_mapping(np.array(query_labels))

                mapped_subclass_support_labels = torch.Tensor(
                    [subclass_support_mapping[label] for label in np.array(subclass_support_labels)])
                mapped_subclass_query_label = torch.Tensor([subclass_support_mapping[query_labels[ind].item()]])

                n_way_subclass = len(mapped_subclass_support_labels.unique())
                scores_subclass = sub_classifier(subclass_support_images, mapped_subclass_support_labels,
                                                 query_data[ind].unsqueeze(0), n_way_subclass)
                pred_subclass = torch.argmax(scores_subclass, dim=1)

                loss_subclass = loss_fn(scores_subclass, mapped_subclass_query_label.long())
                loss += loss_subclass

            # pred = torch.argmax(scores, dim=1)
            # acc = (pred == mapped_query_labels).type(torch.FloatTensor).mean().item()
            writer.add_scalar('data/loss', float(loss), global_count)
            # writer.add_scalar('data/acc', float(acc), global_count)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), -1))

            # tl.add(loss.item())
            # ta.add(acc)

            loss.backward()
            backbone_optimizer.step()
            classifier_optimizer.step()
        '''
        backbone_scheduler.step()
        classifier_scheduler.step()
    
        tl = tl.item()
        ta = ta.item()
    
        backbone.eval()
        classifier.eval()
    
        vl = Averager()
        va = Averager()
    
        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                data, labels = batch
                p = args.shot * args.test_way
                support_data, query_data = data[:p], data[p:]
                support_labels, query_labels = labels[:p], labels[p:]
                mapped_query_labels = torch.arange(args.test_way).repeat(args.query).long()
    
                scores = classifier(support_data, support_labels, query_data
                                                   )
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
        '''

    trlog = torch.load(osp.join(args.save_path, 'trlog'))

    backbone.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    backbone.eval()

    classifier.load_state_dict(
        torch.load(osp.join(args.save_path, 'max_acc' + '_classifier.pth'))['params'])
    classifier.eval()
    # %%
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

            scores = classifier(support_data, support_labels, query_data
                                )
            loss = loss_fn(scores, ls)
            pred = torch.argmax(scores, dim=1)
            acc = (pred == ls).type(torch.FloatTensor).mean().item()

            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            print('batch {}: acc {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    m, pm = compute_confidence_interval(test_acc_record)
    print('Val Best Epoch {}, Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print('Test Acc {:.4f} + {:.4f}'.format(m * 100, pm * 100))