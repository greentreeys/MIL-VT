import copy
import time
from datetime import datetime, timedelta

import PIL
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import utils
from data_manager.dataset import dataset_Aptos
from loss.MultiClassMetrics import *
from models.FinetuneVTmodels import MIL_VT_FineTune
from utils import *

####################################

def main():

    """Basic Setting"""
    data_path = r'data/APTOS/Image/'
    csv_path = r'data/APTOS/CSV/'
    save_model_path = r'data/APTOS/PytorchModel/'
    csvName = csv_path + 'train.csv'  ##the csv file store the path of image and corresponding label

    gpu_ids = [0, 1]
    start_epoch = 0
    max_epoch = 30
    save_fraq = 10

    batch_size = 16
    img_size = 512
    initialLR = 2e-5
    n_classes = 5

    balanceFlag = True  #balanceFlag is set to True to balance the sampling of different classes
    debugFlag = False  #Debug flag is set to True to train on a small dataset

    base_model = 'MIL_VT_small_patch16_'+str(img_size)  #nominate the MIL-VT model to be used
    MODEL_PATH_finetune = 'weights/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar'

    dateTag = datetime.today().strftime('%Y%m%d')
    prefix = base_model + '_' + dateTag
    model_save_dir = os.path.join(save_model_path,  prefix)
    tbFileName = os.path.join(model_save_dir, 'runs/' + prefix)
    savemodelPrefix = prefix + '_ep'

    ##resume training with an interrupted model
    resumeFlag = False
    resumeEpoch = 0
    resumeModel = 'path to resume model'

    print('####################################################')
    print('Save model Path', model_save_dir)
    print('Save training record Path', tbFileName)
    print('####################################################')

    #################################################
    sys.stdout = Logger(os.path.join(model_save_dir,
                     savemodelPrefix[:-3] + 'log_train-%s.txt' % time.strftime("%Y-%m-%d-%H-%M-%S")))
    tbWriter = SummaryWriter(tbFileName)

    torch.cuda.set_device(gpu_ids[0])
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name(gpu_ids[0]), torch.cuda.get_device_capability(gpu_ids[0]))

    #################################################
    """Set up the model, loss function and optimizer"""

    ## set the model and assign the corresponding pretrain weight
    model = MIL_VT_FineTune(base_model, MODEL_PATH_finetune, num_classes=n_classes)
    model = model.cuda()
    if len(gpu_ids) >= 2:
        model = DataParallel(model, device_ids=gpu_ids)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    multiLayers = list()
    for name, layer in model._modules.items():
        if name.__contains__('MIL_'):
            multiLayers.append({'params': layer.parameters(), 'lr': 5*initialLR})
        else:
            multiLayers.append({'params': layer.parameters()})
    optimizer = torch.optim.Adam(multiLayers, lr = initialLR, eps=1e-8, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss().cuda()

    if resumeFlag:
        print(" Loading checkpoint from epoch '%s'" % (
             resumeEpoch))
        checkpoint = torch.load(resumeModel)
        initialLR = checkpoint['lr']
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print('Model weight loaded')

    #################################################
    """Load the CSV as DF and split train / valid set"""
    DF0= pd.read_csv(csvName, encoding='UTF')

    if debugFlag == True:
        indexes = np.arange(len(DF0))
        np.random.seed(0)
        np.random.shuffle(indexes)
        DF0 = DF0.iloc[indexes[:600], :]
        DF0 = DF0.reset_index(drop=True)

    indexes = np.arange(len(DF0))
    np.random.seed(0)
    np.random.shuffle(indexes)
    trainNum = np.int(len(indexes)*0.7)
    valNum = np.int(len(indexes)*0.8)
    DF_train = DF0.loc[indexes[:trainNum]]
    DF_val = DF0.loc[indexes[trainNum:valNum]]
    DF_test = DF0.loc[indexes[valNum:]]
    DF_train = DF_train.reset_index(drop=True)
    DF_val = DF_val.reset_index(drop=True)
    DF_test = DF_test.reset_index(drop=True)

    print('Train: ', len(DF_train), 'Val: ', len(DF_val), 'Test: ', len(DF_test))
    for tempLabel in [0,1,2,3,4]:
        print(tempLabel, np.sum(DF_train['diagnosis']==tempLabel),\
                        np.sum(DF_val['diagnosis']==tempLabel),
                        np.sum(DF_test['diagnosis']==tempLabel))

    #################################################

    transform_train = transforms.Compose([
        transforms.Resize((img_size+40, img_size+40)),
        transforms.RandomCrop((img_size, img_size)),  #padding=10
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset_train = dataset_Aptos(data_path, DF_train, transform = transform_train)
    dataset_valid = dataset_Aptos(data_path, DF_val, transform = transform_test)
    dataset_test = dataset_Aptos(data_path, DF_test, transform=transform_test)

    """assign sample weight to deal with the unblanced classes"""
    weights = make_weights_for_balanced_classes(DF_train, n_classes)                                                           
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    if balanceFlag == True:
        train_loader = DataLoader(dataset_train, batch_size,
                              sampler = sampler,
                              num_workers=8,  drop_last=True, shuffle=False) #shuffle=False when using the balance sampler,
    else:
        train_loader = DataLoader(dataset_train, batch_size, num_workers=8,  drop_last=True, shuffle=True) #shuffle=True,
    valid_loader = DataLoader(dataset_valid, batch_size, num_workers=8, drop_last=False)
    test_loader = DataLoader(dataset_test, batch_size, num_workers=8,  drop_last=False)

    #################################################

    """The training procedure"""

    start_time = time.time()
    train_time = 0
    best_perform = 0
    for epoch in range(start_epoch, max_epoch + 1):
        start_train_time = time.time()
        currentLR = 0

        for param_group in optimizer.param_groups:
            currentLR = param_group['lr']
        print('lr:', currentLR)

        train(epoch, model, criterion, optimizer, train_loader, max_epoch, tbWriter)
        train_time += np.round(time.time() - start_train_time)

        AUC_val, wF1_val = \
            val(epoch, model, criterion, valid_loader, max_epoch, tbWriter)

        if wF1_val > best_perform and debugFlag == False:
            best_perform = wF1_val
            state_dict = model_without_ddp.state_dict()
            saveCheckPointName = savemodelPrefix + '_bestmodel.pth.tar'
            utils.save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
            }, os.path.join(model_save_dir, saveCheckPointName))
            best_model = copy.deepcopy(model)
            print('Checkpoint saved, ', saveCheckPointName)

        if epoch>0 and (epoch) % save_fraq == 0 and debugFlag == False:

            state_dict = model_without_ddp.state_dict()
            wF1_val = round(wF1_val, 3)
            saveCheckPointName = savemodelPrefix + str(epoch) + '_' + str(wF1_val) + '.pth.tar'
            utils.save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
            }, os.path.join(model_save_dir, saveCheckPointName))
            print('Checkpoint saved, ', saveCheckPointName)


    elapsed = np.round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    train_time = str(timedelta(seconds=train_time))


    print('###################################################')
    print('Performance on Test Set with last model')
    test(epoch, model, criterion, test_loader, tbWriter)

    print('###################################################')
    print('Performance on Test Set with best model')
    test(epoch, best_model, criterion, test_loader, tbWriter)


    tbWriter.close()
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))



def train(epoch, model, criterion, optimizer, train_loader, max_epoch,  tbWriter):
    start_time = time.time()
    model.train()
    losses = utils.AverageMeter()
    losses1 = utils.AverageMeter()
    losses2 = utils.AverageMeter()

    ground_truths_multiclass = []
    ground_truths_multilabel = []

    predictions_class = []
    total = 0
    for batch_idx, (inputs, labels_multiclass, labels_onehot) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets_class = labels_multiclass.cuda()
        # # print(targets_class)

        outputs_class, outputs_MIL = model(inputs)
        loss1 = criterion(outputs_class, targets_class)
        loss2 = criterion(outputs_MIL, targets_class)
        loss = 0.5*loss1 + 0.5*loss2

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()

        losses.update(loss.data.cpu().numpy())
        losses1.update(loss1.data.cpu().numpy())
        losses2.update(loss2.data.cpu().numpy())

        ###Update learning rate
        steps = len(train_loader)*epoch + batch_idx
        if steps>0 and steps % 2000 == 0:
            print('steps: ', steps)
            adjust_learning_rate(optimizer, 0.5)

        total += targets_class.size(0)
        _, predicted_class = torch.max(outputs_class.data, 1)

        #"""Save the losses to tensorboard"""
        tbWriter.add_scalar('AllLoss/train', loss, steps)

        outputs_class = utils.softmax(outputs_class.data.cpu().numpy())
        ground_truths_multiclass.extend(labels_multiclass.data.cpu().numpy())
        ground_truths_multilabel.extend(labels_onehot.data.cpu().numpy())
        predictions_class.extend(outputs_class)

    """Mesure the prediction performance on train set"""
    gts = np.asarray(ground_truths_multiclass)
    probs = np.asarray(predictions_class)
    preds = np.argmax(probs, axis=1)
    accuracy = metrics.accuracy_score(gts, preds)

    gts2 = np.asarray(ground_truths_multilabel)
    trues = np.asarray(gts2).flatten()
    probs2 = np.asarray(probs).flatten()
    AUC = metrics.roc_auc_score(trues, probs2)

    wKappa = metrics.cohen_kappa_score(gts, preds, weights='quadratic')
    wF1 = metrics.f1_score(gts, preds, average='weighted')

    """Save the performance to tensorboard"""
    tbWriter.add_scalar('accuraccy/train', accuracy, epoch)
    tbWriter.add_scalar('AUC/train', AUC, epoch)
    tbWriter.add_scalar('wF1/train', wF1, epoch)
    tbWriter.add_scalar('wKapa/train', wKappa, epoch)
    tbWriter.add_scalar('Loss/train', losses.avg, epoch)


    end_time = time.time()
    print('\t Epoch {}/{}'.format(epoch, max_epoch))
    print('\t Train:    Acc %0.3f,  AUC %0.3f, weightedKappa %0.3f, weightedF1 %0.3f, loss1 %.6f, loss2 %.6f, time %3.2f'
        % (accuracy, AUC, wKappa, wF1, losses1.avg, losses2.avg, end_time - start_time))
    return  0


def val(epoch, model, criterion, val_loader, max_epoch, tbWriter):
    start_time = time.time()
    model.eval()
    losses = utils.AverageMeter()


    ground_truths_multiclass = []
    ground_truths_multilabel = []
    predictions_class = []
    scores = []
    total = 0

    for batch_idx, (inputs,  labels_multiclass, labels_onehot) in enumerate(val_loader):
        inputs = Variable(inputs.cuda())
        targets_class = Variable(labels_multiclass.cuda())

        outputs_class = model(inputs)
        loss = criterion(outputs_class, targets_class) #targets_class

        losses.update(loss.data.cpu().numpy())

        outputs_class = utils.softmax(outputs_class.data.cpu().numpy())
        ground_truths_multiclass.extend(labels_multiclass.data.cpu().numpy())
        ground_truths_multilabel.extend(labels_onehot.data.cpu().numpy())
        predictions_class.extend(outputs_class)

        total += targets_class.size(0)

    """Mesure the prediction performance on valid set"""
    gts = np.asarray(ground_truths_multiclass)
    probs = np.asarray(predictions_class)
    preds = np.argmax(probs, axis=1)
    accuracy = metrics.accuracy_score(gts, preds)

    gts2 = np.asarray(ground_truths_multilabel)
    trues = np.asarray(gts2).flatten()
    probs2 = np.asarray(probs).flatten()
    AUC = metrics.roc_auc_score(trues, probs2)

    wKappa = metrics.cohen_kappa_score(gts, preds, weights='quadratic')
    wF1 = metrics.f1_score(gts, preds, average='weighted')

    """Save the performance to tensorboard"""
    tbWriter.add_scalar('accuraccy/valid', accuracy, epoch)
    tbWriter.add_scalar('AUC/valid', AUC, epoch)
    tbWriter.add_scalar('wF1/valid', wF1, epoch)
    tbWriter.add_scalar('wKapa/valid', wKappa, epoch)
    tbWriter.add_scalar('Loss/valid', losses.avg, epoch)


    end_time = time.time()
    print('-------- Epoch {}/{}'.format(epoch, max_epoch))
    print('-------- Val:  Acc %0.3f,  AUC %0.3f, weightedKappa %0.3f, weightedF1 %0.3f, loss %.6f, time %3.2f'
        % (accuracy, AUC, wKappa, wF1, losses.avg,  end_time - start_time))
    print('=============================================================')
    return AUC, wF1



def test(epoch, model, criterion, test_loader,  tbWriter):
    start_time = time.time()
    model.eval()
    losses = utils.AverageMeter()


    ground_truths_multiclass = []
    ground_truths_multilabel = []
    predictions_class = []
    total = 0

    for batch_idx, (inputs, labels_multiclass, labels_onehot) in enumerate(test_loader):
        inputs = Variable(inputs.cuda())
        targets_class = Variable(labels_multiclass.cuda())

        outputs_class = model(inputs)
        loss = criterion(outputs_class, targets_class) #targets_class

        losses.update(loss.data.cpu().numpy())

        outputs_class = utils.softmax(outputs_class.data.cpu().numpy())
        ground_truths_multiclass.extend(labels_multiclass.data.cpu().numpy())
        ground_truths_multilabel.extend(labels_onehot.data.cpu().numpy())
        predictions_class.extend(outputs_class)
        total += targets_class.size(0)


    """Mesure the prediction performance on test set"""
    gts = np.asarray(ground_truths_multiclass)
    probs = np.asarray(predictions_class)
    preds = np.argmax(probs, axis=1)
    accuracy = metrics.accuracy_score(gts, preds)

    gts2 = np.asarray(ground_truths_multilabel)
    trues = np.asarray(gts2).flatten()
    probs2 = np.asarray(probs).flatten()
    AUC = metrics.roc_auc_score(trues, probs2)

    wKappa = metrics.cohen_kappa_score(gts, preds, weights='quadratic')
    wF1 = metrics.f1_score(gts, preds, average='weighted')

    """Save the performance to tensorboard"""
    tbWriter.add_scalar('accuraccy/test', accuracy, epoch)
    tbWriter.add_scalar('AUC/test', AUC, epoch)
    tbWriter.add_scalar('wF1/test', wF1, epoch)
    tbWriter.add_scalar('wKapa/test', wKappa, epoch)
    tbWriter.add_scalar('Loss/test', losses.avg, epoch)

    end_time = time.time()
    print( 'TestSet:  Acc %0.3f,  AUC %0.3f, weightedKappa %0.3f, weightedF1 %0.3f, time %3.2f'
        % (accuracy, AUC, wKappa, wF1,  end_time - start_time))
    print('=============================================================')
    return AUC, wF1



if __name__ == '__main__':
    main()

