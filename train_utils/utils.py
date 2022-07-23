# implement utils of accuracy and train data
import torch
import matplotlib.pyplot as plt

__all__ = ['accuracy', 'AverageMeter', 'try_gpu', 'try_all_gpu', 'train', 'plot_figure']


# implement accuracy function
def accuracy(output, targets, top_k=(1,)):
    batch_size = output.shape[0]
    k = max(top_k)
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# implement average_num class
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# implement device choose
def try_gpu(i=0):
    """return cuda{i} if exist else 'cpu"""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')


def try_all_gpu():
    """return all gpu if gpu exist else cpu"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else torch.device('cpu')


# 训练函数
def train(model, dataloader, criterion, optimizer, valid_loader=None):
    loss_train = AverageMeter()
    acc_train = AverageMeter()
    model.eval()
    model.train()
    for image, labels in dataloader:
        image, labels = image.cuda(), labels.cuda()
        outputs = model(image)
        l = criterion(outputs, labels)
        correct = accuracy(outputs, labels, top_k=(1, ))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss_train.update(l.item(), image.size(0))
        acc_train.update(correct[0].item(), image.size(0))
    print(f'train_loss:{loss_train.avg}, train_acc:{acc_train.avg}')
    if valid_loader:
        model.eval()
        with torch.no_grad():
            loss_valid = AverageMeter()
            acc_valid = AverageMeter()
            for image, labels in valid_loader:
                image, labels = image.cuda(), labels.cuda()
                outputs = model(image)
                l = criterion(outputs, labels)
                correct = accuracy(outputs, labels, top_k=(1, ))
                loss_valid.update(l.item(), image.size(0))
                acc_valid.update(correct[0].item(), image.size(0))
            print(f'valid_loss:{loss_valid.avg}, valid_acc:{acc_valid.avg}')
        return loss_train.avg, acc_train.avg, loss_valid.avg, acc_valid.avg
    else:
        return loss_train.avg, acc_train.avg, 0, 0


# 绘制训练图像
def plot_figure(history, save_figure):
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    valid_loss = history['valid_loss']
    valid_acc = history['valid_acc']
    x = range(len(train_loss))
    plt.figure(figsize=(40, 16), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(x, train_loss, label='train_loss')
    if valid_loss:
        plt.plot(x, valid_loss, label='valid_loss')
    plt.grid(True)
    plt.title('loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, train_acc, label='train_acc')
    if valid_acc:
        plt.plot(x, valid_acc, label='valid_acc')
    plt.grid(True)
    plt.title('acc')
    plt.legend()
    plt.savefig(save_figure)










