"""
训练进度与指标记录工具

本模块提供两个轻量级工具类，用于在深度学习训练循环中方便地记录和展示数值指标（如损失、准确率），
并以整齐的表格形式打印训练进度。特别适合在 epoch 循环内跟踪每个 batch 的性能变化。

主要类
-------
AverageMeter
    计算并存储某个指标的当前值、累计和、计数以及平均值。
    典型用法：在每个 batch 后调用 update() 更新，通过 avg 属性获取当前 epoch 的平均值。

ProgressMeter
    管理多个 AverageMeter 对象，并结合当前 batch 序号及总 batch 数，生成格式化的进度条字符串，
    可以打印或返回用于日志记录。

使用场景
--------
- 在训练循环中统计 loss、accuracy、AUC 等标量指标。
- 在验证循环中记录不同 batch 的指标，并计算整体平均值。
- 配合 tqdm 等工具输出简洁的进度信息。
"""
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    def get_display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)