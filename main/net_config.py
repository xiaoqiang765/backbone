# implement net hyper-parameters config
__all__ = ['args']

args = {'num_epochs': 100,
        'optimizer': 'SGD',
        'optimizer_para': {'lr': 1e-3, 'momentum': 0.9},
        'scheduler': 'CosineAnnealingLR',
        'scheduler_para': {'T_max': 100},
        'batch_size': 128}
