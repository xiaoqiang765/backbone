# implement net hyper-parameters config
__all__ = ['args']

args = {'num_epochs': 400,
        'optimizer': 'SGD',
        'optimizer_para': {'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 1e-4},
        'scheduler': 'CosineAnnealingLR',
        'scheduler_para': {'T_max': 400},
        'batch_size': 32}
