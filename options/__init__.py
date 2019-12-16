from .train import _get_train_opt
# from .test import _get_test_opt

def get_args(mode):
    args = None
    if mode == 'train':
        args =  _get_train_opt()
    else:
        raise ValueError("Invalid mode selection!")

    return args