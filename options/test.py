import argparse

def _get_train_opt():
    parser = argparse.ArgumentParser(description = 'Structure-Aware Residual Pyramid Network for Monocular Depth Estimation(SARPN)')
    parser.add_argument('--backbone', type=str, default='SENet154', help='select a network as backbone')
    parser.add_argument('--trainlist_path', required=True, help='the path of trainlist')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--resume', action='store_true',default=False, help='continue training the model')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum parameter used in the Optimizer.')
    parser.add_argument('--epsilon', default=0.001, type=float, help='epsilon')
    parser.add_argument('--optimizer_name', default="adam", type=str, help="Optimizer selection")
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--checkpoint_dir', required=True, help="the directory to save the checkpoints")
    parser.add_argument('--root_path', required=True, help="the root path of dataset")
    parser.add_argument('--loadckpt', default=None, help="Specify the path to a specific model")
    parser.add_argument('--logdir', required=True, help="the directory to save logs and checkpoints")
    parser.add_argument('--do_summary', action='store_true', default=False, help='whether do summary or not')
    parser.add_argument('--pretrained_dir', required=True,type=str, help="the path of pretrained models")
    return parser.parse_args()