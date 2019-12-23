import argparse

def _get_train_opt():
    parser = argparse.ArgumentParser(description = 'Structure-Aware Residual Pyramid Network for Monocular Depth Estimation(SARPN)')
    
    parser.add_argument('--gpu_ids', type=int, default=0, help='gpu id to use')
    parser.add_argument('--save_dir_model', type=str, default='/results/model_train', help='dir to save model')
    parser.add_argument('--save_dir_res', type=str, default='/results/result_train', help='dir to save result train')

    parser.add_argument('--feature_extractor', type=str, default='SENet154', help='select a network as backbone')
    parser.add_argument('--extractor_path', type=str, default='/senet154-c7b49a05.pth', help='dir where feature extractor is present')

    parser.add_argument('--trainlist_path',type=str,default='/data/data/nyu2_train.csv', help='the path of trainlist')

    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')

    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--resume', action='store_true',default=False, help='continue training the model')

    parser.add_argument('--optimizer', default="adam", type=str, help="Optimizer selection")
    parser.add_argument('--root_path', help="the root path of dataset")

    parser.add_argument('--save_itr',type=int,default=5, help="iteration number after which to save")
    return parser.parse_args()

