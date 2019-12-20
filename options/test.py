import argparse

def _get_test_opt():
    parser = argparse.ArgumentParser(description = 'Structure-Aware Residual Pyramid Network for Monocular Depth Estimation(SARPN)')
    
    parser.add_argument('--gpu_ids', type=int, default=0, help='gpu id to use')
    parser.add_argument('--save_dir_model', type=str, default='/results/model_test', help='dir to save model')
    parser.add_argument('--save_dir_res', type=str, default='/results/result_test', help='dir to save result train')

    parser.add_argument('--feature_extractor', type=str, default='SENet154', help='select a network as backbone')
    parser.add_argument('--extractor_path', type=str, default='/senet154-c7b49a05.pth', help='dir where feature extractor is present')

    parser.add_argument('--testlist_path',type=str,default='/data/data/nyu2_test.csv', help='the path of trainlist')
    parser.add_argument('--model_path',type=str,default='/results/model_train/', help='the path of trainlist')

    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')

    parser.add_argument('--root_path', help="the root path of dataset")
    return parser.parse_args()


