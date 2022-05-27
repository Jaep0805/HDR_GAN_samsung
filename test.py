import argparse
import importlib
import os

from config import config
from utils import parse_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', dest='CKPT_FILE', default = "/data2/jaep0805/HDR-GAN/logs/2022.05.23_21.38.43_11066_unetpps_sphere_sn_inHDR_oHDR_lsTP/best_model-320000", required=False)#no default
    parser.add_argument('--tag', dest='TAG', default=...)
    parser.add_argument('--gpu', dest='CUDA_VISIBLE_DEVICES', default= 2, type=int) #default = -1
    parser.add_argument('--ignore_config', default=False, action='store_true')
    parser.add_argument('--module', default='unetpps')
    parser.add_argument('--cus_test_ds', dest='TEST_DS', default=...)
    parser.add_argument('--test_hw', dest='test_hw', default=(1000,1500), nargs=2) #default = 1500, 1000
    parser.add_argument('--UNETPPS', default = True) #defualt not here

    module_name = 'test_{}'.format(parse_args('--module', 'unetpps'))
    test_module = importlib.import_module(module_name)
    parser = test_module.args(parser)
    args = parser.parse_args()

    if not args.ignore_config:
        ckpt_dir = os.path.dirname(args.CKPT_FILE)
        config_file = os.path.join(ckpt_dir, 'config.yml')
        assert os.path.isfile(config_file)
        config.load(config_file)
    config.apply(args)
    test_module.test()
