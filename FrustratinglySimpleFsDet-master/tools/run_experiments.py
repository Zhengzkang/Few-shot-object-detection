import argparse
import os

import yaml
from subprocess import PIPE, STDOUT, Popen, CalledProcessError, check_call, check_output, run

from class_splits import CLASS_SPLITS
from fsdet.config.config import get_cfg
cfg = get_cfg()


def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset settings
    parser.add_argument('--dataset', type=str, required=True, choices=['coco', 'voc', 'isaid'])
    parser.add_argument('--class-split', type=str, required=True, dest='class_split')  # TODO: allow multiple class splits?
    # CPU and GPU settings
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--num-threads', type=int, default=1)
    # Model settings
    parser.add_argument('--layers', type=int, default=50, choices=[50, 101], help='Layers of ResNet backbone')
    parser.add_argument('--classifier', default='fc', choices=['fc', 'cosine'],
                        help='Use regular fc classifier or cosine classifier')
    parser.add_argument('--tfa', action='store_true',
                        help='Two-stage fine-tuning')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze feature extractor (backbone + proposal generator + roi-head)')
    parser.add_argument('--unfreeze-backbone', action='store_true', dest='unfreeze_backbone',
                        help='Unfreeze the backbone only')
    parser.add_argument('--unfreeze-proposal-generator', action='store_true', dest='unfreeze_proposal_generator',
                        help='Unfreeze the proposal generator (e.g. RPN) only')
    parser.add_argument('--unfreeze-roi-box-head-convs', type=int, nargs='*', default=[],
                        dest='unfreeze_roi_box_head_convs',
                        help="Unfreeze single bbox head conv layers. Layers are identified by numbers starting at 1.")
    parser.add_argument('--unfreeze-roi-box-head-fcs', type=int, nargs='*', default=[],
                        dest='unfreeze_roi_box_head_fcs',
                        help="Unfreeze single bbox head fc layers. Layers are identified by numbers starting at 1.")
    parser.add_argument('--max-iter', type=int, default=-1, dest='max_iter',
                        help='Override maximum iteration. '
                             'Set to -1 to use hard-coded defaults for each dataset and shot')
    parser.add_argument('--lr-decay-steps', type=int, nargs='*', default=[-1], dest='lr_decay_steps',
                        help='Override learning rate decay steps. '
                             'Set to [-1] to use hard-coded defaults for each dataset and shot')
    parser.add_argument('--ckpt-interval', type=int, default=-1, dest='ckpt_interval',
                        help='Override checkpoint interval. '
                             'Set to -1 to use hard-coded defaults for each dataset and shot')
    # Fine-Tuning settings
    parser.add_argument('--double-head', action='store_true',
                        help="use different predictor heads for base classes and novel classes")
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 5, 10],
                        help='Shots to run experiments over')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 20],
                        help='Range of seeds to run. Just a single seed or two seeds representing a range with 2nd '
                             'argument being inclusive as well!')
    parser.add_argument('--explicit-seeds', action='store_true', dest='explicit_seeds',
                        help='Specify a list of explicit seeds, rather than a range of seeds.')
    parser.add_argument('--bs', type=int, default=16, help='Total batch size, not per GPU!')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Set to -1 for automatic linear scaling')
    # Workflow settings
    parser.add_argument('--override-config', default=False, action='store_true',
                        help='Override config file if it already exists')
    parser.add_argument('--override-surgery', default=False, action='store_true',
                        help='Rerun surgery if the surgery checkpoint yet exists. '
                             'Normally not necessary, but can be used while debugging to trigger the surgery every run')
    # Misc
    # TODO: Add resume argument:
    #  resume==True: resume from last checkpoint (current default, no need for change)
    #  resume==False: delete all checkpoints and start fresh training!
    parser.add_argument('--root', type=str, default='./', help='Root of data')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of path')
    parser.add_argument('--ckpt-freq', type=int, default=10, # TODO: add an argument to enable or disable? (either fix amount of iterations or fix total amount of checkpoints)
                        help='Frequency of saving checkpoints')
    # TODO: add argument --eval-only which will just execute evaluations!
    #  -> How can we tell get_cfg that we just want the correct config without doing a surgery?
    # PASCAL arguments
    parser.add_argument('--split', '-s', type=int, default=1, help='Data split')

    args = parser.parse_args()
    return args


def get_empty_ft_config():
    return {
        '_BASE_': str,
        'MODEL': {
            'WEIGHTS': str,
            'MASK_ON': False,  # constant!
            'RESNETS': {
                'DEPTH': int
            },
            'ANCHOR_GENERATOR': {
                'SIZES': [[int]]
            },
            'RPN': {
                'PRE_NMS_TOPK_TRAIN': int,
                'PRE_NMS_TOPK_TEST': int,
                'POST_NMS_TOPK_TRAIN': int,
                'POST_NMS_TOPK_TEST': int
            },
            'ROI_HEADS': {
                'NAME': str,
                'NUM_CLASSES': int,
                'MULTIHEAD_NUM_CLASSES': [int],
                'SCORE_THRESH_TEST': float,
            },
            'ROI_BOX_HEAD': {
                'NAME': str,
                'NUM_HEADS': int,
                'SPLIT_AT_FC': int,
                'FREEZE_CONVS': [int],
                'FREEZE_FCS': [int]
            },
            'BACKBONE': {
                'FREEZE': bool
            },
            'PROPOSAL_GENERATOR': {
                'FREEZE': bool
            }
        },
        'DATASETS': {
            'TRAIN': (str,),
            'TEST': (str,)
        },
        'SOLVER': {
            'IMS_PER_BATCH': int,
            'BASE_LR': float,
            'STEPS': (int,),
            'MAX_ITER': int,
            'CHECKPOINT_PERIOD': int,
            'WARMUP_ITERS': int
        },
        'INPUT': {
            'MIN_SIZE_TRAIN': (int,)
        },
        'TEST': {
            'DETECTIONS_PER_IMAGE': int
        },
        'OUTPUT_DIR': str
    }


def load_yaml_file(fname):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_cmd(cmd):
    # Note: (see:  https://docs.python.org/3.8/library/subprocess.html#subprocess.run)
    # - We don't want to do anything with stdout or stderr (for now), so we just use 'subprocess.run' without 'PIPE'
    #   buffer
    # - we use check=True to abort the whole program if any passed cmd fails (otherwise the commands would fail silently
    #   which would impede debugging)
    run(cmd, shell=True, check=True)

    # Note: (see https://docs.python.org/3.8/library/subprocess.html#popen-constructor)
    # - If we were interested into  processing stdout and stderr, we could use following code using subprocess.Popen.
    #   We cannot use subprocess.run, because that method returns a 'CompletedProcess' instance, whereas
    #   subprocess.Popen returns a Popen class instance representing an active process.
    # - Note that, since we're using the PIPE buffer, we have to constantly pull the content of that buffer. Otherwise,
    #   the program could block if too much output is generated! If the buffer is not full, it outpust all its content
    #   as soon as the process is finished.
    # - It's dangerous to use PIPE together with Popen.wait(). It could lead to a deadlock if the subprocess produces
    #   enough output to fill the pipe buffer.

    #   process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    #   since we redirect both, stdout and stderr into PIPE, we have to read it, otherwise it could block
    #   for line in process.stdout:
    #       print(line.decode('utf-8'))
    #   process.stdout.close()
    #   return_code = process.wait()
    ##  We could also use
    ##  with Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True) as proc:
    ##      print(proc.stdout.read().decode())
    ##  for a very compact version (see https://docs.python.org/3.8/library/subprocess.html#popen-constructor)
    #   if return_code:
    #       print("Error in cmd: {}".format(cmd)) exit(1)

# deprecated run command
# def run_cmd(cmd):
#    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
#    while True:
#        line = p.stdout.readline().decode('utf-8')
#        if not line:
#            break
#        print(line)


def run_exp(config_file, config):
    """
    Run training and evaluation scripts based on given config files.
    """
    run_train(config_file, config)
    run_test(config_file, config)


def run_train(config_file, config):
    output_dir = config['OUTPUT_DIR']
    model_path = os.path.join(args.root, output_dir, 'model_final.pth')
    if not os.path.exists(model_path):
        base_cmd = 'python3 -m tools.train_net'  # 'python tools/train_net.py' or 'python3 -m tools.train_net'
        train_cmd = 'OMP_NUM_THREADS={} CUDA_VISIBLE_DEVICES={} {} ' \
                    '--dist-url auto --num-gpus {} --config-file {} --resume'.\
            format(args.num_threads, comma_sep(args.gpu_ids), base_cmd, len(args.gpu_ids), config_file)
        # TODO:
        #  --dist-url: just for obtaining a deterministic port to identify orphan processes
        #  --resume: ??? Using resume or not results in fine-tuning starting from iteration 1
        #  --opts: normally not necessary if we have set the config file appropriate
        run_cmd(train_cmd)


def run_test(config_file, config):
    output_dir = config['OUTPUT_DIR']
    res_path = os.path.join(args.root, output_dir, 'inference', 'res_final.json')
    if not os.path.exists(res_path):
        base_cmd = 'python3 -m tools.test_net'  # 'python tools/test_net.py' or 'python3 -m tools.test_net'
        test_cmd = 'OMP_NUM_THREADS={} CUDA_VISIBLE_DEVICES={} {} ' \
                   '--dist-url auto --num-gpus {} --config-file {} --resume --eval-only'. \
            format(args.num_threads, comma_sep(args.gpu_ids), base_cmd, len(args.gpu_ids), config_file)
        run_cmd(test_cmd)


def run_ckpt_surgery(dataset, class_split, src1, method, save_dir, src2=None, double_head=False):
    assert method in ['randinit', 'remove', 'combine'], 'Wrong method: {}'.format(method)
    if double_head:
        assert method == 'randinit', "Currently, double head is just supported together with 'combine' surgery!"
    src2_str = ''
    double_head_str = ' --double-head' if double_head else ''
    if method == 'combine':
        assert src2 is not None, 'Need a second source for surgery method \'combine\'!'
        src2_str = '--src2 {}'.format(src2)
    base_command = 'python3 -m tools.ckpt_surgery'  # 'python tools/ckpt_surgery.py' or 'python3 -m tools.ckpt_surgery'
    command = 'OMP_NUM_THREADS={} CUDA_VISIBLE_DEVICES={} {} ' \
              '--dataset {} --class-split {} --method {} --src1 {} --save-dir {} {} {}'\
        .format(args.num_threads, comma_sep(args.gpu_ids), base_command,
                dataset, class_split, method, src1, save_dir, src2_str, double_head_str)
    run_cmd(command)


# TODO: adjust id to different unfreeze strategies?
def get_training_id(layers, mode, shots, classifier, unfreeze=False, tfa=False, suffix=''):
    # A consistent string used
    #   - as directory name to save checkpoints
    #   - as name for configuration files
    pattern = 'faster_rcnn_R_{}_FPN_ft{}_{}{}{}{}{}'
    classifier_str = '_{}'.format(classifier)
    unfreeze_str = '_unfreeze' if unfreeze else ''
    tfa_str = '_TFA' if tfa else ''
    shot_str = '_{}shot'.format(shots)
    return pattern.format(layers, classifier_str, mode, shot_str, unfreeze_str, tfa_str, suffix)


def get_ft_dataset_names(dataset, class_split, mode, shot, seed, train_split='trainval', test_split='test'):
    # Note: For mode 'all' we evaluate on all classes and would, normally, not need the class split but since we allow
    #  for using different colors for base classes and novel classes, we need the class split to load the correct
    #  mapping of colors to classes
    return (
        '{}_{}_{}_{}_{}shot_seed{}'.format(dataset, class_split, train_split, mode, shot, seed),
        '{}_{}_{}_{}'.format(dataset, class_split, test_split, mode)
    )


# Returns fine-tuning configs. Assumes, that there already exist base-training configs!
# TODO: probably split get_config and doing a surgery? (e.g. if we just want wo iterate over multiple configs to do
#  automated inference?
def get_config(seed, shot, surgery_method, override_if_exists=False, rerun_surgery=False):
    """
    For a given seed and shot, generate a config file based on a template
    config file that is used for training/evaluation.
    You can extend/modify this function to fit your use-case.

    *****Presupposition*****
    Base-Training Checkpoint, stored e.g. at checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth

    *****Non-TFA Workflow*****
    ckpt_surgery
    --src1              checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth
    --method            randinit
    --save-dir          checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_all  -> model_reset_surgery.pth

    python3 -m tools.train_net
    --config <config_Kshot>
        -> weights =    checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth
        -> out-dir =    checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_all_Kshot -> model_final.pth

    *****TFA Workflow*****
    ckpt_surgery
    --src1              checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth
    --method            remove
    --save-dir          checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_novel -> model_reset_remove.pth

    python3 -m tools.train_net
    --config <config_Kshot>
        -> weights =    checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_novel/model_reset_remove.pth
        -> out-dir =    checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_novel_Kshot -> model_final.pth

    ckpt_surgery
    --src1              checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth
    --src2              checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_novel_Kshot/model_final.pth
    --method            combine
    --save-dir          checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_novel_Kshot_combine -> model_reset_combine.pth

    python3 -m tools.train_net
        --config <config_Kshot>
        -> weights =    checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_all/model_reset_combine.pth
        -> out-dir =    checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_all_Kshot_TFA -> model_final.pth

    Naming conventions:
    - Directory suffix for surgery checkpoints
        - '_all':                   Models to fine-tune entire classifier (=all classes): model_reset_surgery.pth
        - '_novel_Kshot_combine':   Models to fine-tune entire classifier (=all classes): model_reset_combine.pth
        - '_novel':                 Models to fine-tune novel classifier (=just novel classes):model_reset_remove.pth
    - Fine-Tuning directory suffix dependent on approach
        - '':       TFA(===Fine-tuning on 'model_reset_surgery.pth' weights)
        - '_TFA':   Non-TFA(===Fine-tuning on 'model_reset_combine.pth' weights)
    """
    assert surgery_method in ['randinit', 'remove', 'combine'], 'Wrong surgery method: {}'.format(surgery_method)
    if args.dataset == 'coco':  # only-coco configs
        # COCO
        # (max_iter, (<steps>), checkpoint_period)
        NOVEL_ITERS = {
            1: (500, (10000,), 500),
            2: (1500, (10000,), 500),
            3: (1500, (10000,), 500),
            5: (1500, (10000,), 500),
            10: (2000, (10000,), 500),
            30: (6000, (10000,), 500),
        }  # To fine-tune novel classifier
        ALL_ITERS = {
            1: (16000, (14400,), 1000),  # 1600
            2: (32000, (28800,), 1000),  # 3200
            3: (48000, (43200,), 2000),  # 4800
            5: (80000, (72000,), 4000),  # 8000
            10: (160000, (144000,), 10000),  # 16000
            30: (240000, (216000,), 12000),  # 24000
        }  # To fine-tune entire classifier
    elif args.dataset == 'isaid':  # only-isaid configs
        # iSAID
        # (max_iter, (<steps>), checkpoint_period)
        NOVEL_ITERS = {}  # no values yet set, need to examine the behaviour of novel fine-tuning on iSAID dataset first
        ALL_ITERS = {  # for now, we just support 10, 50 and 100 shot!
            10: (100000, (85000,), 10000),
            50: (100000, (85000,), 10000),
            100: (100000, (85000,), 10000)
        }
    elif args.dataset == 'voc':
        # PASCAL VOC
        # Note: we could as well support all types of surgery here, but we do not intend to use PASCAL VOC dataset!
        raise NotImplementedError("Fine-Tuning logic changed! Please refer to "
                                  "COCO-Workflow for reference.")
        assert not args.tfa, 'Only supports random weights for PASCAL now'
        ITERS = {
            1: (3500, 4000),
            2: (7000, 8000),
            3: (10500, 12000),
            5: (17500, 20000),
            10: (35000, 40000),
        }
        split = 'split{}'.format(args.split)
        mode = 'all{}'.format(args.split)
        temp_split = 'split1'
        temp_mode = 'all1'
        train_split = 'trainval'
        test_split = 'test'
        config_dir = 'configs/PascalVOC-detection'
        ckpt_dir = 'checkpoints/voc/faster_rcnn'
        base_cfg = '../../../Base-RCNN-FPN.yaml'
    else:
        raise ValueError("Dataset {} is not supported!".format(args.dataset))

    # Set some shared configs to save space
    if args.dataset in ['coco', 'isaid']:  # TODO: to ease adding a new dataset, probably invert query 'args.dataset != 'voc''
        if surgery_method == 'remove':  # fine-tuning only-novel classifier
            ITERS = NOVEL_ITERS
            mode = 'novel'
            # Note: it would normally be no problem to support fc or unfreeze in novel fine-tune but you would have to
            #  create a default config for those cases in order for being able to read example configs to modify
            assert args.classifier != 'fc' and not args.unfreeze
        else:  # either combine only-novel fine-tuning with base training or directly fine-tune entire classifier
            ITERS = ALL_ITERS
            mode = 'all'
        split = temp_split = ''
        temp_mode = mode
        train_split = cfg.TRAIN_SPLIT[args.dataset]
        test_split = cfg.TEST_SPLIT[args.dataset]
        config_dir = cfg.CONFIG_DIR_PATTERN[args.dataset].format(args.class_split)
        ckpt_dir = os.path.join(
            cfg.CKPT_DIR_PATTERN[args.dataset].format(args.class_split),
            'faster_rcnn'
        )
        base_cfg = '../../../../Base-RCNN-FPN.yaml'  # adjust depth to 'config_save_dir'

    # Needed to exchange seed and shot in the example config
    seed_str = 'seed{}'.format(seed)  # also used as a directory name
    shot_str = '{}shot'.format(shot)
    # Needed to create appropriate sub directories for the config files
    classifier_str = '_{}'.format(args.classifier)
    unfreeze_str = '_unfreeze' if args.unfreeze else ''
    # sub-directories 'ft_cosine', 'ft_cosine_unfreeze', 'ft_fc', 'ft_(only_)novel' to indicate the type of fine-tuning
    sub_dir_str = 'ft_only_novel' if surgery_method == 'remove' else 'ft' + classifier_str + unfreeze_str

    # Set paths depending on surgery method...
    base_ckpt = os.path.join(ckpt_dir, 'faster_rcnn_R_{}_FPN_base'.format(args.layers), 'model_final.pth')
    train_ckpt_base_dir = os.path.join(args.root, ckpt_dir, seed_str)
    os.makedirs(train_ckpt_base_dir, exist_ok=True)
    if surgery_method == 'randinit':
        surgery_ckpt_name = 'model_reset_surgery.pth'
        novel_ft_ckpt = None
        surgery_ckpt_save_dir = os.path.join(ckpt_dir, 'faster_rcnn_R_{}_FPN_all'.format(args.layers))
        training_identifier = get_training_id(layers=args.layers, mode=mode, shots=shot, classifier=args.classifier,
                                              unfreeze=args.unfreeze, tfa=False, suffix=args.suffix)
    elif surgery_method == 'remove':
        # Note: it would normally be no problem to support fc or unfreeze in novel fine-tune but you would have to
        #  create a default config for those cases in order for being able to read example configs to modify
        assert args.classifier != 'fc' and not args.unfreeze, 'Do not support fc or unfreeze in novel fine-tune!'
        surgery_ckpt_name = 'model_reset_remove.pth'
        novel_ft_ckpt = None
        surgery_ckpt_save_dir = os.path.join(ckpt_dir, 'faster_rcnn_R_{}_FPN_novel'.format(args.layers))
        # Note: we currently have args.tfa set, but we do not yet need it in our directory name
        training_identifier = get_training_id(layers=args.layers, mode=mode, shots=shot, classifier='cosine',
                                              unfreeze=False, tfa=False, suffix=args.suffix)
    else:
        assert surgery_method == 'combine', surgery_method
        surgery_ckpt_name = 'model_reset_combine.pth'
        # Note: novel_ft_ckpt has to match train_ckpt_save_dir of 'remove' surgery!
        # Note: we hard-code the mode to 'novel' because in this phase our actual mode is 'all' but we have to read the
        #  checkpoint of earlier novel fine-tuning whose mode was 'novel'
        novel_ft_ckpt = os.path.join(train_ckpt_base_dir,
                                     get_training_id(layers=args.layers, mode='novel', shots=shot, classifier='cosine',
                                                     unfreeze=False, tfa=False, suffix=args.suffix),
                                     'model_final.pth')
        assert os.path.exists(novel_ft_ckpt), 'Novel weights do not exist!'
        # Note: Here, we also need a seed string in the save directory for the surgery checkpoint, because since we
        #  combine a novel trained checkpoint (which has shot-data and therefore a certain seed), the shot and seed
        #  is imposed by this certain novel training!
        surgery_ckpt_save_dir = os.path.join(ckpt_dir,
                                             seed_str,
                                             'faster_rcnn_R_{}_FPN_novel_{}_combine'.format(args.layers, shot_str))
        training_identifier = get_training_id(layers=args.layers, mode=mode, shots=shot, classifier=args.classifier,
                                              unfreeze=args.unfreeze, tfa=True, suffix=args.suffix)

    train_weight = surgery_ckpt = os.path.join(surgery_ckpt_save_dir, surgery_ckpt_name)
    train_ckpt_save_dir = os.path.join(train_ckpt_base_dir, training_identifier)

    config_prefix = training_identifier

    # config save dir + save file name
    config_save_dir = os.path.join(args.root, config_dir, split, seed_str, sub_dir_str)
    os.makedirs(config_save_dir, exist_ok=True)
    # config_save_file = os.path.join(config_save_dir, prefix + '.yaml')
    config_save_file = os.path.join(config_save_dir, config_prefix + '.yaml')

    if not os.path.exists(surgery_ckpt) or rerun_surgery:
        # surgery model does not exist, so we have to do a surgery!
        run_ckpt_surgery(dataset=args.dataset, class_split=args.class_split, method=surgery_method,
                         src1=base_ckpt, src2=novel_ft_ckpt, save_dir=surgery_ckpt_save_dir,
                         double_head=args.double_head)
        assert os.path.exists(surgery_ckpt)
        print("Finished creating surgery checkpoint {}".format(surgery_ckpt))
    else:
        print("Using already existent surgery checkpoint {}".format(surgery_ckpt))

    if os.path.exists(config_save_file) and not override_if_exists:
        # If the requested config already exists and we do not want to override it, make sure that the necessary
        #  surgery checkpoints exist and return it
        print("Config already exists, returning the existent config...")
        return config_save_file, load_yaml_file(config_save_file)
    print("Creating a new config file: {}".format(config_save_file))
    # Set all values in the empty config
    new_config = get_empty_ft_config()  # get an empty config and fill it appropriately
    # read out some configs from the config files
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    new_config['_BASE_'] = base_cfg
    # Try to read configs of the base config which will override the default configs
    base_config = load_yaml_file(os.path.join(config_save_dir, base_cfg))
    if 'MODEL' in base_config and 'ROI_BOX_HEAD' in base_config['MODEL']:
        roi_box_config = base_config['MODEL']['ROI_BOX_HEAD']
        if 'NUM_CONV' in roi_box_config:
            num_conv = roi_box_config['NUM_CONV']
        if 'NUM_FC' in roi_box_config:
            num_fc = roi_box_config['NUM_FC']

    if args.dataset == 'voc':
        new_config['MODEL']['WEIGHTS'] = new_config['MODEL']['WEIGHTS'].replace('base1', 'base{}'.format(args.split))
        for dset in ['TRAIN', 'TEST']:
            new_config['DATASETS'][dset] = (
                new_config['DATASETS'][dset][0].replace(temp_mode, 'all' + str(args.split))
                ,)

    new_config['MODEL']['WEIGHTS'] = train_weight

    new_config['MODEL']['RESNETS']['DEPTH'] = args.layers
    new_config['MODEL']['ANCHOR_GENERATOR']['SIZES'] = str([[32], [64], [128], [256], [512]])
    new_config['MODEL']['RPN']['PRE_NMS_TOPK_TRAIN'] = 2000  # Per FPN level. TODO: per batch or image?
    new_config['MODEL']['RPN']['PRE_NMS_TOPK_TEST'] = 1000  # Per FPN level. TODO: per batch or image?
    new_config['MODEL']['RPN']['POST_NMS_TOPK_TRAIN'] = 1000  # TODO: per batch or image?
    new_config['MODEL']['RPN']['POST_NMS_TOPK_TEST'] = 1000  # TODO: per batch or image?
    new_config['MODEL']['ROI_HEADS']['NAME'] = 'StandardROIHeads' if not args.double_head else 'StandardROIDoubleHeads'
    num_base_classes = len(CLASS_SPLITS[args.dataset][args.class_split]['base'])
    num_novel_classes = len(CLASS_SPLITS[args.dataset][args.class_split]['novel'])
    num_all_classes = num_base_classes + num_novel_classes
    new_config['MODEL']['ROI_HEADS']['NUM_CLASSES'] = \
        num_novel_classes if surgery_method == 'remove' else num_all_classes
    if args.double_head:
        new_config['MODEL']['ROI_HEADS']['MULTIHEAD_NUM_CLASSES'] = str([num_base_classes, num_novel_classes])
    else:
        del new_config['MODEL']['ROI_HEADS']['MULTIHEAD_NUM_CLASSES']
    new_config['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST'] = 0.05
    new_config['MODEL']['ROI_BOX_HEAD']['NAME'] = 'FastRCNNConvFCHead' if not args.double_head else 'FastRCNNConvFCMultiHead'
    if args.double_head:
        new_config['MODEL']['ROI_BOX_HEAD']['NUM_HEADS'] = 2
        new_config['MODEL']['ROI_BOX_HEAD']['SPLIT_AT_FC'] = 2
    else:
        del new_config['MODEL']['ROI_BOX_HEAD']['NUM_HEADS']
        del new_config['MODEL']['ROI_BOX_HEAD']['SPLIT_AT_FC']
    all_convs = range(1, num_conv + 1)
    unfreeze_convs = all_convs if args.unfreeze else args.unfreeze_roi_box_head_convs
    new_config['MODEL']['ROI_BOX_HEAD']['FREEZE_CONVS'] = str([i for i in all_convs if i not in unfreeze_convs])
    all_fcs = range(1, num_fc + 1)
    unfreeze_fcs = all_fcs if args.unfreeze else args.unfreeze_roi_box_head_fcs
    new_config['MODEL']['ROI_BOX_HEAD']['FREEZE_FCS'] = str([i for i in all_fcs if i not in unfreeze_fcs])
    new_config['MODEL']['BACKBONE']['FREEZE'] = not (args.unfreeze or args.unfreeze_backbone)
    new_config['MODEL']['PROPOSAL_GENERATOR']['FREEZE'] = not (args.unfreeze or args.unfreeze_proposal_generator)
    (train_data, test_data) = get_ft_dataset_names(args.dataset, args.class_split, mode, shot, seed,
                                                   train_split, test_split)
    new_config['DATASETS']['TRAIN'] = str((train_data,))
    new_config['DATASETS']['TEST'] = str((test_data,))
    new_config['SOLVER']['IMS_PER_BATCH'] = args.bs  # default: 16
    lr_scale_factor = args.bs / 16
    new_config['SOLVER']['BASE_LR'] = args.lr if args.lr != -1 else 0.001 * lr_scale_factor
    if len(args.lr_decay_steps) == 1 and args.lr_decay_steps[0] == -1:
        lr_decay_steps = ITERS[shot][1]
    else:
        lr_decay_steps = tuple(args.lr_decay_steps)
    max_iter = ITERS[shot][0] if args.max_iter == -1 else args.max_iter
    ckpt_interval = ITERS[shot][2] if args.ckpt_interval == -1 else args.ckpt_interval
    new_config['SOLVER']['STEPS'] = str(lr_decay_steps)
    new_config['SOLVER']['MAX_ITER'] = max_iter
    new_config['SOLVER']['CHECKPOINT_PERIOD'] = ckpt_interval  # ITERS[shot][0] // args.ckpt_freq
    new_config['SOLVER']['WARMUP_ITERS'] = 0 if args.unfreeze or surgery_method == 'remove' else 10  # TODO: ???
    new_config['INPUT']['MIN_SIZE_TRAIN'] = str((640, 672, 704, 736, 768, 800))  # scales for multi-scale training
    new_config['OUTPUT_DIR'] = train_ckpt_save_dir

    if args.dataset == 'coco':
        new_config['MODEL']['ANCHOR_GENERATOR']['SIZES'] = str([[32], [64], [128], [256], [512]])
        new_config['TEST']['DETECTIONS_PER_IMAGE'] = 100
        new_config['INPUT']['MIN_SIZE_TRAIN'] = str((640, 672, 704, 736, 768, 800))
    elif args.dataset == 'isaid':
        new_config['MODEL']['ANCHOR_GENERATOR']['SIZES'] = str([[16], [32], [64], [128], [256]])
        new_config['MODEL']['RPN']['PRE_NMS_TOPK_TRAIN'] = 3000
        new_config['MODEL']['RPN']['POST_NMS_TOPK_TRAIN'] = 1500
        new_config['MODEL']['RPN']['PRE_NMS_TOPK_TEST'] = 1000
        new_config['MODEL']['RPN']['POST_NMS_TOPK_TEST'] = 1000
        new_config['TEST']['DETECTIONS_PER_IMAGE'] = 100
        new_config['INPUT']['MIN_SIZE_TRAIN'] = str((600, 700, 800, 900, 1000))  # (608, 672, 736, 800, 864, 928, 992)

    with open(config_save_file, 'w') as fp:
        yaml.dump(new_config, fp, sort_keys=False)  # TODO: 'sort_keys=False' requires pyyaml >= 5.1

    return config_save_file, new_config


def comma_sep(elements):
    res = ''
    if not isinstance(elements, (list, tuple)):
        return str(elements)
    assert len(elements) > 0, "need at least one element in the collection {}!".format(elements)
    if len(elements) == 1:
        return str(elements[0])
    for element in elements:
        res += '{},'.format(str(element))
    return res[:-1]  # remove trailing space


def main(args):
    if args.explicit_seeds:
        seeds = args.seeds
    else:
        if len(args.seeds) == 1:
            seeds = [args.seeds[0]]
        else:
            assert len(args.seeds) == 2
            seeds = range(args.seeds[0], args.seeds[1] + 1)
    for shot in args.shots:
        for seed in seeds:
            print('Split: {}, Seed: {}, Shot: {}'.format(args.split, seed, shot))
            if args.tfa:
                config_file, config = get_config(seed, shot, surgery_method='remove',
                                                 override_if_exists=args.override_config,
                                                 rerun_surgery=args.override_surgery)
                run_exp(config_file, config)  # TODO: probably just run train(config_file, config) because evaluation on novel fine-tune might be unnecessary!
                config_file, config = get_config(seed, shot, surgery_method='combine',
                                                 override_if_exists=args.override_config,
                                                 rerun_surgery=args.override_surgery)
                run_exp(config_file, config)
            else:
                config_file, config = get_config(seed, shot, surgery_method='randinit',
                                                 override_if_exists=args.override_config,
                                                 rerun_surgery=args.override_surgery)
                run_exp(config_file, config)


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    main(args)
