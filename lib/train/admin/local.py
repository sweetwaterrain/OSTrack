class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/UserDirectory/gly/code/OSTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/UserDirectory/gly/code/OSTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/UserDirectory/gly/code/OSTrack/pretrained_networks'
        self.lasot_dir = '/home/Newdisk/Datasets/LaSOT/LaSOTBenchmark/'
        self.got10k_dir = '/home/Newdisk/Datasets/GOT10K/data/full_data/train'
        self.got10k_val_dir = '/home/Newdisk/Datasets/GOT10K/data/full_data/val'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = '/home/Newdisk/Datasets/TrackingNet/'
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = '/home/Newdisk/Datasets/coco2017/'
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
