from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/UserDirectory/gly/code/OSTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/Newdisk/Datasets/GOT10K/data/full_data'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/UserDirectory/gly/code/OSTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/UserDirectory/gly/code/OSTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/UserDirectory/gly/code/OSTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/Newdisk/Datasets/LaSOT/LaSOTBenchmark'
    settings.network_path = '/home/UserDirectory/gly/code/OSTrack/pretrained_networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/UserDirectory/gly/code/OSTrack/data/nfs'
    settings.otb_path = '/home/Newdisk/New_dataset/Datasets/OTB100/'
    settings.prj_dir = '/home/UserDirectory/gly/code/OSTrack'
    settings.result_plot_path = '/home/UserDirectory/gly/code/OSTrack/output/test/result_plots'
    settings.results_path = '/home/UserDirectory/gly/code/OSTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/UserDirectory/gly/code/OSTrack/output'
    settings.segmentation_path = '/home/UserDirectory/gly/code/OSTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/UserDirectory/gly/code/OSTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/UserDirectory/gly/code/OSTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/Newdisk/Datasets/TrackingNet'
    settings.uav_path = '/home/Newdisk/New_dataset/Datasets/UAV123'
    settings.vot18_path = '/home/UserDirectory/gly/code/OSTrack/data/vot2018'
    settings.vot22_path = '/home/UserDirectory/gly/code/OSTrack/data/vot2022'
    settings.vot_path = '/home/UserDirectory/gly/code/OSTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

