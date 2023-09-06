import numpy as np
import os
import shutil
import argparse
import _init_paths
from lib.test.evaluation.environment import env_settings

# 对trackingnet的结果进行转换，转换成trackingnet的提交格式
def transform_trackingnet(tracker_name, cfg_name):
    env = env_settings()
    result_dir = env.results_path
    # src_dir是跟踪结果的存放路径，dest_dir是转换后的结果存放路径
    src_dir = os.path.join(result_dir, "%s/%s/trackingnet/" % (tracker_name, cfg_name))
    print(src_dir)
    dest_dir = os.path.join(result_dir, "%s/%s/trackingnet_submit/" % (tracker_name, cfg_name))
    print(dest_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    items = os.listdir(src_dir) # 列出文件夹下所有的目录与文件
    for item in items:
        if "all" in item:
            continue
        if "time" not in item:
            src_path = os.path.join(src_dir, item)  # 原始的跟踪结果
            dest_path = os.path.join(dest_dir, item)    # 转换后的跟踪结果
            bbox_arr = np.loadtxt(src_path, dtype=np.int, delimiter='\t')   # 读取原始的跟踪结果
            np.savetxt(dest_path, bbox_arr, fmt='%d', delimiter=',')    # 保存转换后的跟踪结果
    # make zip archive
    shutil.make_archive(src_dir, "zip", src_dir)    # 压缩src_dir目录为zip文件
    shutil.make_archive(dest_dir, "zip", dest_dir)  # 压缩dest_dir目录为zip文件
    # Remove the original files
    shutil.rmtree(src_dir)  # 删除src_dir目录
    shutil.rmtree(dest_dir) # 删除dest_dir目录


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transform trackingnet results.')
    parser.add_argument('--tracker_name', default='ostrack', type=str, help='Name of tracking method.')
    parser.add_argument('--cfg_name', default='vitb_256_mae_ce_32x4_ep300', type=str, help='Name of config file.')

    args = parser.parse_args()
    transform_trackingnet(args.tracker_name, args.cfg_name)
