import os
from collections import OrderedDict
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print('WARNING: You are using tensorboardX instead sis you have a too old pytorch version.')
    from tensorboardX import SummaryWriter


class TensorboardWriter:
    def __init__(self, directory, loader_names):
        self.directory = directory
        self.writer = OrderedDict({name: SummaryWriter(os.path.join(self.directory, name)) for name in loader_names})

    def write_info(self, script_name, description):
        tb_info_writer = SummaryWriter(os.path.join(self.directory, 'info'))
        tb_info_writer.add_text('Script_name', script_name) # 使用add_text添加文本数据，第一个参数是文本的名称，第二个参数是文本的内容
        tb_info_writer.add_text('Description', description) # 使用add_text添加文本数据，第一个参数是文本的名称，第二个参数是文本的内容
        tb_info_writer.close()

    def write_epoch(self, stats: OrderedDict, epoch: int, ind=-1):
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue
            for var_name, val in loader_stats.items():
                if hasattr(val, 'history') and getattr(val, 'has_new_data', True):
                    self.writer[loader_name].add_scalar(var_name, val.history[ind], epoch)  # 使用add_scalar添加标量数据，第一个参数是标量的名称，第二个参数是标量的值，第三个参数是当前的epoch