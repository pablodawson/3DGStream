#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.extent = 0
        self.sh_degree = 1
        self._source_path = ""
        self._model_path = ""
        self._output_path = ""
        self._video_path = ""
        self.ply_name = "points3D.ply"
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True

        self.min_bounds =[-7.0, -8.0, -9.0]
        self.max_bounds =[22.0, 6.0, 17.0]

        # SOGS
        self.sorting_enabled = True
        self.sorting_normalize = True
        self.xyz_weight = 1.0
        self.features_dc_weight = 1.0
        self.features_rest_weight = 0.0
        self.opacity_weight = 0.0
        self.scaling_weight = 1.0
        self.rotation_weight = 0.0
        self.shuffle_sort = False
        self.improvement_break = 0.0001

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.bwd_depth = False
        self.opt_type='3DGStream'
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.iterations_s2 = 0
        self.first_load_iteration = -1
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.depth_smooth = 0.0
        self.ntc_lr = None
        self.lambda_dxyz = 0.0
        self.lambda_drot= 0.0
        self.densification_interval = 300
        self.opacity_reset_interval = 100000
        self.densify_from_iter = 500
        self.densify_until_iter = 4000
        self.densify_grad_threshold = 0.0002
        self.ntc_conf_path = "configs/cache/cache_F_8.json"
        self.ntc_path = ""
        self.batch_size = 1
        self.spawn_type = "spawn"
        self.s2_type = "split"
        self.s2_adding = False
        self.num_of_split=1
        self.num_of_spawn=2
        self.std_scale=1
        self.min_opacity = 0.01
        self.rotate_sh = True
        self.only_mlp = False
        

        # SOGS
        self.disable_xyz_log_activation = True
        self.lambda_neighbor = 1.0
        self.neighbor_loss_activated = False
        self.xyz_neighbor_weight = 0.0
        self.features_dc_neighbor_weight = 0.0
        self.opacity_neighbor_weight = 1.0
        self.scaling_neighbor_weight = 0.0
        self.rotation_neighbor_weight = 10.0
        self.neighbor_normalize = False
        self.neighbor_blur_kernel_size = 5
        self.neighbor_blur_sigma = 3.0
        self.neighbor_loss_fn = "huber"

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
