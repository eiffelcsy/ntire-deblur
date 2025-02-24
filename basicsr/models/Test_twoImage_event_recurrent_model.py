import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from einops import rearrange

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module("basicsr.models.losses")
metric_module = importlib.import_module("basicsr.metrics")


class TestTwoImageEventRecurrentRestorationModel(BaseModel):
    """Adapted model that takes data items like:
    {
      'frame': <blurry tensor>,
      'frame_gt': <ground truth tensor>,
      'voxel': <event voxel tensor>,
      'image_name': <string name for saving/output>
    }
    """

    def __init__(self, opt):
        super(TestTwoImageEventRecurrentRestorationModel, self).__init__(opt)
        # define network
        self.net_g = define_network(deepcopy(opt["network_g"]))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key=self.opt["path"].get("param_key", "params"),
            )

    def feed_data(self, data):
        """Receives a dict with 'frame', 'frame_gt', 'voxel', 'image_name'."""
        # Use 'frame' as your main input
        self.lq = data["frame"].to(self.device)

        # Event voxel
        self.voxel = data["voxel"].to(self.device)

        # This string identifies the sample (used for naming output files)
        # If your dataset returns a single string, just store it directly:
        if isinstance(data["image_name"], list):
            # In case 'image_name' is a list with single entry
            self.seq_name = data["image_name"][0]
        else:
            self.seq_name = data["image_name"]

        # If there's ground-truth
        if "frame_gt" in data:
            self.gt = data["frame_gt"].to(self.device)

    def transpose(self, t, trans_idx):
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def grids_voxel(self):
        b, c, h, w = self.voxel.size()
        self.original_size_voxel = self.voxel.size()
        assert b == 1
        crop_size = self.opt["val"].get("crop_size")
        import math

        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1
        step_j = (
            crop_size
            if num_col == 1
            else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        )
        step_i = (
            crop_size
            if num_row == 1
            else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)
        )

        parts = []
        idxes = []
        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                for trans_idx in range(self.opt["val"].get("trans_num", 1)):
                    parts.append(
                        self.transpose(
                            self.voxel[:, :, i : i + crop_size, j : j + crop_size],
                            trans_idx,
                        )
                    )
                    idxes.append({"i": i, "j": j, "trans_idx": trans_idx})
                j = j + step_j
            i = i + step_i

        if self.opt["val"].get("random_crop_num", 0) > 0:
            import random

            for _ in range(self.opt["val"].get("random_crop_num")):
                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                trans_idx = random.randint(0, self.opt["val"].get("trans_num", 1) - 1)
                parts.append(
                    self.transpose(
                        self.voxel[:, :, i : i + crop_size, j : j + crop_size],
                        trans_idx,
                    )
                )
                idxes.append({"i": i, "j": j, "trans_idx": trans_idx})

        self.origin_voxel = self.voxel
        self.voxel = torch.cat(parts, dim=0)
        print("----------parts voxel .. ", len(parts), self.voxel.size())
        self.idxes = idxes

    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt["val"].get("crop_size")
        import math

        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1
        step_j = (
            crop_size
            if num_col == 1
            else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        )
        step_i = (
            crop_size
            if num_row == 1
            else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)
        )

        parts = []
        idxes = []
        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                for trans_idx in range(self.opt["val"].get("trans_num", 1)):
                    parts.append(
                        self.transpose(
                            self.lq[:, :, i : i + crop_size, j : j + crop_size],
                            trans_idx,
                        )
                    )
                    idxes.append({"i": i, "j": j, "trans_idx": trans_idx})
                j = j + step_j
            i = i + step_i

        if self.opt["val"].get("random_crop_num", 0) > 0:
            import random

            for _ in range(self.opt["val"].get("random_crop_num")):
                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                trans_idx = random.randint(0, self.opt["val"].get("trans_num", 1) - 1)
                parts.append(
                    self.transpose(
                        self.lq[:, :, i : i + crop_size, j : j + crop_size], trans_idx
                    )
                )
                idxes.append({"i": i, "j": j, "trans_idx": trans_idx})

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size
        print("...", self.device)
        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt["val"].get("crop_size")

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx["i"]
            j = each_idx["j"]
            trans_idx = each_idx["trans_idx"]
            patch = self.transpose_inverse(
                self.output[cnt, :, :, :].unsqueeze(0), trans_idx
            ).squeeze(0)
            preds[0, :, i : i + crop_size, j : j + crop_size] += patch
            count_mt[0, 0, i : i + crop_size, j : j + crop_size] += 1.0

        self.output = preds / count_mt
        self.lq = self.origin_lq
        self.voxel = self.origin_voxel

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt["val"].get("max_minibatch", n)
            i = 0
            while i < n:
                j = min(i + m, n)
                pred = self.net_g(
                    x=self.lq[i:j, :, :, :], event=self.voxel[i:j, :, :, :]
                )
                outs.append(pred)
                i = j
            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        # not used as frequently, but you can adapt it similarly
        self.feed_data({"frame": img.unsqueeze(dim=0), "voxel": voxel.unsqueeze(dim=0)})
        if self.opt["val"].get("grids") is not None:
            self.grids()
            self.grids_voxel()
        self.test()
        if self.opt["val"].get("grids") is not None:
            self.grids_inverse()
        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals["result"]])
        imwrite(sr_img, save_path)

    def dist_validation(
        self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image
    ):
        logger = get_root_logger()
        import os

        if os.environ.get("LOCAL_RANK", "0") == "0":
            return self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image
            )
        else:
            return 0.0

    def nondist_validation(
        self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image
    ):
        dataset_name = self.opt.get("name")
        save_gt = self.opt["val"].get("save_gt", False)
        self.m = self.opt["datasets"]["test"].get("num_end_interpolation", 0)
        self.n = self.opt["datasets"]["test"].get("num_inter_interpolation", 0)
        imgs_per_iter_deblur = 2 * self.m
        imgs_per_iter_interpo = self.n

        with_metrics = self.opt["val"].get("metrics_deblur") is not None
        if with_metrics:
            self.metric_results_deblur = {
                metric: 0.0 for metric in self.opt["val"]["metrics_deblur"].keys()
            }
            self.metric_results_interpo = {
                metric: 0.0 for metric in self.opt["val"]["metrics_interpo"].keys()
            }
            self.metric_results_total = {
                metric: 0.0 for metric in self.metric_results_deblur.keys()
            }

        pbar = tqdm(total=len(dataloader), unit="image")
        cnt = 0
        last_seq_name = "Unknown"
        seq_inner_cnt = 0

        for idx, val_data in enumerate(dataloader):
            # data: {'frame': tensor, 'frame_gt': tensor, 'voxel': tensor, 'image_name': str}
            self.feed_data(val_data)

            # check whether we changed sequence
            if self.seq_name == last_seq_name:
                seq_inner_cnt += 1
            else:
                seq_inner_cnt = 0
                last_seq_name = self.seq_name

            img_name = f"{seq_inner_cnt:04d}"

            # optional tiling
            if self.opt["val"].get("grids") is not None:
                self.grids()
                self.grids_voxel()

            self.test()

            if self.opt["val"].get("grids") is not None:
                self.grids_inverse()

            visuals = self.get_current_visuals()

            # free memory
            del self.lq
            del val_data["frame"]
            del self.voxel
            del val_data["voxel"]
            del self.output
            if "gt" in visuals:
                del self.gt
                del val_data["frame_gt"]
            torch.cuda.empty_cache()

            # 'visuals["result"]' is shape [b, t, c, h, w]
            imgs_per_iter = visuals["result"].size(1)

            for frame_idx in range(imgs_per_iter):
                # rename files: "<seq_name>_<frame_idx>.png"
                file_name = f"{self.seq_name}_{frame_idx:02d}"
                result = visuals["result"][0, frame_idx, :, :, :]
                sr_img = tensor2img([result])  # uint8, BGR
                if "gt" in visuals:
                    gt = visuals["gt"][0, frame_idx, :, :, :]
                    gt_img = tensor2img([gt])  # uint8, BGR

                if save_img:
                    if self.opt["is_train"]:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            self.seq_name,
                            f"{file_name}.png",
                        )
                        save_gt_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            self.seq_name,
                            f"{file_name}_gt.png",
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            self.seq_name,
                            f"{file_name}.png",
                        )
                        save_gt_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            self.seq_name,
                            f"{file_name}_gt.png",
                        )

                    imwrite(sr_img, save_img_path)
                    if save_gt:
                        imwrite(gt_img, save_gt_img_path)

                if with_metrics:
                    # calculate metrics
                    opt_metric = deepcopy(self.opt["val"]["metrics"])
                    if use_image:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop("type")
                            self.metric_results[name] += getattr(
                                metric_module, metric_type
                            )(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop("type")
                        self.metric_results[name] += getattr(
                            metric_module, metric_type
                        )(visuals["result"], visuals["gt"], **opt_)

            pbar.update(1)
            pbar.set_description(f"Test {file_name}")
            cnt += 1

        pbar.close()

        current_metric = 0.0
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name},\t"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
