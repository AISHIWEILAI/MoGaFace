import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import lpips
from piq import ssim, FID#, multi_scale_ssim, brisque
from piq.feature_extractors import InceptionV3
# from basicsr.metrics import niqe
# import cpbd

FID_batch_size = 1024

class FIDMeter:
    def __init__(self, device="cuda", feature_extractor=InceptionV3):
        self.feature_extractor = feature_extractor().to(device)
        self.device = device
        self.fid_metric = FID()
        self.preds_feats = None
        self.truths_feats = None

    def clear(self):
        self.preds_feats = None
        self.truths_feats = None

    def prepare_inputs(self, preds, truths):
        if preds is None:
            preds_num = 0
            truths_num = truths.size(0)
            total_images = truths
        elif truths is None:
            preds_num = preds.size(0)
            truths_num = 0
            total_images = preds
        else:
            preds_num = preds.size(0)
            truths_num = truths.size(0)
            total_images = torch.cat((preds, truths), 0)
        if total_images.size(0) > FID_batch_size:
            total_images = torch.split(total_images, FID_batch_size, dim=0)
        else:
            total_images = [total_images]
        total_feats = []
        for sub_images in total_images:
            sub_images = sub_images.to(self.device)
            feats = self.fid_metric.compute_feats([{'images': sub_images}], feature_extractor=self.feature_extractor)
            feats = feats.detach()
            total_feats.append(feats)
        total_feats = torch.cat(total_feats, 0)
        preds_feats, truths_feats = torch.split(total_feats, [preds_num, truths_num], dim=0)
        return preds_feats, truths_feats

    def update(self, preds, truths):
        preds_feats, truths_feats = self.prepare_inputs(preds, truths)
        preds_feats = preds_feats.cuda()
        truths_feats = truths_feats.cuda()
        if self.preds_feats is None:
            self.preds_feats = preds_feats
            self.truths_feats = truths_feats
        else:
            if preds is not None:
                self.preds_feats = torch.cat((self.preds_feats, preds_feats), 0)
            if preds is not None:
                self.truths_feats = torch.cat((self.truths_feats, truths_feats), 0)

    def measure(self):
        self.preds_feats = self.preds_feats.to(self.device)
        self.truths_feats = self.truths_feats.to(self.device)
        fid = self.fid_metric.compute_metric(self.preds_feats, self.truths_feats).item()
        return fid

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "FID"), self.measure(), global_step)

    def report(self):
        return self.measure()#f'FID = {self.measure():.6f}'

# class SSIMMeter:
#     def __init__(self):
#         self.ssim_values = []
#
#     def clear(self):
#         self.ssim_values = []
#
#     def update(self, preds, truths):
#         preds = preds.permute(0, 3, 1, 2)
#         truths = truths.permute(0, 3, 1, 2)
#         ssim_values = ssim(preds, truths, data_range=1., reduction='none')
#         self.ssim_values.extend([e.item() for e in ssim_values])
#
#     def measure(self):
#         return np.asarray(self.ssim_values).mean()
#
#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)
#
#     def report(self):
#         return f'SSIM = {self.measure():.6f}'
#
#
# class MS_SSIMMeter:
#     def __init__(self):
#         self.ms_ssim_values = []
#
#     def clear(self):
#         self.ms_ssim_values = []
#
#     def update(self, preds, truths):
#         preds = preds.permute(0, 3, 1, 2)
#         truths = truths.permute(0, 3, 1, 2)
#         ms_ssim_values = multi_scale_ssim(preds, truths, data_range=1., reduction='none')
#         self.ms_ssim_values.extend([e.item() for e in ms_ssim_values])
#
#     def measure(self):
#         return np.asarray(self.ms_ssim_values).mean()
#
#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, "MS_SSIM"), self.measure(), global_step)
#
#     def report(self):
#         return f'MS_SSIM = {self.measure():.6f}'
#

# class NIQEMeter:
#     def __init__(self):
#         self.niqe_values = []
#
#     def clear(self):
#         self.niqe_values = []
#
#     def prepare_inputs(self, inputs):
#         outputs = []
#         for i, inp in enumerate(inputs):
#             if torch.is_tensor(inp):
#                 inp = inp * 255
#                 inp = inp.round().to(torch.int)
#                 inp = inp.detach().cpu().numpy()
#                 inp = inp.astype(np.uint8)
#                 inp = cv2.cvtColor(inp, cv2.COLOR_RGB2BGR)
#             outputs.append(inp)
#
#         return outputs
#
#     def update(self, preds, truths):
#         if preds is None:
#             return
#         preds = self.prepare_inputs(preds)  # [B, N, 3] or [B, H, W, 3], range in [0, 255]
#         self.niqe_values.extend([niqe.calculate_niqe(pred, crop_border=0) for pred in preds])
#
#     def measure(self):
#         return np.asarray(self.niqe_values).mean()
#
#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, "NIQE"), self.measure(), global_step)
#
#     def report(self):
#         return f'NIQE = {self.measure():.6f}'

# class BRISQUEMeter:
#     def __init__(self):
#         self.brisque_values = []
#
#     def clear(self):
#         self.brisque_values = []
#
#     def update(self, preds, truths):
#         if preds is None:
#             return
#         preds = preds.permute(0, 3, 1, 2)
#         brisque_values = brisque(preds, data_range=1., reduction='none')
#         self.brisque_values.extend([e.item() for e in brisque_values])
#
#     def measure(self):
#         return np.asarray(self.brisque_values).mean()
#
#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, "BRISQUE"), self.measure(), global_step)
#
#     def report(self):
#         return f'BRISQUE = {self.measure():.6f}'
#
#
# class CPBDMeter:
#     def __init__(self):
#         self.cpbd_values = []
#
#     def clear(self):
#         self.cpbd_values = []
#
#     def prepare_inputs(self, inputs):
#         outputs = []
#         for i, inp in enumerate(inputs):
#             if torch.is_tensor(inp):
#                 inp = inp * 255
#                 inp = inp.round().to(torch.int)
#                 inp = inp.detach().cpu().numpy()
#                 inp = inp.astype(np.uint8)
#                 inp = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)
#             outputs.append(inp)
#
#         return outputs
#
#     def update(self, preds, truths):
#         if preds is None:
#             return
#         preds = self.prepare_inputs(preds)  # [B, N, 3] or [B, H, W, 3], range in [0, 255]
#         self.cpbd_values.extend([cpbd.compute(pred) for pred in preds])
#
#     def measure(self):
#         return np.asarray(self.cpbd_values).mean()
#
#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, "CPBD"), self.measure(), global_step)
#
#     def report(self):
#         return f'CPBD = {self.measure():.6f}'
#
#
# class PSNRMeter:
#     def __init__(self):
#         self.V = 0
#         self.N = 0
#
#     def clear(self):
#         self.V = 0
#         self.N = 0
#
#     def prepare_inputs(self, *inputs):
#         outputs = []
#         for i, inp in enumerate(inputs):
#             if torch.is_tensor(inp):
#                 inp = inp.detach().cpu().numpy()
#             outputs.append(inp)
#
#         return outputs
#
#     def update(self, preds, truths):
#         preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range in [0, 1]
#
#         # simplified since max_pixel_value is 1 here.
#         psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
#
#         self.V += psnr
#         self.N += 1
#
#     def measure(self):
#         return self.V / self.N
#
#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)
#
#     def report(self):
#         return f'PSNR = {self.measure():.6f}'
#
#
# class LPIPSMeter:
#     def __init__(self, net='alex', device=None):
#         self.V = 0
#         self.N = 0
#         self.net = net
#
#         self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.fn = lpips.LPIPS(net=net).eval().to(self.device)
#
#     def clear(self):
#         self.V = 0
#         self.N = 0
#
#     def prepare_inputs(self, *inputs):
#         outputs = []
#         for i, inp in enumerate(inputs):
#             inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
#             inp = inp.to(self.device)
#             outputs.append(inp)
#         return outputs
#
#     def update(self, preds, truths):
#         preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
#         v = self.fn(truths, preds, normalize=True).item()  # normalize=True: [0, 1] to [-1, 1]
#         self.V += v
#         self.N += 1
#
#     def measure(self):
#         return self.V / self.N
#
#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)
#
#     def report(self):
#         return f'LPIPS ({self.net}) = {self.measure():.6f}'
#
#
# class LMDMeter:
#     def __init__(self, backend='dlib', region='mouth', device=None):
#         self.backend = backend
#         self.region = region  # mouth or face
#         if device is None:
#             device = "cpu"
#         else:
#             device = str(device)
#         if self.backend == 'dlib':
#             import dlib
#
#             # load checkpoint manually
#             self.predictor_path = './shape_predictor_68_face_landmarks.dat'
#             if not os.path.exists(self.predictor_path):
#                 raise FileNotFoundError(
#                     'Please download dlib checkpoint from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
#
#             self.detector = dlib.get_frontal_face_detector()
#             self.predictor = dlib.shape_predictor(self.predictor_path)
#
#         else:
#
#             import face_alignment
#             try:
#                 self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#             except:
#                 self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False,
#                                                               device=device)
#
#         self.V = 0
#         self.N = 0
#
#     def get_landmarks(self, img):
#
#         if self.backend == 'dlib':
#             dets = self.detector(img, 1)
#             for det in dets:
#                 shape = self.predictor(img, det)
#                 # ref: https://github.com/PyImageSearch/imutils/blob/c12f15391fcc945d0d644b85194b8c044a392e0a/imutils/face_utils/helpers.py
#                 lms = np.zeros((68, 2), dtype=np.int32)
#                 for i in range(0, 68):
#                     lms[i, 0] = shape.part(i).x
#                     lms[i, 1] = shape.part(i).y
#                 break
#
#         else:
#             lms = self.predictor.get_landmarks(img)[-1]
#
#         # self.vis_landmarks(img, lms)
#         lms = lms.astype(np.float32)
#
#         return lms
#
#     def vis_landmarks(self, img, lms):
#         plt.imshow(img)
#         plt.plot(lms[48:68, 0], lms[48:68, 1], marker='o', markersize=1, linestyle='-', lw=2)
#         plt.show()
#
#     def clear(self):
#         self.V = 0
#         self.N = 0
#
#     def prepare_inputs(self, *inputs):
#         outputs = []
#         for i, inp in enumerate(inputs):
#             inp = inp.detach().cpu().numpy()
#             inp = (inp * 255).astype(np.uint8)
#             outputs.append(inp)
#         return outputs
#
#     def update(self, preds, truths):
#         # assert B == 1
#         preds, truths = self.prepare_inputs(preds[0], truths[0])  # [H, W, 3] numpy array
#
#         # get lms
#         lms_pred = self.get_landmarks(preds)
#         lms_truth = self.get_landmarks(truths)
#
#         if self.region == 'mouth':
#             lms_pred = lms_pred[48:68]
#             lms_truth = lms_truth[48:68]
#
#         # avarage
#         lms_pred = lms_pred - lms_pred.mean(0)
#         lms_truth = lms_truth - lms_truth.mean(0)
#
#         # distance
#         dist = np.sqrt(((lms_pred - lms_truth) ** 2).sum(1)).mean(0)
#
#         self.V += dist
#         self.N += 1
#
#     def measure(self):
#         return self.V / self.N
#
#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, f"LMD ({self.backend})"), self.measure(), global_step)
#
#     def report(self):
#         return f'LMD ({self.backend}) = {self.measure():.6f}'
