import os
import time
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
import pandas as pd
import math
import shutil
import json
import torch.nn.functional as F
import pywt
from sklearn.linear_model import LinearRegression
import joblib
import scipy.fftpack as fftpack
from statsmodels.tsa.seasonal import seasonal_decompose
import timm  # SwinV2 和混合模型需要 timm
import matplotlib.patheffects as path_effects

# 全局标志 (用于timm库导入提示)
TIMM_INIT_MESSAGE_PRINTED = False
try:
    if hasattr(timm, 'create_model'):
        _ = timm.create_model('resnet18', pretrained=False)
        if not TIMM_INIT_MESSAGE_PRINTED:
            print("✅ 成功导入 timm 库并检测到 create_model 功能。")
            TIMM_INIT_MESSAGE_PRINTED = True
    else:
        if not TIMM_INIT_MESSAGE_PRINTED:
            print("⚠️ 未找到 timm.create_model。SwinV2组件可能无法正确初始化。")
            TIMM_INIT_MESSAGE_PRINTED = True
except ImportError:
    if not TIMM_INIT_MESSAGE_PRINTED:
        print("⚠️ 未找到timm库。SwinV2组件可能无法正确初始化。")
        TIMM_INIT_MESSAGE_PRINTED = True
except Exception as e:
    if not TIMM_INIT_MESSAGE_PRINTED:
        print(f"⚠️ 初始化timm库时发生错误: {e}。SwinV2组件可能无法正确初始化。")
        TIMM_INIT_MESSAGE_PRINTED = True


# --- ConvLSTMCell and ConvLSTM Implementation ---
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length for kernel_size, hidden_dim and num_layers.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_layer, c_layer = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_layer, c_layer = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                             cur_state=[h_layer, c_layer])
                output_inner.append(h_layer)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h_layer, c_layer])

        return layer_output_list[-1][:, -1, :, :, :], last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or \
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# --- SwinConvLSTMModel Hybrid Model (MODIFIED __init__ and forward) ---
class SwinConvLSTMModel(nn.Module):
    def __init__(self, input_feature_dim, spatial_dims, pred_steps,
                 swin_model_name, swin_img_size, swin_in_chans,
                 swin_drop_rate, swin_drop_path_rate,
                 convlstm_hidden_dims,
                 convlstm_kernel_sizes,
                 convlstm_num_layers,
                 decoder_channels_list,
                 dropout_rate=0.1,
                 explicit_convlstm_input_dim=None
                 ):
        super().__init__()
        self.pred_steps = pred_steps
        self.H_orig_data, self.W_orig_data = spatial_dims
        self.swin_img_h, self.swin_img_w = swin_img_size

        if input_feature_dim != swin_in_chans:
            self.input_embed = nn.Linear(input_feature_dim, swin_in_chans)
            print(f"HybridModel: Input features ({input_feature_dim}) embedded to Swin in_chans ({swin_in_chans})")
        else:
            self.input_embed = nn.Identity()
            print(
                f"HybridModel: Input features ({input_feature_dim}) directly used as Swin in_chans ({swin_in_chans}).")

        try:
            self.swin_encoder = timm.create_model(
                swin_model_name, pretrained=False, img_size=swin_img_size,
                in_chans=swin_in_chans, num_classes=0, global_pool='',
                drop_rate=swin_drop_rate, drop_path_rate=swin_drop_path_rate
            )
            print(f"✅ 成功创建 Swin Transformer V2: {swin_model_name} for HybridModel")
        except Exception as e:
            print(f"❌ 创建 Swin Transformer V2 ({swin_model_name}) 失败: {e}");
            raise e

        with torch.no_grad():
            dummy_swin_input = torch.randn(1, swin_in_chans, self.swin_img_h, self.swin_img_w)
            swin_output_raw_dummy = self.swin_encoder.forward_features(dummy_swin_input)
            print(f"DEBUG: Raw Swin dummy output shape from forward_features: {swin_output_raw_dummy.shape}")

            self.swin_native_num_features = self.swin_encoder.num_features

            if swin_output_raw_dummy.ndim == 3 and swin_output_raw_dummy.shape[-1] == self.swin_native_num_features:
                B_dummy, L_dummy, C_dummy = swin_output_raw_dummy.shape
                if hasattr(self.swin_encoder.patch_embed, 'grid_size'):
                    gs = self.swin_encoder.patch_embed.grid_size
                    self.swin_feat_h, self.swin_feat_w = gs[0], gs[1]
                else:
                    patch_s = self.swin_encoder.patch_embed.patch_size[0]
                    num_s = len(self.swin_encoder.stages)
                    factor = patch_s * (2 ** (num_s - 1))
                    self.swin_feat_h = self.swin_img_h // factor
                    self.swin_feat_w = self.swin_img_w // factor
                    print(
                        f"DEBUG: Fallback H_feat, W_feat calculation: {self.swin_feat_h}, {self.swin_feat_w} using factor {factor}")
                if L_dummy == self.swin_feat_h * self.swin_feat_w:
                    self.swin_channels_for_convlstm = C_dummy
                    print(
                        f"DEBUG: Swin output is [B,L,C]. C={C_dummy}, H_feat={self.swin_feat_h}, W_feat={self.swin_feat_w}")
                else:
                    raise ValueError(
                        f"Swin output L_dummy ({L_dummy}) != expected H_feat*W_feat ({self.swin_feat_h * self.swin_feat_w}).")
            elif swin_output_raw_dummy.ndim == 4 and swin_output_raw_dummy.shape[
                1] == self.swin_native_num_features:  # [B, C, H, W]
                self.swin_channels_for_convlstm = swin_output_raw_dummy.shape[1]
                self.swin_feat_h = swin_output_raw_dummy.shape[2]
                self.swin_feat_w = swin_output_raw_dummy.shape[3]
                print(f"DEBUG: Swin output is [B,C,H,W]: {swin_output_raw_dummy.shape}")
            elif swin_output_raw_dummy.ndim == 4 and swin_output_raw_dummy.shape[
                3] == self.swin_native_num_features:  # [B, H, W, C] - THIS IS THE NEW CASE
                print(f"DEBUG: Swin output is [B,H,W,C]: {swin_output_raw_dummy.shape}. Will permute to [B,C,H,W].")
                # For __init__, we only need to correctly identify the channel number and feature map size
                self.swin_channels_for_convlstm = swin_output_raw_dummy.shape[3]  # C is the last dim
                self.swin_feat_h = swin_output_raw_dummy.shape[1]  # H is the second dim
                self.swin_feat_w = swin_output_raw_dummy.shape[2]  # W is the third dim
                # The actual permute will happen in the forward pass for each time step
                print(
                    f"DEBUG: Interpreted Swin output as [B,H,W,C]. C={self.swin_channels_for_convlstm}, H_feat={self.swin_feat_h}, W_feat={self.swin_feat_w}")
            else:
                raise ValueError(
                    f"Unexpected Swin output shape: {swin_output_raw_dummy.shape}. Checked for [B,L,C], [B,C,H,W], and [B,H,W,C] with C={self.swin_native_num_features}")

        _convlstm_input_dim = self.swin_channels_for_convlstm
        if explicit_convlstm_input_dim is not None and explicit_convlstm_input_dim != self.swin_channels_for_convlstm:
            self.adapter_conv = nn.Conv2d(self.swin_channels_for_convlstm, explicit_convlstm_input_dim, kernel_size=1)
            _convlstm_input_dim = explicit_convlstm_input_dim
            print(
                f"HybridModel: Added adapter_conv from Swin out ({self.swin_channels_for_convlstm}) to ConvLSTM target in ({_convlstm_input_dim})")
        else:
            self.adapter_conv = nn.Identity()
            print(f"HybridModel: ConvLSTM input dimension set to Swin output channels: {_convlstm_input_dim}")

        self.conv_lstm = ConvLSTM(input_dim=_convlstm_input_dim,
                                  hidden_dim=convlstm_hidden_dims,
                                  kernel_size=convlstm_kernel_sizes,
                                  num_layers=convlstm_num_layers,
                                  batch_first=True, bias=True)

        decoder_input_channels = convlstm_hidden_dims[-1] if isinstance(convlstm_hidden_dims,
                                                                        list) else convlstm_hidden_dims
        decoder_layers = []
        current_decoder_ch = decoder_input_channels
        for i, out_c in enumerate(decoder_channels_list):
            decoder_layers.append(nn.ConvTranspose2d(current_decoder_ch, out_c, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.BatchNorm2d(out_c))
            decoder_layers.append(nn.GELU())
            if i < len(decoder_channels_list) - 1:
                decoder_layers.append(nn.Dropout2d(dropout_rate))
            current_decoder_ch = out_c
        decoder_layers.append(nn.Conv2d(current_decoder_ch, pred_steps, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.input_embed, nn.Linear):
            nn.init.xavier_uniform_(self.input_embed.weight)
            if self.input_embed.bias is not None: nn.init.zeros_(self.input_embed.bias)
        if isinstance(self.adapter_conv, nn.Conv2d):
            if hasattr(self.adapter_conv, 'weight') and self.adapter_conv.weight is not None:
                nn.init.kaiming_normal_(self.adapter_conv.weight, mode='fan_out', nonlinearity='relu')
                if self.adapter_conv.bias is not None: nn.init.constant_(self.adapter_conv.bias, 0)
        for m in self.decoder.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1);
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, H_data, W_data, T_in, F_in = x.shape

        swin_features_sequence = []
        for t in range(T_in):
            x_t = x[:, :, :, t, :]
            if not isinstance(self.input_embed, nn.Identity):
                x_t = self.input_embed(x_t)
            x_t_swin_input = x_t.permute(0, 3, 1, 2).contiguous()

            if x_t_swin_input.shape[2] != self.swin_img_h or x_t_swin_input.shape[3] != self.swin_img_w:
                x_t_swin_input = F.interpolate(x_t_swin_input, size=(self.swin_img_h, self.swin_img_w),
                                               mode='bilinear', align_corners=False)

            swin_output_raw_t = self.swin_encoder.forward_features(x_t_swin_input)

            if swin_output_raw_t.ndim == 3 and swin_output_raw_t.shape[-1] == self.swin_channels_for_convlstm:
                B_t, L_t, C_t = swin_output_raw_t.shape
                if L_t == self.swin_feat_h * self.swin_feat_w:
                    swin_output_t_reshaped = swin_output_raw_t.permute(0, 2, 1).reshape(B_t, C_t, self.swin_feat_h,
                                                                                        self.swin_feat_w)
                else:
                    raise ValueError(
                        f"Swin L_t mismatch in forward: L_t={L_t}, expected H*W={self.swin_feat_h * self.swin_feat_w}")
            elif swin_output_raw_t.ndim == 4 and swin_output_raw_t.shape[
                1] == self.swin_channels_for_convlstm:  # [B, C, H, W]
                swin_output_t_reshaped = swin_output_raw_t
            elif swin_output_raw_t.ndim == 4 and swin_output_raw_t.shape[
                3] == self.swin_channels_for_convlstm:  # [B, H, W, C] - THIS IS THE NEW CASE
                swin_output_t_reshaped = swin_output_raw_t.permute(0, 3, 1, 2)  # Permute to [B, C, H, W]
            else:
                raise ValueError(
                    f"Unexpected Swin output shape in forward: {swin_output_raw_t.shape}. Expected C={self.swin_channels_for_convlstm} in correct dimension.")

            swin_output_adapted_t = self.adapter_conv(swin_output_t_reshaped)
            swin_features_sequence.append(swin_output_adapted_t)

        swin_features_stacked = torch.stack(swin_features_sequence, dim=1)
        convlstm_output_features, _ = self.conv_lstm(swin_features_stacked)
        decoded_features = self.decoder(convlstm_output_features)

        if decoded_features.shape[2] != H_data or decoded_features.shape[3] != W_data:
            predictions_raw = F.interpolate(decoded_features, size=(H_data, W_data),
                                            mode='bilinear', align_corners=False)
        else:
            predictions_raw = decoded_features
        predictions = predictions_raw.permute(0, 2, 3, 1).contiguous()
        return predictions


# --- 配置参数 ---
DATA_PATH = "D:/Pycharm-code/OSTIA/SST_2000-01-01-2021-12-31_Celsius_v2.nc"
WINDOW_SIZE = 21
PRED_STEPS = 7
DROPOUT = 0.1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
SPATIAL_DIMS = (120, 100)
TEST_START = '2018-01-01'
OBS_POINTS = [(30.5, 122.6), (30.1, 122.6)]
NUM_WORKERS = 0
PATIENCE = 10

SWIN_MODEL_NAME = 'swinv2_tiny_window8_256'
SWIN_ENCODER_IMG_SIZE = (64, 64)  # SwinV2 'tiny_patch4_window8_256' has patch_size=4, window_size=8, img_size=256
# If using (64,64) here, timm will adapt. Feature map size will be 64/32 = 2x2
SWIN_IN_CHANS_FOR_HYBRID = 64
SWINV2_DROP_RATE = 0.1
SWINV2_DROP_PATH_RATE = 0.1

EXPLICIT_CONVLSTM_INPUT_DIM = None  # Let ConvLSTM input be Swin's native output channels (e.g. 768 for tiny)
# Or set to e.g. 256 to use adapter_conv to reduce from 768 to 256
CONVLSTM_HIDDEN_DIMS = [128, 64]
CONVLSTM_KERNEL_SIZES = [(3, 3), (3, 3)]
CONVLSTM_NUM_LAYERS = len(CONVLSTM_HIDDEN_DIMS)

HYBRID_DECODER_CHANNELS = [128, 64]  # Decoder upsamples from CONVLSTM_HIDDEN_DIMS[-1] (e.g. 64)

WAVELET_SIGMA_SCALE = 2.2
WINSORIZE_LOWER_PERCENTILE = 0.5
WINSORIZE_UPPER_PERCENTILE = 99.5
SEA_ICE_ENABLED = False
SSIM_WINDOW_SIZE = 11

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# --- 辅助函数, SSTWindowDataset, CopernicusDataProcessor (与之前脚本一致) ---
def apply_fft_filter(data_array, low_freq_cutoff=0.008, high_freq_cutoff=0.45):
    filtered_data = data_array.copy();
    n_time = len(data_array.time);
    n_lat = len(data_array.lat);
    n_lon = len(data_array.lon)
    low_cutoff_idx = int(n_time * low_freq_cutoff);
    high_cutoff_idx = int(n_time * high_freq_cutoff)
    for i in tqdm(range(n_lat), desc="FFT滤波 (纬度)", leave=False, disable=True):
        for j in range(n_lon):
            ts = data_array[:, i, j].values
            if np.all(np.isnan(ts)): continue
            nan_mask = np.isnan(ts);
            mean_val = np.nanmean(ts)
            if np.isnan(mean_val): continue
            ts_filled = np.where(nan_mask, mean_val, ts)
            try:
                fft_vals = fftpack.fft(ts_filled);
                filter_mask = np.ones(len(fft_vals), dtype=complex)
                if high_cutoff_idx < n_time // 2: filter_mask[high_cutoff_idx:n_time - high_cutoff_idx] = 0
                if low_cutoff_idx > 0: filter_mask[1:low_cutoff_idx] = 0; filter_mask[n_time - low_cutoff_idx + 1:] = 0
                filtered_fft = fft_vals * filter_mask;
                filtered_ts = np.real(fftpack.ifft(filtered_fft))
                if np.any(nan_mask): filtered_ts[nan_mask] = np.nan
                filtered_data.values[:, i, j] = filtered_ts
            except Exception:
                pass
    return filtered_data


def decompose_seasonal(data_array, period=365 * 2, model='additive'):
    decomposed_data = data_array.copy();
    n_lat = len(data_array.lat);
    n_lon = len(data_array.lon)
    for i in tqdm(range(n_lat), desc="季节性分解 (纬度)", leave=False, disable=True):
        for j in range(n_lon):
            ts = data_array[:, i, j].values
            if np.all(np.isnan(ts)): continue
            nan_mask = np.isnan(ts);
            mean_val = np.nanmean(ts)
            if np.isnan(mean_val): continue
            ts_filled = np.where(nan_mask, mean_val, ts)
            try:
                ts_series = pd.Series(ts_filled)
                result = seasonal_decompose(ts_series, model=model, period=period, extrapolate_trend='freq')
                if model == 'additive':
                    adjusted_ts = result.trend + result.resid
                else:
                    adjusted_ts = result.trend * result.resid
                adjusted_ts = pd.Series(adjusted_ts).fillna(method='ffill').fillna(method='bfill').values
                if np.any(nan_mask): adjusted_ts[nan_mask] = np.nan
                decomposed_data.values[:, i, j] = adjusted_ts
            except Exception:
                pass
    return decomposed_data


class SSTWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, window_size, pred_steps, time_coordinates_for_data_tensor):
        self.data = data_tensor;
        self.window_size = window_size;
        self.pred_steps = pred_steps
        self.time_coords = time_coordinates_for_data_tensor;
        self.num_times = self.data.size(2)
        self.valid_length = self.num_times - window_size - pred_steps + 1

    def __len__(self): return self.valid_length

    def __getitem__(self, idx):
        input_slice = slice(idx, idx + self.window_size);
        target_slice = slice(idx + self.window_size, idx + self.window_size + self.pred_steps)
        inputs = self.data[:, :, input_slice, :];
        targets_norm_anom = self.data[:, :, target_slice, 0]
        actual_target_dates = self.time_coords[target_slice] if self.time_coords is not None else None
        return inputs, targets_norm_anom, actual_target_dates


class CopernicusDataProcessor:
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            print(f"❌错误: 数据文件 {filepath} 未找到。请检查路径。")
            times_sim = pd.to_datetime([f'2000-01-{d:02d}' for d in range(1, WINDOW_SIZE + PRED_STEPS + 50)])
            lats_sim = np.linspace(20, 50, SPATIAL_DIMS[0])
            lons_sim = np.linspace(110, 140, SPATIAL_DIMS[1])
            sst_data_sim = np.random.rand(len(times_sim), len(lats_sim), len(lons_sim)) * 10 + 15
            self.ds = xr.Dataset({'sst': (('time', 'lat', 'lon'), sst_data_sim)},
                                 coords={'time': times_sim, 'lat': lats_sim, 'lon': lons_sim})
            print("⚠️警告: 由于数据文件未找到，已使用最小化模拟数据进行初始化。功能将受限。")
        else:
            self.ds = xr.open_dataset(filepath);

        self.scalers = {};
        self.land_mask = None
        self._preprocess_dims();
        self._validate_dataset();
        self.has_ice_data = False
        if SEA_ICE_ENABLED: self._try_load_sea_ice()
        self.splits = {'train': ('2000-01-01', '2015-12-31'), 'val': ('2016-01-01', '2017-12-31'),
                       'test': (TEST_START, '2021-12-31')}

    def _try_load_sea_ice(self):
        pass

    def _create_synthetic_ice_data(self):
        pass

    def _preprocess_dims(self):
        if 'analysed_sst' in self.ds.data_vars:
            self.ds = self.ds.rename({'analysed_sst': 'sst'})
        elif 'sea_surface_temperature' in self.ds.data_vars:
            self.ds = self.ds.rename({'sea_surface_temperature': 'sst'})
        if 'latitude' in self.ds.coords: self.ds = self.ds.rename({'latitude': 'lat'})
        if 'longitude' in self.ds.coords: self.ds = self.ds.rename({'longitude': 'lon'})

        if not ('time' in self.ds.coords and 'lat' in self.ds.coords and 'lon' in self.ds.coords):
            raise ValueError("Dataset missing time, lat, or lon coordinates after renaming.")
        self.ds = self.ds.transpose('time', 'lat', 'lon', ...)
        self.times = self.ds.time.values

        if len(self.ds.lat) > SPATIAL_DIMS[0]:
            self.ds = self.ds.isel(lat=slice(0, SPATIAL_DIMS[0]))
        if len(self.ds.lon) > SPATIAL_DIMS[1]:
            self.ds = self.ds.isel(lon=slice(0, SPATIAL_DIMS[1]))

        self.lats = self.ds.lat.values
        self.lons = self.ds.lon.values

        if hasattr(self.ds.sst, '_FillValue'):
            fill_value = self.ds.sst._FillValue
            self.ds['sst'] = self.ds.sst.where(self.ds.sst != fill_value, np.nan)

        self.land_mask = xr.where(self.ds.sst.isnull().all(dim='time'), 0, 1).data.astype(np.float32)
        if self.land_mask.shape != SPATIAL_DIMS:
            print(
                f"WARNING: Land mask shape {self.land_mask.shape} does not match SPATIAL_DIMS {SPATIAL_DIMS}. This might indicate an issue in cropping or SPATIAL_DIMS definition.")

    def _validate_dataset(self):
        if not all(dim in self.ds.dims for dim in ['time', 'lat', 'lon']):
            raise ValueError("Dataset missing required dimensions after _preprocess_dims.")
        if 'sst' not in self.ds.data_vars:
            raise ValueError("Dataset missing 'sst' after _preprocess_dims.")
        if len(self.ds.lat) != SPATIAL_DIMS[0] or len(self.ds.lon) != SPATIAL_DIMS[1]:
            print(
                f"WARNING: Final dataset spatial dimensions ({len(self.ds.lat)}, {len(self.ds.lon)}) do not match SPATIAL_DIMS ({SPATIAL_DIMS[0]}, {SPATIAL_DIMS[1]}). Check cropping logic.")

    def _add_positional_features(self):
        lat_rad = np.deg2rad(self.lats).astype(np.float32);
        lon_rad = np.deg2rad(self.lons).astype(np.float32)
        lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)
        self.ds['x'] = xr.DataArray(np.cos(lat_grid) * np.cos(lon_grid), dims=['lat', 'lon'],
                                    coords={'lat': self.lats, 'lon': self.lons})
        self.ds['y'] = xr.DataArray(np.cos(lat_grid) * np.sin(lon_grid), dims=['lat', 'lon'],
                                    coords={'lat': self.lats, 'lon': self.lons})
        self.ds['z'] = xr.DataArray(np.sin(lat_grid), dims=['lat', 'lon'], coords={'lat': self.lats, 'lon': self.lons})
        self.ds['lat_scaled'] = xr.DataArray((self.lats - np.mean(self.lats)) / (np.std(self.lats) + 1e-6),
                                             dims=['lat'], coords={'lat': self.lats})
        self.ds['lon_scaled'] = xr.DataArray((self.lons - np.mean(self.lons)) / (np.std(self.lons) + 1e-6),
                                             dims=['lon'], coords={'lon': self.lons})

    def _add_temporal_features(self):
        time_dt = self.ds.time.dt;
        day_of_year = time_dt.dayofyear.values - 1
        days_in_year = (365 + time_dt.is_leap_year.values).astype(float);
        year_angle = 2 * np.pi * day_of_year / days_in_year
        self.ds['year_sin'] = xr.DataArray(np.sin(year_angle).astype(np.float32), dims=['time'],
                                           coords={'time': self.ds.time})
        self.ds['year_cos'] = xr.DataArray(np.cos(year_angle).astype(np.float32), dims=['time'],
                                           coords={'time': self.ds.time})
        month_progress = (time_dt.day.values - 1) / time_dt.days_in_month.values.astype(float);
        month_angle = 2 * np.pi * ((time_dt.month.values - 1) + month_progress) / 12
        self.ds['month_sin'] = xr.DataArray(np.sin(month_angle).astype(np.float32), dims=['time'],
                                            coords={'time': self.ds.time})
        self.ds['month_cos'] = xr.DataArray(np.cos(month_angle).astype(np.float32), dims=['time'],
                                            coords={'time': self.ds.time})
        self.ds['season'] = xr.DataArray(((time_dt.month.values % 12 + 3) // 3).astype(np.float32), dims=['time'],
                                         coords={'time': self.ds.time})

    def _normalize(self, dataset_xr, is_test=False, dataset_name=None):
        if dataset_name is None: dataset_name = '测试集' if is_test else '训练集'
        sst_feature_name = 'sst';
        time_features_names = ['year_sin', 'year_cos', 'month_sin', 'month_cos', 'season']
        spatial_features_names = ['x', 'y', 'z', 'lat_scaled', 'lon_scaled']
        all_feature_list_ordered = [sst_feature_name] + time_features_names + spatial_features_names
        if self.has_ice_data and SEA_ICE_ENABLED: all_feature_list_ordered.append('ice')
        n_lat, n_lon = len(dataset_xr.lat), len(dataset_xr.lon)
        n_time = len(dataset_xr.time)
        num_total_features = len(all_feature_list_ordered)
        normalized_data_cube = np.zeros((n_lat, n_lon, n_time, num_total_features), dtype=np.float32)
        for lat_idx in tqdm(range(n_lat), desc=f'标准化 {dataset_name} ({n_lat}x{n_lon})', leave=False, disable=True):
            for lon_idx in range(n_lon):
                sst_pixel_series = dataset_xr[sst_feature_name][:, lat_idx, lon_idx].data.reshape(-1, 1)
                if self.land_mask[lat_idx, lon_idx] == 0:
                    normalized_data_cube[lat_idx, lon_idx, :, 0] = 0.0
                else:
                    if not is_test and (lat_idx, lon_idx) not in self.scalers:
                        valid_sst_for_scaler = sst_pixel_series[~np.isnan(sst_pixel_series).flatten(), :]
                        if valid_sst_for_scaler.shape[0] > 0:
                            self.scalers[(lat_idx, lon_idx)] = StandardScaler().fit(valid_sst_for_scaler)
                        else:
                            self.scalers[(lat_idx, lon_idx)] = None
                    scaler = self.scalers.get((lat_idx, lon_idx))
                    if scaler:
                        nan_mask_pixel = np.isnan(sst_pixel_series).flatten()
                        temp_filled_series = np.where(nan_mask_pixel, 0, sst_pixel_series.flatten())
                        transformed_values = scaler.transform(temp_filled_series.reshape(-1, 1)).flatten()
                        transformed_values[nan_mask_pixel] = np.nan
                        normalized_data_cube[lat_idx, lon_idx, :, 0] = transformed_values
                    elif self.land_mask[lat_idx, lon_idx] == 1:
                        mean_val = np.nanmean(sst_pixel_series);
                        std_val = np.nanstd(sst_pixel_series)
                        if not np.isnan(mean_val) and not np.isnan(std_val) and std_val > 1e-9:
                            normalized_data_cube[lat_idx, lon_idx, :, 0] = (
                                                                                       sst_pixel_series.flatten() - mean_val) / std_val
                        else:
                            normalized_data_cube[lat_idx, lon_idx, :, 0] = np.nan
                    else:
                        normalized_data_cube[lat_idx, lon_idx, :, 0] = 0.0
                for feat_idx_list, feat_name in enumerate(time_features_names):
                    actual_idx_in_cube = all_feature_list_ordered.index(feat_name)
                    normalized_data_cube[lat_idx, lon_idx, :, actual_idx_in_cube] = dataset_xr[feat_name].data
                for feat_idx_list, feat_name in enumerate(spatial_features_names):
                    actual_idx_in_cube = all_feature_list_ordered.index(feat_name)
                    if feat_name == 'lat_scaled':
                        normalized_data_cube[lat_idx, lon_idx, :, actual_idx_in_cube] = dataset_xr[feat_name].data[
                            lat_idx]
                    elif feat_name == 'lon_scaled':
                        normalized_data_cube[lat_idx, lon_idx, :, actual_idx_in_cube] = dataset_xr[feat_name].data[
                            lon_idx]
                    else:
                        normalized_data_cube[lat_idx, lon_idx, :, actual_idx_in_cube] = dataset_xr[feat_name].data[
                            lat_idx, lon_idx]
        normalized_data_cube[:, :, :, 0] = np.nan_to_num(normalized_data_cube[:, :, :, 0], nan=0.0)
        return torch.tensor(normalized_data_cube, dtype=torch.float32)

    def preprocess(self):
        print("\n🚀 启动预处理流程 (完整版)...");
        self.original_sst = self.ds.sst.copy(deep=True);
        self.ds['sst'] = self.ds.sst.where(~np.isnan(self.ds.sst), np.nan);
        if isinstance(self.ds.sst, xr.DataArray) and set(self.ds.sst.dims) == {'time', 'lat', 'lon'}:
            self.ds['sst'] = self._apply_wavelet_denoising(self.ds.sst.transpose('time', 'lat', 'lon'));
        else:
            print("警告: SST数据格式不符合小波去噪预期，跳过。")
        land_mask_xr = xr.DataArray(self.land_mask, dims=['lat', 'lon'], coords={'lat': self.lats, 'lon': self.lons});
        self.ds['sst'] = self.clean_sst_extreme_daily_changes(self.ds.sst, land_mask_xr=land_mask_xr);
        ocean_mask_xr = land_mask_xr.broadcast_like(self.ds.sst);
        ocean_sst = self.ds.sst.where(ocean_mask_xr == 1);
        filled_sst_f = ocean_sst.ffill(dim='time');
        filled_sst_b = filled_sst_f.bfill(dim='time');
        filled_sst_b = filled_sst_b.fillna(0);
        self.ds['sst'] = xr.where(ocean_mask_xr == 1, filled_sst_b, np.nan);
        clim_base_period = self.ds.sst.sel(time=slice(*self.splits['train']))
        if clim_base_period.notnull().any():
            self.climatology = clim_base_period.groupby('time.dayofyear').mean('time', skipna=True)
        else:
            doy_coords = xr.DataArray(np.arange(1, 367), dims=['dayofyear'], name='dayofyear');
            self.climatology = xr.DataArray(0.0, coords=[doy_coords, self.lats, self.lons],
                                            dims=['dayofyear', 'lat', 'lon'])
        day_of_year_values = self.ds.time.dt.dayofyear.data;
        clim_full_data = np.zeros_like(self.ds.sst.data)
        for t_idx, doy_val in enumerate(day_of_year_values):
            try:
                actual_doy_to_use = int(doy_val)
                if actual_doy_to_use not in self.climatology.dayofyear.data:
                    actual_doy_to_use = 365 if doy_val == 366 and 365 in self.climatology.dayofyear.data else \
                        (self.climatology.dayofyear.min().item() if self.climatology.dayofyear.size > 0 else 1)
                clim_slice = self.climatology.sel(dayofyear=actual_doy_to_use).data;
                clim_full_data[t_idx, :, :] = clim_slice
            except Exception:
                clim_slice = self.climatology.isel(
                    dayofyear=0).data if self.climatology.dayofyear.size > 0 else np.zeros(
                    (len(self.lats), len(self.lons)))
                if t_idx > 0:
                    clim_full_data[t_idx, :, :] = clim_full_data[t_idx - 1, :, :]
                else:
                    clim_full_data[t_idx, :, :] = clim_slice
        clim_full = xr.DataArray(data=clim_full_data, dims=['time', 'lat', 'lon'], coords=self.ds.sst.coords)
        self.ds['sst_anom'] = self.ds.sst - clim_full;
        self.ds = self.ds.drop_vars('sst');
        self.ds = self.ds.rename({'sst_anom': 'sst'});
        ocean_points_mask_thw = np.broadcast_to(self.land_mask[np.newaxis, :, :] == 1, self.ds.sst.shape)
        sst_anom_ocean_values = self.ds.sst.data[ocean_points_mask_thw & ~np.isnan(self.ds.sst.data)]
        if sst_anom_ocean_values.size > 0:
            lower_bound = np.percentile(sst_anom_ocean_values, WINSORIZE_LOWER_PERCENTILE);
            upper_bound = np.percentile(sst_anom_ocean_values, WINSORIZE_UPPER_PERCENTILE)
            self.ds['sst'] = self.ds.sst.clip(min=lower_bound, max=upper_bound);
        else:
            print("⚠️ SST异常值数据在海洋点上为空或全为NaN，跳过Winsorizing。")
        self._add_positional_features();
        self._add_temporal_features();
        print(f"✅ 清洗和特征工程完成 | SST有效值占比 (处理后): {self.ds.sst.notnull().mean().item():.2%}")
        train_data = self._normalize(self.ds.sel(time=slice(*self.splits['train'])), is_test=False,
                                     dataset_name="训练集")
        val_data = self._normalize(self.ds.sel(time=slice(*self.splits['val'])), is_test=True, dataset_name="验证集")
        test_data = self._normalize(self.ds.sel(time=slice(*self.splits['test'])), is_test=True, dataset_name="测试集")
        print(f"训练集形状: {train_data.shape}, 验证集形状: {val_data.shape}, 测试集形状: {test_data.shape}")
        return train_data, val_data, test_data

    def clean_sst_extreme_daily_changes(self, sst_data_xr, abs_diff_threshold=2.0, land_mask_xr=None):
        sst_cleaned = sst_data_xr.copy(deep=True);
        prev_sst = sst_cleaned.shift(time=1)
        actual_diff = sst_cleaned - prev_sst
        condition_to_clip = (np.abs(actual_diff) > abs_diff_threshold) & (~prev_sst.isnull())
        if land_mask_xr is not None:
            ocean_mask_bc = land_mask_xr.broadcast_like(condition_to_clip);
            condition_to_clip = condition_to_clip & (ocean_mask_bc == 1)
        num_points_to_clip = condition_to_clip.sum().item()
        if num_points_to_clip > 0:
            signed_thresholded_diff = xr.apply_ufunc(np.sign, actual_diff, dask="parallelized",
                                                     output_dtypes=[actual_diff.dtype]) * abs_diff_threshold
            clipped_values = prev_sst + signed_thresholded_diff;
            sst_final_cleaned = xr.where(condition_to_clip, clipped_values, sst_cleaned)
            return sst_final_cleaned
        return sst_cleaned

    def _apply_wavelet_denoising(self, sst_data_xr_thw, wavelet='db4', level=None, mode='soft',
                                 threshold_sigma_scale=2.2):
        denoised_sst_data_np = sst_data_xr_thw.data.copy()
        num_time, num_lat, num_lon = denoised_sst_data_np.shape
        for i in tqdm(range(num_lat), desc="小波去噪 (纬度)", leave=False, disable=True):
            for j in range(num_lon):
                if self.land_mask[i, j] == 0: continue
                pixel_ts = denoised_sst_data_np[:, i, j].copy()
                if np.all(np.isnan(pixel_ts)) or len(pixel_ts) < 8: continue
                nan_mask = np.isnan(pixel_ts);
                mean_val = np.nanmean(pixel_ts)
                if np.isnan(mean_val): continue
                pixel_ts_filled = np.where(nan_mask, mean_val, pixel_ts)
                try:
                    current_level = level if level is not None else min(
                        pywt.dwt_max_level(len(pixel_ts_filled), wavelet), 4)
                    coeffs = pywt.wavedec(pixel_ts_filled, wavelet=wavelet, level=current_level);
                    coeffs_thresholded = [coeffs[0]]
                    for detail_coeffs in coeffs[1:]:
                        sigma = np.median(np.abs(detail_coeffs)) / 0.6745
                        sigma = np.std(detail_coeffs) if sigma == 0 or np.isnan(sigma) else sigma
                        if sigma == 0 or np.isnan(sigma): sigma = 1e-9
                        threshold = sigma * threshold_sigma_scale;
                        coeffs_thresholded.append(pywt.threshold(detail_coeffs, threshold, mode=mode))
                    pixel_ts_denoised = pywt.waverec(coeffs_thresholded, wavelet=wavelet)
                    if len(pixel_ts_denoised) != len(pixel_ts_filled):
                        pixel_ts_denoised = pixel_ts_denoised[:len(pixel_ts_filled)]
                    pixel_ts_denoised[nan_mask] = np.nan
                    denoised_sst_data_np[:, i, j] = pixel_ts_denoised
                except Exception:
                    temp_fallback = pixel_ts_filled.copy()
                    temp_fallback[nan_mask] = np.nan
                    denoised_sst_data_np[:, i, j] = temp_fallback
        return xr.DataArray(denoised_sst_data_np, coords=sst_data_xr_thw.coords, dims=sst_data_xr_thw.dims,
                            name=sst_data_xr_thw.name)


# --- SSIMLoss (Optional) ---
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def _gaussian(self, window_size, sigma):
        gauss = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-gauss ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def _create_window(self, window_size, C_eff, device):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(C_eff, 1, window_size, window_size).contiguous().to(device)
        return window

    def forward(self, x, y):
        x = x.float().permute(0, 3, 1, 2)
        y = y.float().permute(0, 3, 1, 2)
        B, C_eff, H, W = x.size()
        window = self._create_window(self.window_size, C_eff, x.device)
        K1 = 0.01;
        K2 = 0.03;
        L = 1
        C1 = (K1 * L) ** 2;
        C2 = (K2 * L) ** 2
        padding = self.window_size // 2
        mu1 = F.conv2d(x, window, padding=padding, groups=C_eff)
        mu2 = F.conv2d(y, window, padding=padding, groups=C_eff)
        mu1_sq = mu1.pow(2);
        mu2_sq = mu2.pow(2);
        mu1_mu2 = mu1 * mu2
        sigma1_sq = torch.clamp(F.conv2d(x * x, window, padding=padding, groups=C_eff) - mu1_sq, min=0.0)
        sigma2_sq = torch.clamp(F.conv2d(y * y, window, padding=padding, groups=C_eff) - mu2_sq, min=0.0)
        sigma12 = F.conv2d(x * y, window, padding=padding, groups=C_eff) - mu1_mu2
        ssim_numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ssim_denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = ssim_numerator / (ssim_denominator + 1e-6);
        ssim_map = torch.clamp(ssim_map, -1.0, 1.0)
        return 1.0 - ssim_map.mean()


# --- ModelTrainer ---
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, scalers, land_mask=None, model_name="Hybrid_SwinConvLSTM"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
        self.model = model.to(self.device);
        self.model_name = model_name;
        self.scalers_dict = scalers
        self.land_mask_np = land_mask.astype(np.float32) if land_mask is not None else np.ones(
            (SPATIAL_DIMS[0], SPATIAL_DIMS[1]), dtype=np.float32)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        self.l1_criterion = nn.L1Loss();
        self.huber_criterion = nn.HuberLoss(delta=0.3);
        self.mse_criterion = nn.MSELoss()
        self.ssim_loss_fn = SSIMLoss(window_size=SSIM_WINDOW_SIZE)
        self.optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=2,
                                                                        eta_min=max(1e-7, LEARNING_RATE / 100))
        self.train_loader = train_loader;
        self.val_loader = val_loader
        lat_dim, lon_dim = SPATIAL_DIMS
        means_sst = np.full((lat_dim, lon_dim), np.nan, dtype=np.float32)
        stds_sst = np.full((lat_dim, lon_dim), np.nan, dtype=np.float32)
        if scalers:
            for (i, j), scaler_item in scalers.items():
                if i < lat_dim and j < lon_dim:
                    if scaler_item and hasattr(scaler_item, 'mean_') and scaler_item.mean_ is not None and \
                            hasattr(scaler_item, 'scale_') and scaler_item.scale_ is not None and \
                            scaler_item.scale_.size > 0 and scaler_item.scale_[0] != 0:
                        means_sst[i, j] = scaler_item.mean_[0]
                        stds_sst[i, j] = scaler_item.scale_[0]
        means_sst = np.nan_to_num(means_sst, nan=0.0)
        stds_sst = np.nan_to_num(stds_sst, nan=1.0)
        stds_sst[stds_sst == 0] = 1.0
        self.mean_tensor = torch.tensor(means_sst, dtype=torch.float32, device=self.device)[None, :, :, None]
        self.std_tensor = torch.tensor(stds_sst, dtype=torch.float32, device=self.device)[None, :, :, None]
        self.best_val_loss = float('inf');
        self.early_stop_counter = 0;
        self.patience = PATIENCE
        self.train_losses = [];
        self.val_losses = [];
        self.learning_rates = [];
        self.start_epoch = 0
        self.loss_weights = {'l1': 0.6, 'huber': 0.1, 'mse': 0.0, 'ssim': 0.3}
        models_dir = "models";
        os.makedirs(models_dir, exist_ok=True)
        self.best_model_path = os.path.join(models_dir, f"best_{self.model_name}_model.pth")
        self.latest_model_path = os.path.join(models_dir, f"latest_{self.model_name}_model.pth")
        self._try_load_checkpoint()

    def _try_load_checkpoint(self):
        load_path = None
        if os.path.exists(self.best_model_path):
            print(f"发现已保存的最佳模型: {self.best_model_path}")
            load_path = self.best_model_path
        elif os.path.exists(self.latest_model_path):
            print(f"未找到最佳模型，但发现最新的模型: {self.latest_model_path}")
            load_path = self.latest_model_path
        if load_path:
            user_input = input(f"是否加载模型 {os.path.basename(load_path)} 并继续训练? (y/n): ").strip().lower()
            if user_input == 'y':
                self.load_checkpoint(load_path)
            else:
                print("将从头开始训练。")
        else:
            print("未找到已保存的模型，将从头开始训练。")

    def load_checkpoint(self, checkpoint_path):
        print(f"🔄 正在加载检查点: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and self.optimizer: self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and self.scheduler: self.scheduler.load_state_dict(
                checkpoint['scheduler_state_dict'])
            if 'grad_scaler_state_dict' in checkpoint and self.grad_scaler: self.grad_scaler.load_state_dict(
                checkpoint['grad_scaler_state_dict'])
            self.start_epoch = checkpoint.get('epoch', -1) + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if 'scalers' in checkpoint: self.scalers_dict = checkpoint['scalers']
            if 'land_mask' in checkpoint: self.land_mask_np = checkpoint['land_mask']
            self.train_losses = checkpoint.get('train_losses', []);
            self.val_losses = checkpoint.get('val_losses', [])
            self.learning_rates = checkpoint.get('learning_rates', [])
            print(f"✅ 检查点加载成功。将从轮次 {self.start_epoch} 继续。最佳验证损失: {self.best_val_loss:.4f}")
            return True
        except Exception as e:
            print(f"❌ 加载检查点 '{checkpoint_path}' 失败: {e}。将从头开始。")
            self.start_epoch = 0;
            self.best_val_loss = float('inf')
            self.train_losses = [];
            self.val_losses = [];
            self.learning_rates = []
            return False

    def compute_combined_loss(self, outputs_norm, targets_norm, denorm_outputs, denorm_targets):
        losses = {};
        losses['l1'] = self.l1_criterion(outputs_norm, targets_norm)
        losses['huber'] = self.huber_criterion(outputs_norm, targets_norm)
        losses['mse'] = self.mse_criterion(outputs_norm, targets_norm)
        losses['ssim'] = self.ssim_loss_fn(outputs_norm, targets_norm)
        losses['real_l1'] = self.l1_criterion(denorm_outputs, denorm_targets)
        combined_loss = sum(self.loss_weights.get(k, 0) * v for k, v in losses.items() if k != 'real_l1')
        return combined_loss, losses

    def train_epoch(self):
        self.model.train();
        total_real_l1_loss = 0.0;
        valid_batches = 0
        current_display_epoch = self.start_epoch + len(self.train_losses)
        progress = tqdm(self.train_loader, desc=f"训练 ({self.model_name}) E{current_display_epoch + 1}", leave=False)
        self.optimizer.zero_grad()
        for batch_idx, (inputs, targets_norm, _) in enumerate(progress):
            inputs = inputs.to(self.device).contiguous()
            targets_norm = targets_norm.to(self.device).contiguous()
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                outputs_norm = self.model(inputs)
                denorm_outputs = outputs_norm * self.std_tensor + self.mean_tensor
                denorm_targets = targets_norm * self.std_tensor + self.mean_tensor
                loss, loss_components = self.compute_combined_loss(outputs_norm, targets_norm, denorm_outputs,
                                                                   denorm_targets)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"批次 {batch_idx} 无效损失 (NaN/Inf), 跳过。");
                continue
            self.grad_scaler.scale(loss).backward();
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1 == len(self.train_loader)):
                self.grad_scaler.step(self.optimizer);
                self.grad_scaler.update()
                self.optimizer.zero_grad()
            total_real_l1_loss += loss_components['real_l1'].item();
            valid_batches += 1
            progress.set_postfix(loss=f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
                                 mae_norm=f"{loss_components['l1'].item():.4f}",
                                 mae_real=f"{loss_components['real_l1'].item():.4f}",
                                 ssim_loss=f"{loss_components['ssim'].item():.4f}",
                                 lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        return total_real_l1_loss / valid_batches if valid_batches > 0 else float('inf')

    def validate(self):
        self.model.eval();
        total_metrics = {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
        all_targets_flat, all_outputs_flat = [], [];
        valid_batches = 0
        with torch.no_grad():
            for inputs, targets_norm, _ in tqdm(self.val_loader, desc=f"验证 ({self.model_name})", leave=False):
                inputs = inputs.to(self.device).contiguous()
                targets_norm = targets_norm.to(self.device).contiguous()
                with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    outputs_norm = self.model(inputs)
                    denorm_outputs = outputs_norm * self.std_tensor + self.mean_tensor
                    denorm_targets = targets_norm * self.std_tensor + self.mean_tensor
                    denorm_outputs = torch.clamp(denorm_outputs, -20.0, 50.0)
                total_metrics['mae'] += self.l1_criterion(denorm_outputs, denorm_targets).item()
                total_metrics['rmse'] += np.sqrt(self.mse_criterion(denorm_outputs, denorm_targets).item())
                all_targets_flat.append(denorm_targets.cpu().numpy().flatten());
                all_outputs_flat.append(denorm_outputs.cpu().numpy().flatten())
                valid_batches += 1
        if valid_batches == 0: return float('inf'), {k: float('inf') for k in total_metrics}
        for k_metric in ['mae', 'rmse']: total_metrics[k_metric] /= valid_batches
        all_targets_flat = np.concatenate(all_targets_flat);
        all_outputs_flat = np.concatenate(all_outputs_flat)
        valid_idx = ~np.isnan(all_targets_flat) & ~np.isnan(all_outputs_flat) & \
                    ~np.isinf(all_targets_flat) & ~np.isinf(all_outputs_flat)
        if np.sum(valid_idx) > 1:
            true_valid = all_targets_flat[valid_idx];
            pred_valid = all_outputs_flat[valid_idx]
            ss_res = np.sum((true_valid - pred_valid) ** 2);
            ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
            total_metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-9)) if ss_tot > 1e-9 else 0.0
        else:
            total_metrics['r2'] = 0.0
        return total_metrics['mae'], total_metrics

    def run_training(self):
        interrupted_or_error = False
        try:
            for epoch_offset in range(EPOCHS - self.start_epoch):
                current_epoch = self.start_epoch + epoch_offset
                if current_epoch >= EPOCHS: break
                train_mae_real = self.train_epoch()
                current_lr = self.optimizer.param_groups[0]['lr']
                val_mae_real, val_metrics_all = self.validate()
                self.train_losses.append(train_mae_real);
                self.val_losses.append(val_mae_real);
                self.learning_rates.append(current_lr)
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_mae_real if np.isfinite(val_mae_real) else float('inf'))
                else:
                    self.scheduler.step()
                print(
                    f"E{current_epoch + 1}/{EPOCHS} ({self.model_name}) - TrMAE(real): {train_mae_real:.4f}, ValMAE(real): {val_mae_real:.4f} (RMSE: {val_metrics_all['rmse']:.4f}, R2: {val_metrics_all['r2']:.4f}), LR: {current_lr:.2e}")
                checkpoint_data = {
                    'model_state_dict': self.model.state_dict(), 'epoch': current_epoch,
                    'best_val_loss': self.best_val_loss,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'grad_scaler_state_dict': self.grad_scaler.state_dict(),
                    'scalers': self.scalers_dict, 'land_mask': self.land_mask_np,
                    'train_losses': self.train_losses, 'val_losses': self.val_losses,
                    'learning_rates': self.learning_rates
                }
                torch.save(checkpoint_data, self.latest_model_path)
                if val_mae_real < self.best_val_loss:
                    self.best_val_loss = val_mae_real;
                    self.early_stop_counter = 0
                    torch.save(checkpoint_data, self.best_model_path)
                    print(f"✅ Best {self.model_name} model saved @ E{current_epoch + 1}")
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.patience:
                        print(f"⚠️ {self.model_name} early stopping @ E{current_epoch + 1}");
                        interrupted_or_error = True;
                        break
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user!"); interrupted_or_error = True
        except Exception as e:
            print(f"\n❌ Training error: {e}");
            import traceback;
            traceback.print_exc();
            interrupted_or_error = True
        print(f"{self.model_name} training finished/stopped. Best validation MAE (real): {self.best_val_loss:.4f}")
        load_final_model_path = None
        if os.path.exists(self.best_model_path):
            load_final_model_path = self.best_model_path
        elif os.path.exists(self.latest_model_path) and interrupted_or_error:
            load_final_model_path = self.latest_model_path
        if load_final_model_path:
            print(f"Loading model from: {load_final_model_path} for final use.")
            try:
                checkpoint = torch.load(load_final_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)
                print(f"✅ Model loaded from {load_final_model_path}.")
            except Exception as e_load:
                print(f"❌ Failed to load model from {load_final_model_path}: {e_load}")
        else:
            print("⚠️ No saved model found to load for final use.")
        return self.model


# --- ForecastGenerator, ForecastVisualizer, save_prediction (largely same) ---
class ForecastGenerator:
    def __init__(self, model, scalers, land_mask_np, climatology_ds):
        self.device = next(model.parameters()).device;
        self.model = model.to(self.device).eval()
        self.land_mask = land_mask_np
        self.scalers = scalers
        self.climatology = climatology_ds

    def generate(self, init_window_tensor):
        self.model.eval()
        with torch.no_grad():
            inputs_model = init_window_tensor.unsqueeze(0).to(self.device).float().contiguous()
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda' and torch.cuda.is_available()):
                pred_anom_norm_batched = self.model(inputs_model)
            pred_anom_norm = pred_anom_norm_batched.squeeze(0).cpu().numpy()
        H, W, T_pred = pred_anom_norm.shape
        restored_anom = np.full_like(pred_anom_norm, np.nan, dtype=np.float32)
        for i in range(H):
            for j in range(W):
                if self.land_mask[i, j] == 1:
                    scaler = self.scalers.get((i, j))
                    if scaler:
                        pixel_preds_norm = pred_anom_norm[i, j, :].reshape(-1, 1)
                        try:
                            restored_anom[i, j, :] = scaler.inverse_transform(pixel_preds_norm).flatten()
                        except Exception:
                            restored_anom[i, j, :] = np.nan
        return restored_anom


class ForecastVisualizer:
    def __init__(self, pred_ds, real_ds, mae, rmse, r2, output_dir_suffix="hybrid_model", model_name_tag=" (Hybrid)"):
        self.pred_ds = pred_ds;
        self.real_ds = real_ds;
        if pred_ds is not None and real_ds is not None and 'sst' in real_ds and 'sst_pred' in pred_ds:
            self.error = real_ds.sst - pred_ds.sst_pred
        else:
            self.error = None
        self.mae, self.rmse, self.r2 = mae, rmse, r2
        self.output_dir = os.path.join("results", f"plots_{output_dir_suffix}");
        self.model_name_tag = model_name_tag;
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"可视化结果将保存到: {os.path.abspath(self.output_dir)}")

    def plot_scatter(self, filename="01_scatter_comparison.png"):
        if self.pred_ds is None or self.real_ds is None: print("散点图: 预测或真实数据为空，跳过。"); return
        plt.figure(figsize=(8, 8));
        pred_flat = self.pred_ds.sst_pred.data.flatten();
        true_flat = self.real_ds.sst.data.flatten()
        valid_idx = ~np.isnan(true_flat) & ~np.isnan(pred_flat);
        pred_flat_valid, true_flat_valid = pred_flat[valid_idx], true_flat[valid_idx]
        if not true_flat_valid.size > 0: print("散点图: 无有效数据点，跳过。"); plt.close(); return
        plt.scatter(true_flat_valid, pred_flat_valid, alpha=0.3, c='#1f77b4', edgecolors='w', linewidths=0.5)
        min_val = min(np.min(true_flat_valid) if true_flat_valid.size > 0 else 0,
                      np.min(pred_flat_valid) if pred_flat_valid.size > 0 else 0)
        max_val = max(np.max(true_flat_valid) if true_flat_valid.size > 0 else 1,
                      np.max(pred_flat_valid) if pred_flat_valid.size > 0 else 1)
        if min_val != max_val: plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        stats_text = (f"MAE = {self.mae:.2f}°C\nRMSE = {self.rmse:.2f}°C\nR² = {self.r2:.2f}");
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        plt.xlabel('真实值 (°C)');
        plt.ylabel('预测值 (°C)');
        plt.title(f'预测值与真实值散点图{self.model_name_tag}');
        plt.grid(alpha=0.3);
        plt.axis('equal');
        plt.tight_layout();
        plt.savefig(os.path.join(self.output_dir, filename));
        plt.close();
        print(f"✅ 散点图: {filename}")

    def plot_spatial_comparison(self, day_idx=0, filename_prefix="02_spatial_comparison"):
        if self.pred_ds is None or self.real_ds is None or self.error is None: print(
            "空间对比图: 数据缺失，跳过。"); return
        if not (0 <= day_idx < len(self.real_ds.time)): print(f"空间对比图: 天数索引 {day_idx} 超出范围。"); return
        plt.figure(figsize=(18, 6));
        real_day_data = self.real_ds.sst.isel(time=day_idx);
        pred_day_data = self.pred_ds.sst_pred.isel(time=day_idx);
        error_day_data = self.error.isel(time=day_idx)
        common_min = np.nanmin([real_day_data.min().item(), pred_day_data.min().item()])
        common_max = np.nanmax([real_day_data.max().item(), pred_day_data.max().item()])
        if np.isnan(common_min) or np.isnan(common_max):
            common_min = np.nanmin(self.real_ds.sst.data);
            common_max = np.nanmax(self.real_ds.sst.data)
        if np.isnan(common_min) or np.isnan(common_max): common_min, common_max = 0, 1
        current_date_dt = pd.to_datetime(self.real_ds.time.isel(time=day_idx).values);
        current_date_str = current_date_dt.strftime('%Y-%m-%d')
        plt.subplot(1, 3, 1);
        pred_day_data.plot.imshow(interpolation='bilinear', cmap='jet', vmin=common_min, vmax=common_max,
                                  add_colorbar=True);
        plt.title(f"预测 - {current_date_str}{self.model_name_tag}")
        plt.subplot(1, 3, 2);
        real_day_data.plot.imshow(interpolation='bilinear', cmap='jet', vmin=common_min, vmax=common_max,
                                  add_colorbar=True);
        plt.title(f"真实 - {current_date_str}")
        plt.subplot(1, 3, 3);
        error_mae_day = np.nanmean(np.abs(error_day_data.data))
        error_clim_max = max(2, abs(error_mae_day) * 2 if not np.isnan(error_mae_day) else 2)
        error_day_data.plot.imshow(interpolation='bilinear', cmap='RdBu_r', vmin=-error_clim_max, vmax=error_clim_max,
                                   add_colorbar=True);
        plt.title(f"误差 - {current_date_str} (MAE: {error_mae_day:.2f}°C)")
        plt.tight_layout();
        filename = f"{filename_prefix}_day{day_idx + 1}_{current_date_str}.png"
        plt.savefig(os.path.join(self.output_dir, filename));
        plt.close()
        print(f"✅ 空间对比图: {filename}")

    def plot_mae_temporal_evolution(self, filename="03a_mae_temporal_error.png"):
        if self.error is None or self.pred_ds is None: print("MAE时间演化图: 数据缺失，跳过。"); return
        time_vals = pd.to_datetime(self.pred_ds.time.values);
        time_labels = time_vals.strftime('%Y-%m-%d')
        mae_temporal = np.nanmean(np.abs(self.error.data), axis=(1, 2))
        plt.figure(figsize=(12, 6));
        plt.plot(time_vals, mae_temporal, 'o-', label='MAE', color='blue');
        for i, txt in enumerate(mae_temporal):
            if not np.isnan(txt): plt.annotate(f"{txt:.2f}", (time_vals[i], mae_temporal[i]),
                                               textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9,
                                               color='blue')
        plt.xticks(ticks=time_vals, labels=time_labels, rotation=45, ha="right");
        plt.ylabel('平均绝对误差 (°C)');
        plt.title(f'MAE 时间演化{self.model_name_tag}');
        plt.legend();
        plt.grid(alpha=0.4);
        plt.tight_layout();
        plt.savefig(os.path.join(self.output_dir, filename));
        plt.close();
        print(f"✅ MAE时间演化图: {filename}")

    def plot_rmse_temporal_evolution(self, filename="03b_rmse_temporal_error.png"):
        if self.error is None or self.pred_ds is None: print("RMSE时间演化图: 数据缺失，跳过。"); return
        time_vals = pd.to_datetime(self.pred_ds.time.values);
        time_labels = time_vals.strftime('%Y-%m-%d')
        rmse_temporal = np.sqrt(np.nanmean(self.error.data ** 2, axis=(1, 2)))
        plt.figure(figsize=(12, 6));
        plt.plot(time_vals, rmse_temporal, 's--', label='RMSE', color='orangered');
        for i, txt in enumerate(rmse_temporal):
            if not np.isnan(txt): plt.annotate(f"{txt:.2f}", (time_vals[i], rmse_temporal[i]),
                                               textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9,
                                               color='orangered')
        plt.xticks(ticks=time_vals, labels=time_labels, rotation=45, ha="right");
        plt.ylabel('均方根误差 (°C)');
        plt.title(f'RMSE 时间演化{self.model_name_tag}');
        plt.legend();
        plt.grid(alpha=0.4);
        plt.tight_layout();
        plt.savefig(os.path.join(self.output_dir, filename));
        plt.close();
        print(f"✅ RMSE时间演化图: {filename}")

    def plot_observation_series(self, filename="04_observation_point_series.png"):
        if self.pred_ds is None or self.real_ds is None: print("观测点序列图: 数据缺失，跳过。"); return
        try:
            time_coords_plot = pd.to_datetime(self.real_ds.time.values)
        except:
            print("观测点序列图: 处理时间坐标失败。"); return
        if len(time_coords_plot) == 0: print("观测点序列图: 时间坐标为空。"); return
        plt.figure(figsize=(14, 7));
        colors = plt.cm.get_cmap('tab10', len(OBS_POINTS) * 2)
        for idx, (lat, lon) in enumerate(OBS_POINTS):
            try:
                real_series = self.real_ds.sst.sel(lat=lat, lon=lon, method='nearest')
                pred_series = self.pred_ds.sst_pred.sel(time=real_series.time, lat=lat, lon=lon, method='nearest')
                plt.plot(time_coords_plot, real_series.data, 'o-', color=colors(idx * 2),
                         label=f'真实 ({lat}°N,{lon}°E)', ms=5)
                plt.plot(time_coords_plot, pred_series.data, 'x--', color=colors(idx * 2 + 1),
                         label=f'预测 ({lat}°N,{lon}°E){self.model_name_tag}', ms=5, alpha=0.8)
                for j, val_r in enumerate(real_series.data):
                    if not np.isnan(val_r): plt.annotate(f"{val_r:.1f}", (time_coords_plot[j], val_r),
                                                         xytext=(-2, 8 if j % 2 == 0 else 10),
                                                         textcoords="offset points", ha='right', va='bottom',
                                                         fontsize=7, color=colors(idx * 2))
                for j, val_p in enumerate(pred_series.data):
                    if not np.isnan(val_p): plt.annotate(f"{val_p:.1f}", (time_coords_plot[j], val_p),
                                                         xytext=(2, -12 if j % 2 == 0 else -15),
                                                         textcoords="offset points", ha='left', va='top', fontsize=7,
                                                         color=colors(idx * 2 + 1), fontstyle='italic', alpha=0.9,
                                                         path_effects=[
                                                             plt.matplotlib.patheffects.withStroke(linewidth=0.5,
                                                                                                   foreground='w')])
            except Exception as e:
                print(f"观测点 ({lat},{lon}) 绘图失败: {e}")
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=min(10, len(time_coords_plot)), prune='both'))
        plt.xticks(rotation=45, ha="right");
        plt.ylabel('海表温度 (°C)');
        plt.title(f'观测点温度时序对比{self.model_name_tag}');
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.);
        plt.grid(alpha=0.4);
        plt.tight_layout(rect=[0, 0, 0.85, 1]);
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight');
        plt.close()
        print(f"✅ 观测点对比图: {filename}")


def save_prediction(pred_ds, filename_base, format='netcdf'):
    dir_path = os.path.dirname(os.path.abspath(filename_base));
    os.makedirs(dir_path, exist_ok=True)
    actual_filename = filename_base if filename_base.endswith(f".{format}") else f"{filename_base}.{format}"
    try:
        if format == 'netcdf':
            if os.path.exists(actual_filename): os.remove(actual_filename)
            pred_ds.to_netcdf(actual_filename, engine='h5netcdf', invalid_netcdf=True)
        elif format == 'pickle':
            pd.to_pickle(pred_ds, actual_filename)
        print(f"✅ {format.upper()}文件保存成功: {os.path.abspath(actual_filename)}")
    except Exception as e:
        print(f"🆘 保存为 {format.upper()} 失败：{e}")


def custom_collate_fn(batch):
    inputs_list, targets_list, dates_list = [], [], []
    for item in batch:
        inputs_list.append(item[0])
        targets_list.append(item[1])
        dates_list.append(item[2])
    collated_inputs = torch.stack(inputs_list, dim=0)
    collated_targets = torch.stack(targets_list, dim=0)
    return collated_inputs, collated_targets, dates_list


# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"❌ 主程序错误: 数据文件 {DATA_PATH} 未找到。");
        exit()

    print("🚀 SwinV2-ConvLSTM Hybrid Model (完整预处理) 实验开始...")
    processor = CopernicusDataProcessor(DATA_PATH)
    train_data_tensor, val_data_tensor, test_data_tensor = processor.preprocess()

    train_times = processor.ds.sel(time=slice(*processor.splits['train'])).time.values
    val_times = processor.ds.sel(time=slice(*processor.splits['val'])).time.values
    test_times = processor.ds.sel(time=slice(*processor.splits['test'])).time.values

    train_dataset = SSTWindowDataset(train_data_tensor, WINDOW_SIZE, PRED_STEPS, train_times)
    val_dataset = SSTWindowDataset(val_data_tensor, WINDOW_SIZE, PRED_STEPS, val_times)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available(), drop_last=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS,
                            pin_memory=torch.cuda.is_available(), drop_last=False, collate_fn=custom_collate_fn)

    print("\n🧠 初始化 SwinV2-ConvLSTM Hybrid 模型...")
    input_feature_dim_actual = train_data_tensor.shape[-1]
    print(f"从数据推断的实际输入特征维度 (F_in): {input_feature_dim_actual}")

    hybrid_model = SwinConvLSTMModel(
        input_feature_dim=input_feature_dim_actual,
        spatial_dims=SPATIAL_DIMS,
        pred_steps=PRED_STEPS,
        swin_model_name=SWIN_MODEL_NAME,
        swin_img_size=SWIN_ENCODER_IMG_SIZE,
        swin_in_chans=SWIN_IN_CHANS_FOR_HYBRID,
        swin_drop_rate=SWINV2_DROP_RATE,
        swin_drop_path_rate=SWINV2_DROP_PATH_RATE,
        explicit_convlstm_input_dim=EXPLICIT_CONVLSTM_INPUT_DIM,
        convlstm_hidden_dims=CONVLSTM_HIDDEN_DIMS,
        convlstm_kernel_sizes=CONVLSTM_KERNEL_SIZES,
        convlstm_num_layers=CONVLSTM_NUM_LAYERS,
        decoder_channels_list=HYBRID_DECODER_CHANNELS,
        dropout_rate=DROPOUT
    )
    print(
        f"SwinV2-ConvLSTM Hybrid 模型总参数量: {sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad):,}")

    land_mask_np = processor.land_mask

    trainer_hybrid = ModelTrainer(hybrid_model, train_loader, val_loader, processor.scalers,
                                  land_mask=land_mask_np, model_name="SwinConvLSTM_Hybrid")

    print("\n🏋️ 开始/继续模型训练 (SwinV2-ConvLSTM Hybrid)...")
    trained_hybrid_model = trainer_hybrid.run_training()

    print("\n🔮 使用训练好的 Hybrid 模型生成7天预测...")
    if test_data_tensor.shape[2] < WINDOW_SIZE:
        raise ValueError(f"测试数据时间维度 ({test_data_tensor.shape[2]}) 过短。")

    all_test_dates_pd = pd.to_datetime(test_times)
    target_end_date_for_input_window = pd.to_datetime('2021-12-24')

    try:
        candidate_indices = np.where(all_test_dates_pd <= target_end_date_for_input_window)[0]
        if not candidate_indices.size:
            print(f"警告: 目标日期 {target_end_date_for_input_window.strftime('%Y-%m-%d')} 未找到。用测试集末尾。")
            end_idx_in_test_tensor_time = len(all_test_dates_pd) - 1
        else:
            end_idx_in_test_tensor_time = candidate_indices[-1]
    except Exception:
        print(f"警告: 查找目标日期时出错。用测试集末尾。")
        end_idx_in_test_tensor_time = len(all_test_dates_pd) - 1

    start_idx_in_test_tensor_time = end_idx_in_test_tensor_time - WINDOW_SIZE + 1
    if start_idx_in_test_tensor_time < 0:
        raise ValueError(f"无法构建输入窗口，开始索引 ({start_idx_in_test_tensor_time}) <0。")

    print(
        f"预测输入窗口 (test_data_tensor time indices): {start_idx_in_test_tensor_time} to {end_idx_in_test_tensor_time}")
    print(
        f"对应日期: {all_test_dates_pd[start_idx_in_test_tensor_time].strftime('%Y-%m-%d')} to {all_test_dates_pd[end_idx_in_test_tensor_time].strftime('%Y-%m-%d')}")

    test_window_tensor = test_data_tensor[:, :, start_idx_in_test_tensor_time: end_idx_in_test_tensor_time + 1, :]
    print(f"选取的预测输入窗口形状: {test_window_tensor.shape}")

    generator_hybrid = ForecastGenerator(trained_hybrid_model, processor.scalers, land_mask_np, processor.climatology)
    pred_anom_hybrid_hw_t = generator_hybrid.generate(test_window_tensor)
    pred_anom_hybrid_t_hw = pred_anom_hybrid_hw_t.transpose(2, 0, 1)

    actual_forecast_start_date = all_test_dates_pd[end_idx_in_test_tensor_time] + pd.Timedelta(days=1)
    max_original_date = pd.to_datetime(processor.times.max());
    test_split_end_date_pd = pd.to_datetime(processor.splits['test'][1])
    effective_end_limit = min(max_original_date, test_split_end_date_pd)
    forecast_dates_actual = pd.date_range(start=actual_forecast_start_date, periods=PRED_STEPS, freq='D')
    forecast_dates_actual = forecast_dates_actual[forecast_dates_actual <= effective_end_limit]

    if len(forecast_dates_actual) == 0:
        print("❌ 错误: 无法生成任何有效的预测日期。");
        exit()
    if len(forecast_dates_actual) < PRED_STEPS:
        print(f"警告: 实际预测日期 ({len(forecast_dates_actual)}) < PRED_STEPS ({PRED_STEPS}).")
        pred_anom_hybrid_t_hw = pred_anom_hybrid_t_hw[:len(forecast_dates_actual), :, :]

    clim_list_hybrid = []
    for doy_val in forecast_dates_actual.dayofyear.values:
        try:
            actual_doy_to_use = int(doy_val)
            if actual_doy_to_use not in processor.climatology.dayofyear.data:
                actual_doy_to_use = 365 if doy_val == 366 and 365 in processor.climatology.dayofyear.data else \
                    (processor.climatology.dayofyear.min().item() if processor.climatology.dayofyear.size > 0 else 1)
            clim_slice = processor.climatology.sel(dayofyear=actual_doy_to_use).data
            clim_list_hybrid.append(clim_slice)
        except Exception as e_clim:
            print(f"获取气候态失败 for doy {doy_val}: {e_clim}, 使用0填充")
            clim_list_hybrid.append(np.zeros((SPATIAL_DIMS[0], SPATIAL_DIMS[1])))
    if not clim_list_hybrid: print("❌ 错误: 未能为预测日期生成气候态数据。"); exit()
    clim_arr_hybrid_t_hw = np.stack(clim_list_hybrid, axis=0)
    if pred_anom_hybrid_t_hw.shape[0] != clim_arr_hybrid_t_hw.shape[0]:
        clim_arr_hybrid_t_hw = clim_arr_hybrid_t_hw[:pred_anom_hybrid_t_hw.shape[0], :, :]
    pred_sst_hybrid_t_hw = pred_anom_hybrid_t_hw + clim_arr_hybrid_t_hw

    pred_ds_hybrid = xr.Dataset(
        {'sst_pred': (['time', 'lat', 'lon'], pred_sst_hybrid_t_hw)},
        coords={'lat': processor.lats, 'lon': processor.lons, 'time': forecast_dates_actual}
    )
    test_real_hybrid = processor.original_sst.sel(time=forecast_dates_actual, lat=processor.lats, lon=processor.lons,
                                                  method='nearest').to_dataset(name='sst').load()

    ocean_mask_broadcast = np.broadcast_to(land_mask_np[None, ...] == 1, pred_sst_hybrid_t_hw.shape)
    pred_values_ocean = pred_sst_hybrid_t_hw[ocean_mask_broadcast]
    true_values_ocean = test_real_hybrid.sst.transpose('time', 'lat', 'lon').data[ocean_mask_broadcast]
    valid_mask = ~np.isnan(true_values_ocean) & ~np.isnan(pred_values_ocean) & \
                 ~np.isinf(true_values_ocean) & ~np.isinf(pred_values_ocean)
    mae_hybrid, rmse_hybrid, r2_hybrid = np.nan, np.nan, np.nan
    if np.sum(valid_mask) > 1:
        pred_valid = pred_values_ocean[valid_mask];
        true_valid = true_values_ocean[valid_mask]
        mae_hybrid = np.mean(np.abs(pred_valid - true_valid))
        rmse_hybrid = np.sqrt(np.mean((true_valid - pred_valid) ** 2))
        ss_res = np.sum((true_valid - pred_valid) ** 2);
        ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
        r2_hybrid = 1 - (ss_res / (ss_tot + 1e-9)) if ss_tot > 1e-9 else 0.0
    else:
        print("警告: 计算评估指标的有效数据点不足。")

    print("\n📊 SwinV2-ConvLSTM Hybrid 模型评估结果:")
    print(f"  - MAE: {mae_hybrid:.2f}°C");
    print(f"  - RMSE: {rmse_hybrid:.2f}°C");
    print(f"  - R²: {r2_hybrid:.2f}")

    visualizer_hybrid = ForecastVisualizer(pred_ds_hybrid, test_real_hybrid, mae_hybrid, rmse_hybrid, r2_hybrid,
                                           output_dir_suffix="hybrid_swin_convlstm",
                                           model_name_tag=" (Hybrid Swin-ConvLSTM)")
    if not (np.isnan(mae_hybrid) or np.isnan(rmse_hybrid) or np.isnan(r2_hybrid)):
        visualizer_hybrid.plot_scatter(filename="hybrid_01_scatter.png")
        if len(forecast_dates_actual) > 0:
            visualizer_hybrid.plot_spatial_comparison(day_idx=0, filename_prefix="hybrid_02_spatial")
            if len(forecast_dates_actual) > 1:
                visualizer_hybrid.plot_spatial_comparison(day_idx=len(forecast_dates_actual) - 1,
                                                          filename_prefix="hybrid_02_spatial_lastday")
        visualizer_hybrid.plot_mae_temporal_evolution(filename="hybrid_03a_mae_temporal.png")
        visualizer_hybrid.plot_rmse_temporal_evolution(filename="hybrid_03b_rmse_temporal.png")
        visualizer_hybrid.plot_observation_series(filename="hybrid_04_observation_series.png")
    else:
        print("由于评估指标无效，部分可视化步骤已跳过。")

    if len(forecast_dates_actual) > 0:
        save_prediction(pred_ds_hybrid, "results/hybrid_swin_convlstm_forecast", format='netcdf')
    else:
        print("没有预测结果可保存。")
    print("SwinV2-ConvLSTM Hybrid 模型实验完成！")
