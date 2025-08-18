import numpy as np
import pandas as pd
import scipy.ndimage
import pywt
import statsmodels.api as sm
from scipy import signal
import wfdb
from scipy.signal import resample_poly
from helper_code import *

# 정규화 함수 딕셔너리
normalization_methods = {
    "mean": lambda sig: (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-5),
    "minmax": lambda sig: (sig - np.min(sig, axis=0)) / (np.max(sig, axis=0) - np.min(sig, axis=0) + 1e-5),
    "median": lambda sig: (sig - np.median(sig, axis=0)) / (np.std(sig, axis=0) + 1e-5),
    "none": lambda sig: sig
}

def rolling_method(sig, window_size=45):
    signal_df = pd.DataFrame(sig)
    rolling_avg_12 = signal_df.rolling(window=window_size).mean()
    for i in range(sig.shape[1]):
        signal_df.iloc[:, i] -= rolling_avg_12.iloc[:, i]
    return signal_df.to_numpy()

def lowess_method(sig, frac=0.03):
    residuals = np.zeros(sig.shape)
    for i in range(sig.shape[1]):
        smoothed = sm.nonparametric.lowess(sig[:, i], np.arange(len(sig[:, i])), frac=frac)[:, 1]
        residuals[:, i] = sig[:, i] - smoothed
    return residuals

def wavelet_denoising(signal, wavelet='db6', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def oc_co_filter(signal, fs, k_factor):
    k_size = int(np.round(k_factor * fs))
    struct_elem = np.ones(k_size)
    opened = scipy.ndimage.grey_opening(signal, structure=struct_elem)
    closed = scipy.ndimage.grey_closing(opened, structure=struct_elem)
    closed_opened = scipy.ndimage.grey_opening(closed, structure=struct_elem)
    return closed_opened

def oc_co(sig, fs=400):
    num_leads = sig.shape[1]
    return_dat = np.zeros_like(sig)
    for i in range(num_leads):
        denoised = wavelet_denoising(sig[:, i])
        fb = oc_co_filter(denoised, fs, 0.11)
        fc = oc_co_filter(fb, fs, 0.27)
        return_dat[:, i] = denoised - fc
    return return_dat

def oc_co_wo_let(sig, fs=400):
    num_leads = sig.shape[1]
    return_dat = np.zeros_like(sig)
    for i in range(num_leads):
        fb = oc_co_filter(sig[:, i], fs, 0.11)
        fc = oc_co_filter(fb, fs, 0.27)
        return_dat[:, i] = sig[:, i] - fc
    return return_dat

def bandpass_filtering(recording, low= 1, high= 47):
    b, a = signal.butter(3, [low / 250, high / 250], 'bandpass')
    bp_filtered_recording = signal.filtfilt(b, a, recording, axis= 0)
    return bp_filtered_recording

def butter_method(sig, low=1, high=47):
    return bandpass_filtering(sig, low, high)


preprocessing_methods = {
    "rolling": rolling_method,
    "lowess": lowess_method,
    "oc_co": lambda sig: oc_co(sig, fs=400),
    "oc_co_wo_let": lambda sig: oc_co_wo_let(sig, fs=400),
    "butter": butter_method,
    "none": lambda sig: sig
}

def process_pipeline(signal, mod="median", method="rolling"):
    standardized_signal = normalization_methods[mod](signal)
    
    if np.isnan(standardized_signal).any():
        print(f"[Pipeline NaN] NaN after normalization, mod={mod}")
        
    processed_signal = preprocessing_methods[method](standardized_signal)
    
    if np.isnan(processed_signal).any():
        print(f"[Pipeline NaN] NaN after preprocessing, method={method}")
        
    return processed_signal

def load_signals_and_resample(file_path, target_fs=400):
    signal, fields = wfdb.rdsamp(file_path)
    orig_fs = fields['fs']
    
    if orig_fs != target_fs:
        signal = resample_poly(signal, up=target_fs, down=orig_fs)
    
    return signal, fields

def get_source2(string):
    source, has_source = get_variable(string, source_string)
    if not has_source:
        source = ''
    else :
        if source == 'SaMi-Trop':
            source = 'samitrop'
        elif source == 'PTB-XL':
            source = 'ptbxl'
        elif source == 'CODE-15%':
            source = 'code15'
        source = source
    return source


import numpy as np
import neurokit2 as nk

def compute_avg_qrs(signal, sr=1000):
    used_lead = 1   # Lead II
    ecg = signal[used_lead, :]
    try:
        _, info = nk.ecg_peaks(ecg, sampling_rate=sr)
        peaks = info["ECG_R_Peaks"]
    except:
        peaks = []

    avg_radius_sec = 0.6
    avg_samps = int(avg_radius_sec * sr * 2)
    avg_qrs = np.zeros((avg_samps, signal.shape[0]))
    num_used_peaks = 0

    for i in peaks:
        ss = i - avg_samps // 2
        es = i + avg_samps // 2
        if ss < 0 or es > ecg.shape[0]:
            continue
        avg_qrs += signal[:, ss:es].T
        num_used_peaks += 1

    if num_used_peaks > 0:
        avg_qrs /= num_used_peaks

    avg_qrs = avg_qrs.T
    return avg_qrs

def lead12toXYZ(leadI, leadII, leadV1, leadV2, leadV3, leadV4, leadV5, leadV6):
    X = 0.38*leadI - 0.07*leadII - 0.13*leadV1 + 0.05*leadV2 - 0.01*leadV3 + 0.14*leadV4 + 0.06*leadV5 + 0.54*leadV6
    Y = -0.07*leadI + 0.93*leadII + 0.06*leadV1 - 0.02*leadV2 - 0.05*leadV3 + 0.06*leadV4 - 0.17*leadV5 + 0.13*leadV6
    Z = 0.11*leadI - 0.23*leadII - 0.43*leadV1 - 0.06*leadV2 - 0.14*leadV3 - 0.20*leadV4 - 0.11*leadV5 + 0.31*leadV6
    return X, Y, Z

def compute_vcg_features(X, Y, Z, fs=1000):
    lng = len(X)
    sr = int(fs * 0.2)
    samplesFrom = int(lng/2 - sr)
    sampleTo = int(lng/2 + sr)
    Xc = X[samplesFrom:sampleTo]
    Yc = Y[samplesFrom:sampleTo]
    Zc = Z[samplesFrom:sampleTo]
    duration = (sampleTo - samplesFrom) / fs
    Xbase = Xc - Xc[0]
    Ybase = Yc - Yc[0]
    Zbase = Zc - Zc[0]
    sx = np.sum(np.abs(Xbase)) / fs
    sy = np.sum(np.abs(Ybase)) / fs
    sz = np.sum(np.abs(Zbase)) / fs
    areaX = sx * sx
    areaY = sy * sy
    areaZ = sz * sz
    total_area = np.sqrt(areaX + areaY + areaZ)
    feats = np.array([
        duration,
        total_area,
        areaX,
        areaY,
        areaZ
    ], dtype=np.float32)
    return feats

def extract_handcrafted_features(signal, sr):
    try:
        avg_qrs = compute_avg_qrs(signal, sr)
        I = avg_qrs[0, :]
        II = avg_qrs[1, :]
        V1 = avg_qrs[6, :]
        V2 = avg_qrs[7, :]
        V3 = avg_qrs[8, :]
        V4 = avg_qrs[9, :]
        V5 = avg_qrs[10, :]
        V6 = avg_qrs[11, :]
        X, Y, Z = lead12toXYZ(I, II, V1, V2, V3, V4, V5, V6)
        feats = compute_vcg_features(X, Y, Z, sr)
    except Exception as e:
        print(f"[VCG feature error] {e}")
        feats = np.zeros(5, dtype=np.float32)
    return feats


##### extract other nadcrafted features

def fast_qrs_delineate(signal, rpeaks, sampling_rate):
    """빠르게 QRS onset/offset 추정하는 간단한 함수."""
    if isinstance(rpeaks, dict):
        rpeaks = rpeaks["ECG_R_Peaks"]

    onsets = []
    offsets = []
    for r in rpeaks:
        window = int(0.2 * sampling_rate)  # ±200ms window
        start = max(0, r - window)
        end = min(len(signal), r + window)
        segment = signal[start:end]

        # 미분 및 임계값 설정
        diff = np.diff(segment)
        threshold = 1.2*np.std(diff)

        # Onset 추정
        onset_idx = np.argmax(diff > threshold)
        onset = onset_idx + start if onset_idx > 0 else r
        onset = min(onset, r-1)

        # Offset 추정
        offset_idx = np.argmax(diff[::-1] < -threshold)
        offset = end - offset_idx if offset_idx > 0 else r
        offset = max(r+1, offset)

        onsets.append(onset)
        offsets.append(offset)

    return onsets, offsets

from scipy.signal import butter, filtfilt
import scipy.signal as sig

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

def extract_handcrafted_features2(ecg, fs, rm_noise = True):

    # 리드 인덱스
    lead_I, lead_II, lead_aVF = ecg[:, 0], ecg[:, 1], ecg[:, 5]
    lead_V1, lead_V6 = ecg[:, 6], ecg[:, 11]

    if rm_noise:
        try:
            lead_II = bandpass_filter(lead_II, fs, lowcut=0.5, highcut=40.0)
        except:
            return np.zeros(4)

    # =============================================================================
    # 1. R-peak + 파형 경계 추출 (한 줄)
    # =============================================================================
#    _, rpeaks_info = nk.ecg_peaks(lead_II, sampling_rate=fs)

    try:
        _, rpeaks_info = nk.ecg_peaks(lead_II, sampling_rate=fs)
    except:
        return np.zeros(4)

    if len(rpeaks_info['ECG_R_Peaks']) <= 1 :
#        print(f"Warning: Not enough R-peaks found in {fnm}. Skipping feature extraction.")
        return np.zeros(4) 

    rpeaks = rpeaks_info["ECG_R_Peaks"]
    rpeaks_fixed = nk.signal_fixpeaks(rpeaks, sampling_rate=fs)[1]
    rpeaks_fixed = [int(x) for x in rpeaks_fixed if not np.isnan(x)]
    rr_sec = np.diff(rpeaks_fixed) / fs
    r_onsets, r_offsets = fast_qrs_delineate(lead_II, rpeaks_fixed, sampling_rate=fs)



#    rpeaks      = info["ECG_R_Peaks"]                 # 모든 R 인덱스
#    r_onsets    = info["ECG_R_Onsets"]                # QRS 시작 (NaN 가능)
#    r_offsets   = info["ECG_R_Offsets"]               # QRS 종료 (NaN 가능)
#    rr_sec      = info["ECG_fixpeaks_rr"]             # Kubios 교정 RR [s]

#    print(f"ecg_process Elapsed time: {time.time() - start_time:.4f} seconds")
   
    # -----------------------------------------------------------------------------
    # 헬퍼: index 리스트 + fs 를 msec 로 변환 (NaN 제외)
    # -----------------------------------------------------------------------------
    def idx_to_ms(on, off):
        if np.isnan(on) or np.isnan(off):
            return np.nan
        return (off - on) / fs * 1_000

    # =============================================================================
    # 2-A. RBBB ± LAFB 피처 --------------------------------------------------------
    # =============================================================================
    # ① QRS Width (ms)  ──────
    qrs_ms = [idx_to_ms(on, off) for on, off in zip(r_onsets, r_offsets)]

    # ② Mean QRS Axis (deg)  ── (I & aVF 적분 → arctan2)
    def mean_qrs_axis(lead_I, lead_aVF, r_on, r_off):
        areas_I, areas_aVF = [], []
        for on, off in zip(r_on, r_off):
            if np.isnan(on) or np.isnan(off):
                continue
            on, off = int(on), int(off)
            areas_I  .append(np.trapz(lead_I  [on:off], dx=1/fs))
            areas_aVF.append(np.trapz(lead_aVF[on:off], dx=1/fs))
        V_I, V_aVF = np.mean(areas_I), np.mean(areas_aVF)
        return np.degrees(np.arctan2(V_aVF, V_I))

    axis_deg = mean_qrs_axis(lead_I, lead_aVF, r_onsets, r_offsets)

    # ③ RSR′(V1) 비율  ───────
    def rsr_flag(signal, on, off, min_amp=0.15):
        if np.isnan(on) or np.isnan(off):
            return False
        seg = signal[int(on):int(off)]
        peaks, _ = sig.find_peaks(seg, height=min_amp, distance=0.02*fs)
        return len(peaks) >= 2                      # 두 개 이상 +피크 → RSR′

    rsr_ratio = np.mean([rsr_flag(lead_V1, on, off) for on, off in zip(r_onsets, r_offsets)])

    # ④ wide-S(I/V6) 비율  ─────
    def wideS_flag(signal, on, off, amp_th=-0.10, dur_th=0.04):
        if np.isnan(on) or np.isnan(off):
            return False
        seg   = signal[int(on):int(off)]
        s_idx = np.argmin(seg)
        if seg[s_idx] >= amp_th:                    # 깊이 기준 미통과
            return False
        zero  = np.where(seg[s_idx:] > 0)[0]
        width = zero[0]/fs if len(zero) else 0
        return width >= dur_th                     # 폭 ≥ 40 ms ?

    wideS_ratio = np.mean([
        wideS_flag(lead_I,  on, off) or
        wideS_flag(lead_V6, on, off)
        for on, off in zip(r_onsets, r_offsets)
    ])

    feat_RBBB = np.array( [(np.nanmedian(qrs_ms) - 50.5) / 16 ,  (axis_deg -14.8) / 62,
        (rsr_ratio - 0.01) / 0.07,
            (wideS_ratio - 0.004) / 0.04 ] )

    return feat_RBBB





# Reorder channels in signal.
def reorder_signal(input_signal, input_channels, output_channels):
    # Do not allow repeated channels with potentially different values in a signal.
    assert(len(set(input_channels)) == len(input_channels))
    assert(len(set(output_channels)) == len(output_channels))

    if input_channels == output_channels:
        output_signal = input_signal
    else:
        output_channels = normalize_names(input_channels, output_channels)

        input_signal = np.asarray(input_signal)
        num_samples = np.shape(input_signal)[0]
        num_channels = len(output_channels)
        data_type = input_signal.dtype

        output_signal = np.zeros((num_samples, num_channels), dtype=data_type)
        for i, output_channel in enumerate(output_channels):
            for j, input_channel in enumerate(input_channels):
                if input_channel == output_channel:
                    output_signal[:, i] = input_signal[:, j]

    return output_signal