#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from scipy.stats import pearsonr
from apicultor.constraints.dynamic_range import *
from apicultor.utils.algorithms import *
from apicultor.machine_learning.lstm_synth_w import *
from apicultor.sonification.Sonification import normalize, write_file
import numpy as np
import asyncio
from pathos.pools import ProcessPool as Pool
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import lfilter, fftconvolve, firwin, medfilt
from soundfile import read
import os
import sys
import logging
import librosa
import warnings
# TODO: *Remove wows, clippings, pops

# you should comment what you've already processed (avoid over-processing)
warnings.simplefilter("ignore", RuntimeWarning)
logging.basicConfig(level=logging.DEBUG)
logging.disable(logging.WARNING)
logger = logging.getLogger(__name__)


def energy(mag): return np.sum((10 ** (mag / 20)) ** 2)

def dec(at,scalar): return at * (500*scalar/100)  # calculate decay time based on a maximum scalar

def rel(at,scalar): return at * (1000*scalar/100)  # calculate release time based on a maximum scalar


# convert attack and release time to coefficients
def to_coef(at, sr): return np.exp((np.log(9)*-1) / (sr * at))


def width_interpolation(w_idx, size):
    w_interp = []
    for i in range(size):
        w_interp.append((w_idx[i]*(1-((np.sin(2*np.pi*1/3)+1)/2.0))
                         ) + (w_idx[i-1]*((np.sin(2*np.pi*1/3)+1)/2.0)))
    return w_interp


def lstm_synth_predict(audio, predict_from_bin=None):
    if predict_from_bin == True:
        w = np.load('width.npy')
        w_inter = width_interpolation(w, 1)
    else:
        w_inter = lstmw
    stft = librosa.stft(audio, hop_length=1024, win_length=2048).T
    output = librosa.istft((stft*w_inter).T, hop_length=1024)
    return output


def centBin2freq(cent, reff, binsInOctave):
    pow(2, (cent - reff) / binsInOctave)


def freq2CentBin(freq): return np.floor(np.log2(freq))


def hum_removal(song):
    _outSampleRate = 2000
    psd = []
    for frame in song.FrameGenerator():
        psd.append(periodogram(song.frame, song.fs,
                   'flattop', scaling='spectrum')[1][:song.N])
    psd = np.array(psd)
    _medianFilterSize = song.frame.size * 60 / (_outSampleRate)
    _spectSize = psd[0].size

    _timeStamps = len(psd)
    if _timeStamps < 10:
        _timeWindow = int(_timeStamps/2)
    else:
        _timeWindow = 10
    psdWindow = np.zeros(shape=(int(_spectSize), 10))
    _medianFilterSize = .439
    binsToSkip = 6
    binResolution = 20
    _binsInOctave = 1200.0 / binResolution
    pitchContinuity = (binsToSkip / _binsInOctave) * \
        1200. / (1000. * song.H / _outSampleRate)

    _Q0 = .1
    _Q1 = .55
    _Q0sample = _Q0 * _timeWindow + 0.5
    _Q1sample = _Q1 * _timeWindow + 0.5
    _iterations = _timeStamps - _timeWindow + 1
    r = np.zeros(shape=(int(_spectSize), int(_iterations)))
    _EPS = np.finfo(float).eps
    for i in range(_spectSize):
        for j in range(_timeWindow):
            psdWindow[i][j] = psd[j][i]
        psdIdxs = np.argsort(psdWindow[i])
        Q0 = psdWindow[i][psdIdxs[int(_Q0sample)]]
        Q1 = psdWindow[i][psdIdxs[int(_Q1sample)]]
        r[i][0] = Q0 / (Q1 + _EPS)

    for i in range(_spectSize):
        for j in range(_timeWindow, _timeStamps):
            psdWindow[i] = np.roll(psdWindow[i], -1)
            psdWindow[i][_timeWindow - 1] = psd[j][i]
            psdIdxs = np.argsort(psdWindow[i])
            Q0 = psdWindow[i][psdIdxs[round(_Q0sample)]]
            Q1 = psdWindow[i][psdIdxs[round(_Q1sample)]]
            r[i][j - _timeWindow + 1] = Q0 / (Q1 + _EPS)

    rSpec = np.zeros(_spectSize)

    for j in range(_iterations):
        for i in range(_spectSize):
            rSpec[i] = r[i][j]
            filtered = medfilt(rSpec, 7)
            for m in range(_spectSize):
                r[m][j] -= filtered[m]

    kernerSize = min(int(_timeWindow / 2), _iterations)
    kernerSize -= (kernerSize + 1) % 2

    for i in range(_spectSize):
        filtered = medfilt(r[i], kernerSize)
        for j in range(_iterations):
            r[i][j] = filtered[j]

    frames = list(song.FrameGenerator())
    bins = []
    values = []
    for j in range(_iterations):
        for i in range(_spectSize):
            rSpec[i] = r[i][j]
        threshold = 5 * np.std(rSpec)
        song.frame = frames[j]
        song.window()
        song.Spectrum()
        song.spectral_peaks()
        try:
            song.pitch_salience_function(threshold)
            song.pitch_salience_function_peaks(threshold)
            bins.append(song.salience_bins)
            values.append(song.salience_values)
        except Exception as e:
            print(e)

    song.salience_bins = np.array(bins)
    song.salience_values = np.array(values)

    song.pitch_contours()
    timeWindowSecs = _timeWindow * song.H / _outSampleRate
    for i in range(len(song.contours_bins)):
        song._contours_starts[i] += timeWindowSecs
        song.contours_freqs_mean[i] = centBin2freq(
            np.mean(song.contours_bins[i]), _referenceTerm, _binsInOctave)
        song.contours_saliences_mean[i] = np.mean(song.contours_saliences[i])
        song.contours_ends[i] = song.contours_starts[i] + \
            song.contours_saliences[i].size * song.H / _outSampleRate

    return song.contours_freqs_mean


def vecvec2Array(v):
    v2D = np.zeros(shape=(v.shape[0], v.shape[1]))
    for i in range(v2D[:, 0]):
        for j in range(v2D[:, 1]):
            v2D[i][j] = v[i][j]
    return v2D


def discontinuity_detector(song):
    predicted = np.zeros(song.H)
    y = []
    frames = []
    errors = []
    errors_filt = []
    samples_peaking_frame = []
    frame_idx = []
    power = []
    frame_counter = 0
    energy_thld = 0.001
    times_thld = 8
    sub_frame = 32

    for frame in song.FrameGenerator():
        song.window()
        power.append(np.sqrt(np.mean(song.windowed_x ** 2)))
        frames.append(song.windowed_x)
        frame_un = np.array(song.windowed_x[song.H // 2: song.H * 3 // 2])
        norm = np.max(np.abs(song.windowed_x))
        if not norm:
            continue
        song.windowed_x /= norm

        lpc_f = song.LPC()

        lpc_f1 = lpc_f[::-1]

        for idx, i in enumerate(range(song.H // 2, song.H * 3 // 2)):
            predicted[idx] = - \
                np.sum(np.multiply(song.windowed_x[i - 14:i], lpc_f1))

        error = np.abs(
            song.windowed_x[song.H // 2: song.H * 3 // 2] - predicted)

        threshold1 = 8 * np.std(error)

        med_filter = medfilt(error, kernel_size=7)
        filtered = np.abs(med_filter - error)

        mask = []
        for i in range(0, len(error), sub_frame):
            r = np.sqrt(np.mean(frame_un[i:i + sub_frame]**2)) > energy_thld
            mask += [r] * sub_frame
        mask = mask[:len(error)]
        mask = np.array([mask]).astype(float)[0]

        if sum(mask) == 0:
            threshold2 = 1000  # just skip silent frames
        else:
            threshold2 = times_thld * (np.std(error[mask.astype(bool)]) +
                                       np.median(error[mask.astype(bool)]))

        threshold = np.max([threshold1, threshold2])

        samples_peaking = np.sum(filtered >= threshold)
        if samples_peaking >= 1:
            y.append(frame_counter * song.H / 44100.)
            frame_idx.append(frame_counter)

        frames.append(song.windowed_x)
        errors.append(error)
        errors_filt.append(filtered)
        samples_peaking_frame.append(samples_peaking)

        frame_counter += 1
    return np.array(y)


def derivative(array):
    output = np.zeros(array.size)
    output[0] = array[0]
    for i in range(len(array)):
        output[i] = array[i] - array[i-1]
    return output


def false_stereo(frame):
    try:
        frame.shape[1] == 2
    except Exception as e:
        raise IndexError("Signal is not stereo")
    silenceThreshold = 10 ** (-50 * 0.05)
    if frame[0] < silenceThreshold and frame[1] < silenceThreshold:
        raise ValueError(
            "Can't determine if a silent signal has false stereo!")
    r_thresh = 0.9995
    r = pearsonr(frame[0], frame[1])
    if r > r_thresh:
        return True
    else:
        return False


def noise_burst_detector(song):
    _thresholdCoeff = 8
    silenceThreshold = 10 ** (-50 * 0.05)
    _alpha = .9

    if np.sqrt(np.mean(song.frame**2)) < silenceThreshold:
        return

    second_derivative = derivative(derivative(song.windowed_x))

    _threshold = 1 * (1 - _alpha) + (_thresholdCoeff *
                                     robustRMS(second_derivative, 2)) * _alpha

    indexes = []

    for i in range(len(second_derivative)):
        if (second_derivative[i] > _threshold):
            indexes.append(i)
    return indexes


def robustRMS(x, k):
    robustX = np.abs(x) ** 2
    median = np.median(robustX)
    robustX[robustX > median * k] = median * k
    return np.sqrt(np.mean(robustX**2))


def robustPower(x, k):
    robustX = np.abs(x) ** 2
    median = np.median(robustX)
    robustX[robustX > median * k] = median * k
    return np.sum(robustX) / len(robustX)


def robustStd(x, k):
    robustX = np.abs(x) ** 2
    median = np.median(robustX)
    robustX[robustX > median * k] = median * k
    return np.std(robustX)


def robustMedian(x, k):
    robustX = np.abs(x) ** 2
    median = np.median(robustX)
    robustX[robustX > median * k] = median * k
    return np.median(robustX)


def click_find(song, frame, silenceThreshold, powerEstimationThreshold, detectionThreshold, filter_signal, start_proc, end_proc, parallel=False):
    """
    Function that gets the click indexes
    """
    song.frame = frame
    song.window()
    if np.sqrt(np.mean(song.windowed_x ** 2)) < silenceThreshold:
        #idx_ += 1
        if filter_signal is True:
            return frame
        return

    try:
        lpc, _ = song.LPC(parallel)
    except Exception as e:
        #print('Error computing at', parallel, ':', e)
        return

    lpc /= np.max(lpc)
    e = deconvolve(song.frame, lpc)[0]

    e_mf = np.convolve(e[::-1], -lpc)[::-1]

    # Thresholding
    th_p = np.max([robustPower(e, powerEstimationThreshold) *
                   detectionThreshold, silenceThreshold])

    detections = [i + start_proc for i, v in
                  enumerate(e_mf[start_proc:end_proc]**2) if v >= th_p]
    if detections:
        starts = [detections[0]]
        ends = []
        end = detections[0]
        for idx, d in enumerate(detections[1:], 1):
            if d == detections[idx-1] + 1:
                end = d
            else:
                ends.append(end)
                starts.append(d)
                end = d
            ends.append(end)
        y_starts = []
        for start in starts:
            cutOff = (song.fs/2) * start / song.N
            frame = song.IIR(frame, cutOff, 'low')
            y_starts.append(start)
        #print('Signal has detections')
        # for end in ends:
        #     y.append(end + idx_)
        if filter_signal is True:
            return frame
        else:
            return y_starts
    else:
        #print('Signal does not have detections')
        if filter_signal is True:
            return frame
        else:
            return []

    #idx_ += 1

# low pass filtering


def find_clicks(song, filter_signal, parallel=True):
    idx_ = 0
    threshold = 10
    powerEstimationThreshold = 10
    silenceThreshold = 10 ** (-50 * 0.05)
    detectionThreshold = 10 ** (30 * 0.05)

    start_proc = int(song.M / 2 - song.H / 2)
    end_proc = int(song.M / 2 + song.H / 2)

    #y = []
    clickless = np.zeros(song.x.size)
    from pathos.pools import ProcessPool as Pool
    pthread = Pool(nodes=2)
    frames = (frame for frame in song.FrameGenerator())
    class_copies = (song for frame in song.FrameGenerator())
    silences = (silenceThreshold for frame in song.FrameGenerator())
    max_powers = (powerEstimationThreshold for frame in song.FrameGenerator())
    detections = (detectionThreshold for frame in song.FrameGenerator())
    isfiltering = (filter_signal for frame in song.FrameGenerator())
    scope_start = (start_proc for frame in song.FrameGenerator())
    scope_end = (end_proc for frame in song.FrameGenerator())
    parallels = (j for j in range(len(list(song.FrameGenerator()))))
    pt = pthread.amap(click_find, class_copies, frames, silences, max_powers,
                      detections, isfiltering, scope_start, scope_end, parallels)
    while not pt.ready():
        pass
    clicks = pt.get()
    #print('CLICKS AT FUN', clicks)
    return clicks


def greatestCommonDivisor(x, y, epsilon):
    if (x < y):
        return greatestCommonDivisor(y, x, epsilon)
    if (x == 0):
        return 0
    error = 2147483647
    ratio = 2147483647
    bpmDistance(x, y, error, ratio)
    if (abs(error) < epsilon):
        return y
    a = int(x+0.5)
    b = int(y+0.5)
    while (abs(error) > epsilon):
        bpmDistance(a, b, error, ratio)
        remainder = a % b
        a = b
        b = remainder
    return a


def bpmDistance(x, y, error, ratio):
    ratio = x/y
    error = -1
    if (ratio < 1):
        ratio = round(1./ratio)
        error = (x*ratio-y)/min(y, Real(x*ratio))*100
    else:
        ratio = round(ratio)
        error = (x-y*ratio)/min(x, Real(y*ratio))*100
    return error, ratio


def areEqual(a, b, tolerance):
    error = 0
    ratio = 0
    bpmDistance(a, b, error, ratio)
    return (abs(error) < tolerance) and (int(ratio) == 1)


def HarmonicBpm():
    harmonicBpms = np.zeros(bpms.size)
    harmonicRatios = np.zeros(bpms.size)
    for i in range(bpms.size):
        ratio = _bpm/bpms[i]
        if (ratio < 1):
            ratio = 1.0/ratio
        gcd = greatestCommonDivisor(_bpm, bpms[i], _tolerance)
        if (gcd > _threshold):
            harmonicBpms[i] = bpms[i]
            if (gcd < mingcd):
                mingcd = gcd

    harmonicBpms = np.sort(harmonicBpms)
    i = 0
    prevBestBpm = -1
    while i < harmonicBpms.size:
        prevBpm = harmonicBpms[i]
        while i < harmonicBpms.size:
            areEqual(prevBpm, harmonicBpms[i], _tolerance)
            error = 0
            r = 0
            bpmDistance(_bpm, harmonicBpms[i], error, r)
            error = abs(error)
            if (error < minError):
                bestBpm = harmonicBpms[i]
                minError = error
        i += 1
        if not areEqual(prevBestBpm, bestBpm, _tolerance):
            bestHarmonicBpms[bestBpm]
        else:
            e1 = 0,
            e2 = 0,
            r1 = 0,
            r2 = 0
            bpmDistance(
                _bpm, bestHarmonicBpms[bestHarmonicBpms.size-1], e1, r1)
            bpmDistance(_bpm, bestBpm, e2, r2)
            e1 = abs(e1)
            e2 = abs(e2)
            if (e1 > e2):
                bestHarmonicBpms[bestHarmonicBpms.size()-1] = bestBpm
        prevBestBpm = bestBpm
    return bestHarmonicBpms

def expand(song,threshold,dB,normalize=True):
    """
    This method filters a signal by expanding the
    robust peaks (peaks above a given threshold).
    If normalize is False, it will return the given
    signal with raised peaks
    """
    output = np.zeros(song.x.size)
    start = 0
    end = 2048
    for frame in song.FrameGenerator():
       #compute rms of a signal
       rms = np.sqrt(np.mean(frame**2))
       #raise peaks to the given intensity
       if rms > threshold:
           try:
               output[start:end] += 10**(dB*.05)
           except Exception as e:
               continue
       else:
           try:
               output[start:end] += frame
           except Exception as e:
               continue
       start += 1024
       end += 1024    
    #preserve peaks
    if normalize == True:
        output /= output.max() 
    return np.array(output)



def constantQ_transform(audio):
    pin = 0
    output = np.zeros(audio.size)
    pend = audio.size
    while pin < pend:
        selection = pin+2048
        song.frame = audio[pin:selection]
        constantQ = constantq.cqt(
            song.frame, sr=song.fs, hop_length=1024, n_bins=113)
        output[pin:selection] = constantq.icqt(
            constantQ, sr=song.fs, hop_length=1024)
        pin += 1024
    return output

def at_start(attacking_samples, decaying_samples,on_count, start_decay):
    total = attacking_samples + decaying_samples
    if on_count - start_decay < total:
        at_start = on_count - (attacking_samples*(on_count-start_decay)) / total
    else:
        at_start = on_count - attacking_samples
    return at_start

def noise_gate(audio,fs=44100,thresh_db=-32,reduction_db=-20,freq_cut=0,ca=.001,cdec=.01, hold_time=0):
    """
    RMS noise as least autocorrelated (stationary) samples (other peaks and heart beatings) 
    of its signal envelope (spsynth rather than anisotropic diffusion)
    enhanced phase values are probabilistic
    short-term phases (music: rms of tones, remanent noise: wider critical bandwidth) are discarded by the ear
    Additive noise is noise in environment. Humans can discriminate only part of it due to masking, leading to
    conditions:
        mask (Wiener gain) = psd/(psd+psd_noise) => harmonic distortion and higher critical noise
        if mask[i] === min(mask):
                alpha[i] (or beta[i]) = max(alpha)
        elif mask[i] === max(mask):
                alpha[i] (or beta[i]) = min(alpha)
        elif min(mask) > mask[i] > max(mask):
                alpha[i] (or beta) = max(alpha) *(( max(mask) - mask[i] )/( max(mask) - min(mask) ) + min(alpha) *( ( mask[i] - min(mas
k) )/( max(mask)- min(mask) ))
    """
    pend = len(audio)-(int(fs/10)+1102)
    song = sonify(audio, fs)
    pin = 0
    output = np.zeros(len(audio))
    amp = audio.max()   
    holda = ca
    half_fs = fs/2
    state = 'gate_off'
    attacking_samples = fs * ca + .5
    decaying_samples = fs * cdec + .5
    decay_start = decaying_samples
    lin_thresh = db2lin(thresh_db)
    lin_reduction = db2lin(reduction_db) 
    decay_end = 0   
    #to coefficient
    decay_factor = np.exp(reduction_db/decaying_samples)
    #to coefficient
    attack_factor = np.exp(-reduction_db/attacking_samples)
    gate_val = lin_thresh
    lookahead = ca
    delay_size = np.max([1,round(fs*lookahead)])
    stop_count = 0
    decay_end = 0
    current = 0
    ptr = 2048
    gate = []
    output = np.zeros(len(audio))
    attack_start = at_start(attacking_samples, decaying_samples, 0, decay_start)      
    if freq_cut == 0:
        freq_cut = 3
        ngate = 2048
    else:
        freq_cut = int(freq_cut*2/half_fs)
        ngate = freq_cut
    while pin < pend:
        selection = pin+2048
        song.frame = audio[selection:selection+2048]
        togo = audio.size - selection
        #print('Togo', togo, 'Selection', selection, 'Pin', pin)
        togo = min([togo,selection])
        song.window()
        song.Spectrum()
        now = selection
        dbs = 20*np.log10(song.magnitude_spectrum*song.H**2)
        #print('dbs',dbs)
        spectrum_db = db2lin(dbs)
        if np.any(spectrum_db >= lin_thresh):
            path_time = np.where(spectrum_db >= lin_thresh)[0][0]
            now += path_time
            print('Current state', state, 'and estimated future', selection+path_time)
        else:
            path_time = 0
            #print('Current now', now, 'and estimated future', selection+path_time)    
        #RMS in dB
        log_spectrum = np.sqrt(np.mean(song.frame**2))
        #print('Log spectrum', log_spectrum,'Threshold',lin_thresh)
        #dec_coef = to_coef(cdec, fs)
        #at_coef = to_coef(ca, fs) 
        #now = pin + selection + togo - togo 
        #now = selection
        #print('State',state,'Gate gain', gate_val,'Now',selection/44100, 'Stop count',stop_count/44100, 'Decay end', decay_end/44100, 'Decay start', decay_start/44100, 'Attack start', attack_start/44100)   
        if log_spectrum >= lin_thresh:
            print('Estimated now at',state,':',path_time)           
        if state == 'must_hold':
            if log_spectrum > lin_thresh:
                decay_end = now + delay_size 
            elif now >= stop_count: #gate is closed
                state = 'must_close'
                stop_count = now + decaying_samples      
                decay_start = now
            #break
        #falls looking for a next time    
        elif state == 'must_close':
            gate_val *= decay_factor
            if log_spectrum >= lin_thresh:
                decay_start = now + delay_size
                attack_start = now + delay_size
                attack_start = at_start(attacking_samples, decaying_samples, attack_start, decay_start)        
                state = 'must_keep_close'
                #print('Changing state to keep close')
            elif now == stop_count: #gate closes  
                state = 'gate_off'
                gate_val = lin_reduction
            #break
        #falls until new open    
        elif state == 'must_keep_close':
                gate_val *= decay_factor
                if log_spectrum >= lin_thresh:
                    decay_end = now + delay_size
                if now >= attack_start:
                    state = 'must_open'
                elif now >= stop_count:
                    state = 'must_off_until'
                    gate_val = lin_reduction
                #break    
        elif state == 'gate_off':
            if log_spectrum >= lin_thresh:
                decay_end = now + delay_size
                attack_start = now + delay_size
                attack_start = at_start(attacking_samples, decaying_samples, attack_start, decay_start)                        
            if now >= attack_start:
                state = 'must_open'
            else:
                state = 'must_off_until'
            #break

        elif state == 'must_off_until':
            if log_spectrum >= lin_thresh:
                decay_end = now + delay_size   
            if now >= attack_start:
                state = 'must_open'
            #break
        elif state == 'must_open':
            gate_val *= attack_factor
            if log_spectrum >= lin_thresh:
                decay_end = now + delay_size
                #print('Hold updating while open')
            if now >= attack_start:
                #print('Feasible', db2lin(spectrum_db))
                gate_val = 1
                state = 'must_hold'
            #break
        current += 2
        ptr += togo
        if log_spectrum >= lin_thresh:
            pin += path_time
        else:    
            pin = pin + song.H
        #gate acts in time domain of the signal
        #print('Gate value', gate_val, 'state', state)
        #print('Attack factor',attack_factor)
        #print('Decay factor',decay_factor)
        gate.append(gate_val)
        #print('Audio', audio[selection:selection+2048])
        np.add.at(output, range(selection,selection+2048), audio[selection:selection+2048]*gate_val)
        #print('Output',output)
    return np.float32(output)

def hiss_removal(audio):
    """
    RMS noise as least autocorrelated (stationary) samples (other peaks and heart beatings) 
    of its signal envelope (spsynth rather than anisotropic diffusion)
    enhanced phase values are probabilistic
    short-term phases (music: rms of tones, remanent noise: wider critical bandwidth) are discarded by the ear
    Additive noise is noise in environment. Humans can discriminate only part of it due to masking, leading to
    conditions:
        mask (Wiener gain) = psd/(psd+psd_noise) => harmonic distortion and higher critical noise
        if mask[i] === min(mask):
                alpha[i] (or beta[i]) = max(alpha)
        elif mask[i] === max(mask):
                alpha[i] (or beta[i]) = min(alpha)
        elif min(mask) > mask[i] > max(mask):
                alpha[i] (or beta) = max(alpha) *(( max(mask) - mask[i] )/( max(mask) - min(mask) ) + min(alpha) *( ( mask[i] - min(mask) )/( max(mask)- min(mask) ))
    """
    pend = len(audio)-(4410+1102)
    song = sonify(audio, 44100)
    song.FrameGenerator().__next__()
    song.window()
    song.Spectrum()
    noise_fft = song.fft(song.windowed_x)[:song.H+1]
    noise_power = np.log10(np.abs(noise_fft + 2 ** -16))
    noise_floor = np.exp(2.0 * noise_power.mean())
    mn = song.magnitude_spectrum
    e_n = energy(mn)
    pin = 0
    output = np.zeros(len(audio))
    hold_time = 0
    ca = 0
    cr = 0
    amp = audio.max()
    while pin < pend:
        selection = pin+2048
        song.frame = audio[pin:selection]
        song.window()
        song.M = 2048
        song.Spectrum()
        e_m = energy(song.magnitude_spectrum)
        SNR = 10 * np.log10(e_m / e_n)
        ft = song.fft(song.windowed_x)[:song.H+1]
        power_spectral_density = np.abs(ft) ** 2
        song.Envelope()
        song.AttackTime()
        rel_time = rel(song.attack_time,1)
        rel_coef = to_coef(rel_time, 44100)
        at_coef = to_coef(song.attack_time, 44100)
        ca = ca + song.attack_time
        cr = cr + rel_time
        if SNR > 0:
            np.add.at(output, range(pin, selection), audio[pin:selection])
        else:
            if np.any(power_spectral_density < noise_floor):
                gc = dyn_constraint_satis(
                    ft, [power_spectral_density, noise_floor], 0.12589254117941673)
                #gc = dyn_constraint_satis(
                #    ft, [power_spectral_density, noise_floor], 0.)
                if ca > hold_time:
                    gc = np.complex64([at_coef * gc[i - 1] + (1 - at_coef)
                                      * x if x > gc[i - 1] else x for i, x in enumerate(gc)])
                if ca <= hold_time:
                    gc = np.complex64([gc[i - 1] for i, x in enumerate(gc)])
                if cr > hold_time:
                    gc = np.complex64([rel_coef * gc[i - 1] + (1 - rel_coef)
                                      * x if x <= gc[i - 1] else x for i, x in enumerate(gc)])
                if cr <= hold_time:
                    gc = np.complex64([gc[i - 1] for i, x in enumerate(gc)])
                #print ("Reducing noise floor, this is taking some time")
                song.Phase(song.fft(song.windowed_x))
                song.phase = song.phase[:song.magnitude_spectrum.size]
                ft *= gc
                song.magnitude_spectrum = np.sqrt(
                    pow(ft.real, 2) + pow(ft.imag, 2))
                np.add.at(output, range(pin, selection),
                          song.ISTFT(song.magnitude_spectrum))
            else:
                np.add.at(output, range(pin, selection), audio[pin:selection])
        pin = pin + song.H
        hold_time += selection/44100
    hissless = amp * output / output.max()  # amplify to normal level
    return np.float32(hissless)

# optimizers and biquad_filter taken from Linear Audio Lib


def z_from_f(f, fs):
    out = []
    for x in f:
        if x == np.inf:
            out.append(-1.)
        else:
            out.append(((fs/np.pi)-x)/((fs/np.pi)+x))
    return out


def Fz_at_f(Poles, Zeros, f, fs, norm=0):
    omega = 2*np.pi*f/fs
    ans = 1.
    for z in Zeros:
        ans = ans*(np.exp(omega*1j)-z_from_f([z], fs))
    for p in Poles:
        ans = ans/(np.exp(omega*1j)-z_from_f([p], fs))
    if norm:
        ans = ans/max(abs(ans))
    return ans


def z_coeff(Poles, Zeros, fs, g, fg, fo='none'):
    if fg == np.inf:
        fg = fs/2
    if fo == 'none':
        beta = 1.0
    else:
        beta = f_warp(fo, fs)/fo
    a = np.poly(z_from_f(beta*np.array(Poles), fs))
    b = np.poly(z_from_f(beta*np.array(Zeros), fs))
    gain = np.array(
        10.**(g/20.)/abs(Fz_at_f(beta*np.array(Poles), beta*np.array(Zeros), fg, fs)))

    return (a, b*gain)


def biquad_filter(xin, z_coeff):
    a = z_coeff[0]
    b = z_coeff[1]
    xout = np.zeros(len(xin))
    xout[0] = b[0]*xin[0]
    xout[1] = b[0]*xin[1] + b[1]*xin[0] - a[1]*xout[0]

    for j in range(2, len(xin)):
        xout[j] = b[0]*xin[j]+b[1]*xin[j-1]+b[2] * \
            xin[j-2]-a[1]*xout[j-1]-a[2]*xout[j-2]

    return xout


Usage = "./quality.py [DATA_PATH] [HTRF_SIGNAL_PATH]"


def main():
    if len(sys.argv) < 3:
        print("\nBad amount of input arguments\n", Usage, "\n")
        sys.exit(1)

    try:
        DATA_PATH = sys.argv[1]
        RIAA = [[50.048724, 2122.0659], [500.48724, np.inf]]
        # Diffuse_Equalization = [[20,50.048724,100,f/Hz,2000,5000,10000,20000]] #gain = 18,20,18,17,12,8,5,7,-27
        abz = z_coeff(RIAA[0], RIAA[1], 44100, 0, 10000)

        if not os.path.exists(DATA_PATH):
            raise IOError("Must download sounds")

        for subdir, dirs, files in os.walk(DATA_PATH):
            for f in files:
                print(("Rewriting with LSTM Synthesis in %s" % f))
                # 1. Read the audio
                audio = read(DATA_PATH+'/'+f)[0]

                # 2. Convert to mono/stereo if needed. This is generally safe and often a prerequisite for further processing.
                audio = mono_stereo(audio)

                # 3. LSTM Synth Model Prediction (This is a creative/generative step)
                # This step is likely transforming the audio in a fundamental way, and it should happen before
                # any attempts to "clean" or "master" the audio for output, especially if Dolby is already present.
                # It's producing "neural" audio, which is your new source.
                neural = lstm_synth_predict(audio)

                # --- Point of Dolby Consideration ---
                # At this point, if your *original* 'audio' variable was the Dolby-encoded source,
                # then 'neural' is the *result* of your model, which might or might not retain
                # Dolby characteristics depending on what lstm_synth_predict does.
                # If 'neural' is now your primary audio signal, the following steps will apply to *it*.
                # If the goal is to process the *original* Dolby audio and then apply neural synthesis,
                # the order might need to be rethought entirely. Assuming 'neural' is your primary working audio:

                # 4. Anti-aliasing filtering (Essential and generally safe)
                # This removes frequencies above the sample rate, preventing digital artifacts.
                # This should be done early in the post-synthesis chain to ensure a clean signal.
                print(("Rewriting without aliasing in %s" % f))
                song = sonify(neural, 48000) # Assuming sonify prepares 'neural' for processing
                audio = song.IIR(song.x, 20000, 'lowpass') # Use 20kHz as a typical human hearing limit

                # 5. Remove DC (Direct Current) offset (Essential and generally safe)
                # DC offset can cause issues with playback devices and should be removed early.
                print(("Rewriting without DC in %s" % f))
                audio = song.IIR(audio, 40, 'highpass') # A low cut-off for DC removal

                # 6. Hum removal (Noise reduction, generally safe *if applied carefully*)
                # Removing specific hum frequencies (like 50/60 Hz) is a common and usually non-destructive process.
                print(("Rewriting with Hum removal applied in %s" % f))
                song.x = np.float32(audio) # Ensure song.x is updated with the current audio state
                without_hum = song.BandReject(np.float32(audio), 50, 16) # Apply hum removal
                del audio # Clean up for memory

                # 7. Subsonic rumble removal (Noise reduction, generally safe)
                # Removes very low frequencies that can be problematic for playback systems but are inaudible.
                print(("Rewriting with subsonic rumble removal applied in %s" % f))
                song.x = without_hum # Update song.x again
                without_rumble = song.IIR(song.x, 20, 'highpass') # Remove frequencies below 20Hz
                del without_hum # Clean up for memory

                # 8. Normalize (Crucial before any psychoacoustic processing or final output)
                # Normalization brings the audio to an optimal level without clipping.
                # It's important to do this *before* Equal Loudness or any RIAA if you were to reintroduce it.
                normalized_audio = normalize(without_rumble)
                del without_rumble

                # 9. Equal Loudness Contour (Psychoacoustic processing, *after* normalization)
                # This attempts to adjust the frequency response to match human hearing perception at different loudness levels.
                # This is a creative/enhancement step and should generally be applied late in the chain.
                print(("Rewriting with Equal Loudness contour in %s" % f))
                song.x = np.float32(normalized_audio) # Update song.x with normalized audio
                audio = song.EqualLoudness(normalized_audio)
                del normalized_audio

                # 10. Silence removal (Cleanup, generally safe)
                # This is a final cleanup step to remove completely silent portions.
                # It relies on the magnitude (loudness) of the signal, so it should be done after
                # all other processing that affects volume or frequency content.
                db_mag = 20 * np.log10(abs(audio))
                print(("Rewriting without silence in %s" % f))
                silence_threshold = -130 # complete silence
                loud_audio = np.delete(audio, np.where(db_mag < silence_threshold)) # remove it
                del audio

                # 11. Write the final file
                write_file(subdir+'/'+os.path.splitext(f)[0], 48000, loud_audio)

    except Exception as e:
        logger.exception(e)
        exit(1)


if __name__ == '__main__':
    asyncio.run(main())

