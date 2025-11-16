#!/usr/bin/env python3
"""
Advanced audio feature extraction based on 2024 research for dementia detection.
Implements Sound Object-Based Voice Biomarkers and multi-level feature engineering.
"""

import numpy as np
import librosa
import librosa.display
from scipy import signal
from scipy.stats import entropy, kurtosis, skew
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import parselmouth, fall back to basic analysis if not available
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    print("Warning: parselmouth not available. Using basic acoustic analysis.")
    PARSELMOUTH_AVAILABLE = False


class AdvancedVoiceBiomarkers:
    """
    Advanced voice biomarker extraction based on latest 2024 research.
    Implements sound object-based features and multi-level analysis.
    """
    
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        
    def extract_sound_objects(self, audio_file):
        """
        Extract sound objects from sustained vowel recordings (/a/).
        Based on "Sound Object–Based Voice Biomarkers (2024)" research.
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=self.sr)
            
            # Segment into sound objects using energy-based detection
            sound_objects = self._segment_sound_objects(y, sr)
            
            # Extract features from each sound object
            object_features = []
            for obj in sound_objects:
                if len(obj) > 0.1 * sr:  # Min 100ms duration
                    obj_feat = self._extract_object_features(obj, sr)
                    object_features.append(obj_feat)
            
            if not object_features:
                return np.zeros(50)  # Return zeros if no valid objects
                
            # Aggregate object features
            aggregated = self._aggregate_object_features(object_features)
            return aggregated
            
        except Exception as e:
            print(f"Error extracting sound objects from {audio_file}: {e}")
            return np.zeros(50)
    
    def _segment_sound_objects(self, y, sr):
        """Segment audio into sound objects using energy and spectral changes."""
        # Voice activity detection
        frame_length = 2048
        hop_length = 512
        
        # Short-time energy
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Spectral centroid for detecting spectral changes
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Combine energy and spectral features for segmentation
        combined_signal = energy * (spectral_centroid / np.max(spectral_centroid))
        
        # Find peaks (sound object boundaries)
        peaks, _ = signal.find_peaks(combined_signal, distance=20, prominence=0.1)
        
        # Convert frame indices to sample indices
        peak_samples = librosa.frames_to_samples(peaks, hop_length=hop_length)
        
        # Extract segments between peaks
        segments = []
        for i in range(len(peak_samples) - 1):
            start = peak_samples[i]
            end = peak_samples[i + 1]
            segments.append(y[start:end])
        
        return segments
    
    def _extract_object_features(self, audio_obj, sr):
        """Extract features from individual sound objects."""
        features = []
        
        # Spectral features
        mfccs = librosa.feature.mfcc(y=audio_obj, sr=sr, n_mfcc=8)
        features.extend([np.mean(mfccs), np.std(mfccs), np.var(mfccs)])
        
        # Formant-like features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_obj, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_obj, sr=sr))
        features.extend([spectral_centroid, spectral_bandwidth])
        
        # Energy and duration
        energy = np.mean(librosa.feature.rms(y=audio_obj))
        duration = len(audio_obj) / sr
        features.extend([energy, duration])
        
        # Zero crossing rate (articulation measure)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_obj))
        features.append(zcr)
        
        return np.array(features)
    
    def _aggregate_object_features(self, object_features):
        """Aggregate features across all sound objects."""
        if not object_features:
            return np.zeros(50)
            
        stacked = np.vstack(object_features)
        
        # Statistical aggregation
        aggregated = []
        aggregated.extend(np.mean(stacked, axis=0))  # Mean across objects
        aggregated.extend(np.std(stacked, axis=0))   # Std across objects
        aggregated.extend(np.max(stacked, axis=0))   # Max across objects
        aggregated.extend(np.min(stacked, axis=0))   # Min across objects
        
        # Additional statistics
        aggregated.append(len(object_features))      # Number of sound objects
        aggregated.append(np.mean(np.diff(stacked, axis=0)))  # Transition dynamics
        
        return np.array(aggregated)
    
    def extract_acoustic_prosodic_features(self, audio_file):
        """
        Extract acoustic-prosodic features based on 2024 research.
        Focus on pitch variability, articulation rate, and pause duration.
        """
        try:
            if PARSELMOUTH_AVAILABLE:
                return self._extract_with_parselmouth(audio_file)
            else:
                return self._extract_with_librosa(audio_file)
        except Exception as e:
            print(f"Error extracting acoustic features from {audio_file}: {e}")
            return np.zeros(25)
    
    def _extract_with_librosa(self, audio_file):
        """Extract acoustic features using librosa (fallback method)."""
        y, sr = librosa.load(audio_file, sr=self.sr)
        
        features = {}
        
        # Basic pitch analysis using librosa
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, threshold=0.1)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            features['pitch_variability'] = np.std(pitch_values) / (np.mean(pitch_values) + 1e-8)
            features['pitch_entropy'] = entropy(np.histogram(pitch_values, bins=20)[0] + 1e-10)
        else:
            features.update({f'pitch_{k}': 0 for k in ['mean', 'std', 'range', 'variability', 'entropy']})
        
        # Spectral features as formant proxies
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Approximate formant features
        for i in range(1, 4):  # F1, F2, F3 approximations
            if i == 1:
                formant_approx = spectral_centroids * 0.5  # Lower frequencies
            elif i == 2:
                formant_approx = spectral_centroids  # Mid frequencies
            else:
                formant_approx = spectral_centroids * 1.5  # Higher frequencies
            
            if len(formant_approx) > 0:
                features[f'f{i}_mean'] = np.mean(formant_approx)
                features[f'f{i}_std'] = np.std(formant_approx)
                features[f'f{i}_variability'] = np.std(formant_approx) / (np.mean(formant_approx) + 1e-8)
            else:
                features.update({f'f{i}_{k}': 0 for k in ['mean', 'std', 'variability']})
        
        # Voice stability approximations
        # Use zero crossing rate and spectral features as proxies
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        features['jitter_local'] = np.std(zcr) / (np.mean(zcr) + 1e-8)
        features['jitter_rap'] = np.var(zcr)
        features['shimmer_local'] = np.std(rms) / (np.mean(rms) + 1e-8)
        features['shimmer_apq3'] = np.var(rms)
        
        # Harmonics-to-Noise Ratio approximation
        stft = librosa.stft(y)
        harmonic, percussive = librosa.decompose.hpss(stft)
        harmonic_power = np.mean(np.abs(harmonic)**2)
        percussive_power = np.mean(np.abs(percussive)**2)
        hnr_approx = 10 * np.log10((harmonic_power + 1e-8) / (percussive_power + 1e-8))
        
        features['hnr_mean'] = hnr_approx
        features['hnr_std'] = 0  # Can't estimate std with single value
        
        # Speech rate estimation
        features['speech_rate'] = self._estimate_speech_rate_librosa(y, sr)
        features['articulation_rate'] = self._estimate_articulation_rate_librosa(y, sr)
        
        # Pause analysis
        pause_features = self._analyze_pauses_librosa(y, sr)
        features.update(pause_features)
        
        return np.array(list(features.values()))
    
    def _extract_with_parselmouth(self, audio_file):
        """Extract acoustic features using Parselmouth (full method)."""
        sound = parselmouth.Sound(str(audio_file))
        
        features = {}
        
        # Pitch analysis
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            features['pitch_variability'] = np.std(pitch_values) / np.mean(pitch_values)
            features['pitch_entropy'] = entropy(np.histogram(pitch_values, bins=20)[0] + 1e-10)
        else:
            features.update({f'pitch_{k}': 0 for k in ['mean', 'std', 'range', 'variability', 'entropy']})
        
        # Formant analysis (F1, F2, F3)
        formants = sound.to_formant_burg()
        for i in range(1, 4):  # F1, F2, F3
            formant_values = []
            for t in np.arange(0, sound.duration, 0.01):  # Every 10ms
                f_val = call(formants, "Get value at time", i, t, "Hertz", "Linear")
                if not np.isnan(f_val) and f_val > 0:
                    formant_values.append(f_val)
            
            if formant_values:
                features[f'f{i}_mean'] = np.mean(formant_values)
                features[f'f{i}_std'] = np.std(formant_values)
                features[f'f{i}_variability'] = np.std(formant_values) / np.mean(formant_values)
            else:
                features.update({f'f{i}_{k}': 0 for k in ['mean', 'std', 'variability']})
        
        # Jitter and Shimmer (voice stability)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        if call(point_process, "Get number of points") > 2:
            features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_rap'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            features['shimmer_local'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_apq3'] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        else:
            features.update({f'{k}': 0 for k in ['jitter_local', 'jitter_rap', 'shimmer_local', 'shimmer_apq3']})
        
        # Harmonics-to-Noise Ratio (HNR)
        harmonicity = sound.to_harmonicity()
        hnr_values = harmonicity.values
        hnr_values = hnr_values[~np.isnan(hnr_values)]
        if len(hnr_values) > 0:
            features['hnr_mean'] = np.mean(hnr_values)
            features['hnr_std'] = np.std(hnr_values)
        else:
            features['hnr_mean'] = features['hnr_std'] = 0
        
        # Speech rate and articulation features
        features['speech_rate'] = self._estimate_speech_rate(sound)
        features['articulation_rate'] = self._estimate_articulation_rate(sound)
        
        # Pause analysis
        pause_features = self._analyze_pauses(sound)
        features.update(pause_features)
        
        return np.array(list(features.values()))
    
    def _estimate_speech_rate_librosa(self, y, sr):
        """Estimate speech rate using librosa."""
        try:
            # Use onset detection as proxy for syllables
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            duration = len(y) / sr
            speech_rate = len(onset_frames) / duration
            return speech_rate
        except:
            return 0
    
    def _estimate_articulation_rate_librosa(self, y, sr):
        """Estimate articulation rate using librosa."""
        try:
            # Voice activity detection using energy
            rms = librosa.feature.rms(y=y)[0]
            threshold = np.percentile(rms, 30)
            voiced_frames = rms > threshold
            
            voiced_duration = np.sum(voiced_frames) * (len(y) / sr / len(rms))
            
            if voiced_duration > 0:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
                articulation_rate = len(onset_frames) / voiced_duration
            else:
                articulation_rate = 0
                
            return articulation_rate
        except:
            return 0
    
    def _analyze_pauses_librosa(self, y, sr):
        """Analyze pause patterns using librosa."""
        try:
            # Use RMS energy for pause detection
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            
            # Detect pauses (low energy regions)
            threshold = np.percentile(rms, 20)
            pause_frames = rms < threshold
            
            # Find continuous pause segments
            pause_segments = []
            in_pause = False
            pause_start = 0
            
            frame_duration = 512 / sr  # Duration of each RMS frame
            
            for i, is_pause in enumerate(pause_frames):
                if is_pause and not in_pause:
                    pause_start = i
                    in_pause = True
                elif not is_pause and in_pause:
                    pause_duration = (i - pause_start) * frame_duration
                    if pause_duration > 0.1:  # Min 100ms pause
                        pause_segments.append(pause_duration)
                    in_pause = False
            
            duration = len(y) / sr
            
            if pause_segments:
                pause_features = {
                    'pause_count': len(pause_segments),
                    'pause_duration_mean': np.mean(pause_segments),
                    'pause_duration_std': np.std(pause_segments),
                    'pause_duration_total': np.sum(pause_segments),
                    'pause_ratio': np.sum(pause_segments) / duration
                }
            else:
                pause_features = {k: 0 for k in ['pause_count', 'pause_duration_mean', 
                                                'pause_duration_std', 'pause_duration_total', 'pause_ratio']}
            
            return pause_features
        except:
            return {k: 0 for k in ['pause_count', 'pause_duration_mean', 
                                   'pause_duration_std', 'pause_duration_total', 'pause_ratio']}
    
    def _estimate_speech_rate(self, sound):
        """Estimate speech rate (syllables per second)."""
        try:
            # Simple syllable estimation based on intensity peaks
            intensity = sound.to_intensity()
            intensity_values = intensity.values.flatten()
            
            # Find peaks in intensity (proxy for syllables)
            peaks, _ = signal.find_peaks(intensity_values, distance=10, prominence=0.1)
            
            speech_rate = len(peaks) / sound.duration
            return speech_rate
        except:
            return 0
    
    def _estimate_articulation_rate(self, sound):
        """Estimate articulation rate excluding pauses."""
        try:
            # Voice activity detection
            intensity = sound.to_intensity()
            intensity_values = intensity.values.flatten()
            
            # Threshold for voiced segments
            threshold = np.percentile(intensity_values, 30)
            voiced_frames = intensity_values > threshold
            
            # Calculate articulation rate
            voiced_duration = np.sum(voiced_frames) * (sound.duration / len(intensity_values))
            
            if voiced_duration > 0:
                peaks, _ = signal.find_peaks(intensity_values[voiced_frames], distance=10)
                articulation_rate = len(peaks) / voiced_duration
            else:
                articulation_rate = 0
                
            return articulation_rate
        except:
            return 0
    
    def _analyze_pauses(self, sound):
        """Analyze pause patterns in speech."""
        try:
            intensity = sound.to_intensity()
            intensity_values = intensity.values.flatten()
            
            # Detect pauses (low intensity regions)
            threshold = np.percentile(intensity_values, 20)
            pause_frames = intensity_values < threshold
            
            # Find continuous pause segments
            pause_segments = []
            in_pause = False
            pause_start = 0
            
            for i, is_pause in enumerate(pause_frames):
                if is_pause and not in_pause:
                    pause_start = i
                    in_pause = True
                elif not is_pause and in_pause:
                    pause_duration = (i - pause_start) * (sound.duration / len(intensity_values))
                    if pause_duration > 0.1:  # Min 100ms pause
                        pause_segments.append(pause_duration)
                    in_pause = False
            
            if pause_segments:
                pause_features = {
                    'pause_count': len(pause_segments),
                    'pause_duration_mean': np.mean(pause_segments),
                    'pause_duration_std': np.std(pause_segments),
                    'pause_duration_total': np.sum(pause_segments),
                    'pause_ratio': np.sum(pause_segments) / sound.duration
                }
            else:
                pause_features = {k: 0 for k in ['pause_count', 'pause_duration_mean', 
                                                'pause_duration_std', 'pause_duration_total', 'pause_ratio']}
            
            return pause_features
        except:
            return {k: 0 for k in ['pause_count', 'pause_duration_mean', 
                                   'pause_duration_std', 'pause_duration_total', 'pause_ratio']}
    
    def extract_spectral_features(self, audio_file):
        """Extract enhanced spectral features."""
        try:
            y, sr = librosa.load(audio_file, sr=self.sr)
            
            features = []
            
            # Enhanced MFCCs with derivatives
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Statistical aggregation of MFCCs
            for coeff_set in [mfccs, mfcc_delta, mfcc_delta2]:
                features.extend([np.mean(coeff_set), np.std(coeff_set), 
                               np.var(coeff_set), skew(coeff_set.flatten()),
                               kurtosis(coeff_set.flatten())])
            
            # Spectral features
            spectral_features = [
                librosa.feature.spectral_centroid(y=y, sr=sr),
                librosa.feature.spectral_bandwidth(y=y, sr=sr),
                librosa.feature.spectral_contrast(y=y, sr=sr),
                librosa.feature.spectral_flatness(y=y),
                librosa.feature.spectral_rolloff(y=y, sr=sr)
            ]
            
            for feat in spectral_features:
                features.extend([np.mean(feat), np.std(feat)])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([np.mean(chroma), np.std(chroma), np.var(chroma)])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting spectral features from {audio_file}: {e}")
            return np.zeros(28)
    
    def extract_comprehensive_features(self, audio_file):
        """Extract all advanced features for a single audio file."""
        try:
            # Sound object-based features (50 features)
            sound_obj_features = self.extract_sound_objects(audio_file)
            
            # Acoustic-prosodic features (25 features)  
            acoustic_features = self.extract_acoustic_prosodic_features(audio_file)
            
            # Enhanced spectral features (28 features)
            spectral_features = self.extract_spectral_features(audio_file)
            
            # Combine all features
            all_features = np.concatenate([
                sound_obj_features,
                acoustic_features, 
                spectral_features
            ])
            
            return all_features
            
        except Exception as e:
            print(f"Error extracting comprehensive features from {audio_file}: {e}")
            return np.zeros(103)  # 50 + 25 + 28


def extract_advanced_features_batch(audio_dir, output_csv):
    """Extract advanced features for all audio files in directory."""
    print("=== EXTRACTING ADVANCED VOICE BIOMARKERS ===")
    
    extractor = AdvancedVoiceBiomarkers()
    audio_dir = Path(audio_dir)
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(list(audio_dir.rglob(ext)))
    
    print(f"Found {len(audio_files)} audio files")
    
    results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"Processing {i}/{len(audio_files)}: {audio_file.name}")
        
        # Extract features
        features = extractor.extract_comprehensive_features(audio_file)
        
        # Determine label from path
        label = 1 if 'dementia' in str(audio_file).lower() else 0
        
        # Create feature dictionary
        feature_dict = {
            'filepath': str(audio_file),
            'subject_id': audio_file.stem,
            'label': label
        }
        
        # Add numbered features
        for j, feat_val in enumerate(features):
            feature_dict[f'advanced_feat_{j:03d}'] = feat_val
        
        results.append(feature_dict)
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Advanced features saved to: {output_csv}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {len([c for c in df.columns if c.startswith('advanced_feat_')])}")
    
    return df


if __name__ == "__main__":
    # Extract advanced features
    audio_dir = Path("data/processed")
    output_csv = "data/processed/advanced_features.csv"
    
    extract_advanced_features_batch(audio_dir, output_csv)
