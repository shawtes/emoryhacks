#!/usr/bin/env python3
"""
Advanced Voice Biomarker Feature Extraction (2024 Research)
Extracts cutting-edge features and combines with existing features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import librosa.display
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def extract_sound_object_features(y, sr):
    """
    Sound Object-Based Voice Biomarkers (2024)
    Extract features from sustained vowel recordings
    """
    features = {}
    
    # Sound objects based on spectral characteristics
    try:
        # Spectral centroid variations
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_centroid_skew'] = np.mean((spectral_centroids - np.mean(spectral_centroids))**3) / (np.std(spectral_centroids)**3)
        
        # Spectral bandwidth variations
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral rolloff variations
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate variations (voice quality)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_skew'] = np.mean((zcr - np.mean(zcr))**3) / (np.std(zcr)**3 + 1e-8)
        
    except Exception as e:
        print(f"Error in sound object features: {e}")
        # Fill with zeros if extraction fails
        for key in ['spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_skew',
                   'spectral_bandwidth_mean', 'spectral_bandwidth_std', 
                   'spectral_rolloff_mean', 'spectral_rolloff_std',
                   'zcr_mean', 'zcr_std', 'zcr_skew']:
            features[key] = 0.0
    
    return features

def extract_voice_quality_features(y, sr):
    """
    Voice Quality Features (HNR, Jitter, Shimmer approximations)
    """
    features = {}
    
    try:
        # Harmonic-to-Noise Ratio approximation
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # Estimate harmonic content
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        # HNR approximation
        harmonic_energy = np.sum(harmonic**2)
        noise_energy = np.sum(percussive**2)
        features['hnr_approximation'] = 10 * np.log10((harmonic_energy + 1e-8) / (noise_energy + 1e-8))
        
        # Jitter approximation (pitch period variations)
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 1:
            pitch_periods = [1/p if p > 0 else 0 for p in pitch_values]
            pitch_periods = [p for p in pitch_periods if p > 0]
            if len(pitch_periods) > 1:
                features['jitter_approximation'] = np.std(pitch_periods) / np.mean(pitch_periods)
            else:
                features['jitter_approximation'] = 0.0
        else:
            features['jitter_approximation'] = 0.0
        
        # Shimmer approximation (amplitude variations)
        rms_energy = librosa.feature.rms(y=y)[0]
        if len(rms_energy) > 1:
            features['shimmer_approximation'] = np.std(rms_energy) / np.mean(rms_energy)
        else:
            features['shimmer_approximation'] = 0.0
            
        # Voice stability measures
        features['pitch_stability'] = 1.0 / (1.0 + features['jitter_approximation'])
        features['amplitude_stability'] = 1.0 / (1.0 + features['shimmer_approximation'])
        
    except Exception as e:
        print(f"Error in voice quality features: {e}")
        features.update({
            'hnr_approximation': 0.0,
            'jitter_approximation': 0.0,
            'shimmer_approximation': 0.0,
            'pitch_stability': 1.0,
            'amplitude_stability': 1.0
        })
    
    return features

def extract_prosodic_features(y, sr):
    """
    Prosodic Features (intonation, rhythm, pauses)
    """
    features = {}
    
    try:
        # Speech rate approximation (based on onset detection)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        duration = len(y) / sr
        features['speech_rate'] = len(onset_times) / duration if duration > 0 else 0.0
        
        # Pause detection (silence segments)
        rms = librosa.feature.rms(y=y)[0]
        silence_threshold = np.mean(rms) * 0.1  # Adaptive threshold
        silence_frames = rms < silence_threshold
        
        # Pause statistics
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                in_silence = True
                silence_start = i
            elif not is_silent and in_silence:
                in_silence = False
                silence_segments.append(i - silence_start)
        
        if silence_segments:
            features['pause_count'] = len(silence_segments)
            features['pause_duration_mean'] = np.mean(silence_segments) * (len(y) / sr) / len(silence_frames)
            features['pause_duration_std'] = np.std(silence_segments) * (len(y) / sr) / len(silence_frames)
        else:
            features['pause_count'] = 0
            features['pause_duration_mean'] = 0.0
            features['pause_duration_std'] = 0.0
        
        # Pitch contour analysis
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 1:
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            features['pitch_variability'] = np.std(pitch_values)
            features['pitch_mean'] = np.mean(pitch_values)
        else:
            features['pitch_range'] = 0.0
            features['pitch_variability'] = 0.0
            features['pitch_mean'] = 0.0
            
    except Exception as e:
        print(f"Error in prosodic features: {e}")
        features.update({
            'speech_rate': 0.0,
            'pause_count': 0,
            'pause_duration_mean': 0.0,
            'pause_duration_std': 0.0,
            'pitch_range': 0.0,
            'pitch_variability': 0.0,
            'pitch_mean': 0.0
        })
    
    return features

def extract_formant_features(y, sr):
    """
    Formant Features (F1-F3 approximations)
    """
    features = {}
    
    try:
        # Get MFCC features (related to formants)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Formant approximations based on spectral characteristics
        # F1 approximation (low frequency emphasis)
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Find spectral peaks that might correspond to formants
        spectral_peaks = []
        for frame in range(magnitude.shape[1]):
            spectrum = magnitude[:, frame]
            # Find local maxima
            peaks = []
            for i in range(1, len(spectrum)-1):
                if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                    peaks.append((freqs[i], spectrum[i]))
            
            # Sort by magnitude and take top 3
            peaks.sort(key=lambda x: x[1], reverse=True)
            spectral_peaks.append(peaks[:3])
        
        # Average formant estimates
        if spectral_peaks:
            f1_estimates = [peak[0][0] for peak in spectral_peaks if len(peak) > 0]
            f2_estimates = [peak[1][0] for peak in spectral_peaks if len(peak) > 1]
            f3_estimates = [peak[2][0] for peak in spectral_peaks if len(peak) > 2]
            
            features['f1_mean'] = np.mean(f1_estimates) if f1_estimates else 0.0
            features['f2_mean'] = np.mean(f2_estimates) if f2_estimates else 0.0
            features['f3_mean'] = np.mean(f3_estimates) if f3_estimates else 0.0
            
            features['f1_std'] = np.std(f1_estimates) if len(f1_estimates) > 1 else 0.0
            features['f2_std'] = np.std(f2_estimates) if len(f2_estimates) > 1 else 0.0
            features['f3_std'] = np.std(f3_estimates) if len(f3_estimates) > 1 else 0.0
        else:
            features.update({
                'f1_mean': 0.0, 'f2_mean': 0.0, 'f3_mean': 0.0,
                'f1_std': 0.0, 'f2_std': 0.0, 'f3_std': 0.0
            })
            
    except Exception as e:
        print(f"Error in formant features: {e}")
        features.update({
            'f1_mean': 0.0, 'f2_mean': 0.0, 'f3_mean': 0.0,
            'f1_std': 0.0, 'f2_std': 0.0, 'f3_std': 0.0
        })
    
    return features

def extract_advanced_spectral_features(y, sr):
    """
    Advanced Spectral Features (2024 Research)
    """
    features = {}
    
    try:
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        
        # Mel-frequency features (extended)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_db = librosa.power_to_db(mel_spectrogram)
        
        features['mel_energy_mean'] = np.mean(mel_db)
        features['mel_energy_std'] = np.std(mel_db)
        features['mel_energy_skew'] = np.mean((mel_db - np.mean(mel_db))**3) / (np.std(mel_db)**3 + 1e-8)
        features['mel_energy_kurtosis'] = np.mean((mel_db - np.mean(mel_db))**4) / (np.std(mel_db)**4 + 1e-8)
        
    except Exception as e:
        print(f"Error in advanced spectral features: {e}")
        # Fill with default values
        for i in range(7):  # Typical spectral contrast bands
            features[f'spectral_contrast_{i}_mean'] = 0.0
            features[f'spectral_contrast_{i}_std'] = 0.0
        features.update({
            'chroma_mean': 0.0, 'chroma_std': 0.0,
            'tonnetz_mean': 0.0, 'tonnetz_std': 0.0,
            'mel_energy_mean': 0.0, 'mel_energy_std': 0.0,
            'mel_energy_skew': 0.0, 'mel_energy_kurtosis': 0.0
        })
    
    return features

def extract_all_advanced_features(audio_file):
    """
    Extract all advanced voice biomarkers from an audio file
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Extract different feature sets
        sound_object_feats = extract_sound_object_features(y, sr)
        voice_quality_feats = extract_voice_quality_features(y, sr)
        prosodic_feats = extract_prosodic_features(y, sr)
        formant_feats = extract_formant_features(y, sr)
        spectral_feats = extract_advanced_spectral_features(y, sr)
        
        # Combine all features
        all_features = {}
        all_features.update(sound_object_feats)
        all_features.update(voice_quality_feats)
        all_features.update(prosodic_feats)
        all_features.update(formant_feats)
        all_features.update(spectral_feats)
        
        # Ensure all values are finite
        for key, value in all_features.items():
            if not np.isfinite(value):
                all_features[key] = 0.0
        
        return all_features
        
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return {}

def combine_with_existing_features(existing_features_file, advanced_features_file, output_file):
    """
    Combine existing features with new advanced features
    """
    print("🔗 COMBINING EXISTING + ADVANCED FEATURES")
    
    # Load existing features
    print(f"📊 Loading existing features from: {existing_features_file}")
    existing_df = pd.read_csv(existing_features_file)
    print(f"   Existing features: {len([col for col in existing_df.columns if col not in ['subject_id', 'label', 'filepath']])} features")
    print(f"   Samples: {len(existing_df)}")
    
    # Load advanced features
    print(f"🔬 Loading advanced features from: {advanced_features_file}")
    advanced_df = pd.read_csv(advanced_features_file)
    print(f"   Advanced features: {len([col for col in advanced_df.columns if col not in ['subject_id', 'label', 'filepath']])} features")
    print(f"   Samples: {len(advanced_df)}")
    
    # Merge on subject_id
    print("🔄 Merging feature sets...")
    combined_df = pd.merge(existing_df, advanced_df, on=['subject_id', 'label'], how='inner', suffixes=('_existing', '_advanced'))
    
    # Handle filepath columns
    if 'filepath_existing' in combined_df.columns:
        combined_df['filepath'] = combined_df['filepath_existing']
        combined_df = combined_df.drop(['filepath_existing', 'filepath_advanced'], axis=1)
    
    # Count features
    metadata_cols = ['subject_id', 'label', 'filepath']
    feature_cols = [col for col in combined_df.columns if col not in metadata_cols]
    
    print(f"✅ Combined dataset:")
    print(f"   Total features: {len(feature_cols)} features")
    print(f"   Total samples: {len(combined_df)}")
    print(f"   Class distribution: {combined_df['label'].value_counts().to_dict()}")
    
    # Save combined features
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print(f"💾 Combined features saved to: {output_file}")
    
    return combined_df

def main():
    """
    Main function to create combined feature dataset
    """
    print("🚀 ADVANCED VOICE BIOMARKER EXTRACTION (2024 RESEARCH)")
    print("Combining existing features with cutting-edge voice biomarkers")
    
    # File paths
    existing_features_file = Path("data/processed/features_clean.csv")
    advanced_features_file = Path("data/processed/advanced_features_only.csv")
    combined_features_file = Path("data/processed/combined_features.csv")
    
    # Check if existing features exist
    if not existing_features_file.exists():
        print(f"❌ Existing features file not found: {existing_features_file}")
        return
    
    # Extract advanced features if not already done
    if not advanced_features_file.exists():
        print("🔬 Extracting advanced features from audio data...")
        
        # Load existing features to get file list
        existing_df = pd.read_csv(existing_features_file)
        
        advanced_data = []
        
        for idx, row in existing_df.iterrows():
            subject_id = row['subject_id']
            label = row['label']
            
            print(f"Processing {idx+1}/{len(existing_df)}: {subject_id}")
            
            # Try to find corresponding audio file
            audio_dirs = [
                Path("data/raw/Audio_data/Dementia") / f"{subject_id}.mp3",
                Path("data/raw/Audio_data/Dementia") / f"{subject_id}.wav",
                Path("data/raw/Audio_data/Control") / f"{subject_id}.mp3",
                Path("data/raw/Audio_data/Control") / f"{subject_id}.wav",
                Path("data/raw") / f"{subject_id}.mp3",
                Path("data/raw") / f"{subject_id}.wav"
            ]
            
            audio_file = None
            for potential_file in audio_dirs:
                if potential_file.exists():
                    audio_file = potential_file
                    break
            
            if audio_file:
                # Extract advanced features
                features = extract_all_advanced_features(audio_file)
                features['subject_id'] = subject_id
                features['label'] = label
                features['filepath'] = str(audio_file)
                advanced_data.append(features)
            else:
                print(f"   ⚠️  Audio file not found for {subject_id}")
                # Create dummy entry with zeros
                dummy_features = {key: 0.0 for key in [
                    'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_skew',
                    'hnr_approximation', 'jitter_approximation', 'shimmer_approximation',
                    'speech_rate', 'pause_count', 'pitch_range', 'f1_mean', 'f2_mean'
                ]}
                dummy_features.update({
                    'subject_id': subject_id,
                    'label': label,
                    'filepath': 'unknown'
                })
                advanced_data.append(dummy_features)
        
        # Save advanced features
        if advanced_data:
            advanced_df = pd.DataFrame(advanced_data)
            advanced_features_file.parent.mkdir(parents=True, exist_ok=True)
            advanced_df.to_csv(advanced_features_file, index=False)
            print(f"💾 Advanced features saved to: {advanced_features_file}")
        else:
            print("❌ No advanced features extracted!")
            return
    
    # Combine features
    combined_df = combine_with_existing_features(
        existing_features_file, 
        advanced_features_file, 
        combined_features_file
    )
    
    print("✅ FEATURE COMBINATION COMPLETE!")
    print(f"🎯 Ready for enhanced Gradient Boosting training with {len([col for col in combined_df.columns if col not in ['subject_id', 'label', 'filepath']])} total features")

if __name__ == "__main__":
    main()
