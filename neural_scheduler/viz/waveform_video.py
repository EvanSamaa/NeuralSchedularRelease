"""Rolling dual-waveform video of a two-channel dialogue.

Each channel is drawn as a scrolling waveform that lights up while its
energy is above a threshold; the two inputs become the left and right audio
channels of the video. Needs ``opencv-python`` and ``librosa``; muxing audio
needs ``ffmpeg`` on ``PATH`` (``moviepy`` is used as a fallback).
"""

import os
import subprocess
from pathlib import Path

import cv2
import librosa
import numpy as np
import soundfile as sf


def create_dual_waveform_video(audio1_path, audio2_path, output_name="output.mp4", 
                                frame_rate=30, vad_threshold=0.02, window_size=1.5):
    """
    Generate a video with dual rolling waveforms that light up during speech.
    
    Args:
        audio1_path: Path to first audio file (top waveform)
        audio2_path: Path to second audio file (bottom waveform)
        output_name: Name of output video file
        frame_rate: Video frame rate (default: 30)
        vad_threshold: Energy threshold for speech detection (default: 0.02)
        window_size: Window size in seconds for rolling display (default: 1.5)
    """
    # Load audio files
    y1, sr1 = librosa.load(audio1_path, sr=None, mono=True)
    y2, sr2 = librosa.load(audio2_path, sr=None, mono=True)
    
    # Resample if needed to match sample rates
    if sr1 != sr2:
        sr = max(sr1, sr2)
        if sr1 != sr:
            y1 = librosa.resample(y1, orig_sr=sr1, target_sr=sr)
        if sr2 != sr:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr)
    else:
        sr = sr1
    
    # Pad shorter audio to match lengths
    max_len = max(len(y1), len(y2))
    y1 = np.pad(y1, (0, max_len - len(y1)))
    y2 = np.pad(y2, (0, max_len - len(y2)))
    
    # Combine stereo audio for output
    stereo_audio = np.stack([y1, y2], axis=1)
    
    # Video parameters
    width, height = 1280, 720
    duration = len(y1) / sr
    total_frames = int(duration * frame_rate)
    
    # Output path in same folder as input
    output_path = Path(audio1_path).parent / output_name
    
    # Temporary files
    temp_video = str(output_path.with_name(output_path.stem + '_temp_video.mp4'))
    temp_audio = str(output_path.with_name(output_path.stem + '_temp_audio.wav'))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(temp_video, fourcc, frame_rate, (width, height))
    
    # Calculate samples per frame and window
    samples_per_frame = int(sr / frame_rate)
    window_samples = int(window_size * sr)
    
    # Energy-based VAD with shorter hop for smoother detection
    hop_length = 256
    energy1 = librosa.feature.rms(y=y1, hop_length=hop_length)[0]
    energy2 = librosa.feature.rms(y=y2, hop_length=hop_length)[0]
    
    # Normalize energies
    energy1 = energy1 / (np.max(energy1) + 1e-8)
    energy2 = energy2 / (np.max(energy2) + 1e-8)
    
    # Compute envelope for better visualization (not for normalization)
    envelope1 = librosa.feature.rms(y=y1, hop_length=hop_length, frame_length=2048)[0]
    envelope2 = librosa.feature.rms(y=y2, hop_length=hop_length, frame_length=2048)[0]
    
    print(f"Generating video: {total_frames} frames...")
    print(f"Duration: {duration:.2f}s, Window: {window_size}s")
    
    for frame_idx in range(total_frames):
        # Create black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Current sample position
        current_sample = frame_idx * samples_per_frame
        current_time = current_sample / sr
        
        # Get window of samples centered on current position
        start_idx = max(0, current_sample - window_samples // 2)
        end_idx = min(len(y1), current_sample + window_samples // 2)
        
        # Extract windows
        window1 = y1[start_idx:end_idx]
        window2 = y2[start_idx:end_idx]
        
        # Determine speech activity at current time
        energy_idx = int(current_sample / hop_length)
        energy_idx = min(energy_idx, len(energy1) - 1)
        
        is_speech1 = energy1[energy_idx] > vad_threshold
        is_speech2 = energy2[energy_idx] > vad_threshold
        
        # Get current envelope values for amplitude scaling
        env1 = envelope1[energy_idx]
        env2 = envelope2[energy_idx]
        
        # Colors: green for speech, blue for silence
        color1 = (0, 255, 0) if is_speech1 else (255, 100, 0)
        color2 = (0, 255, 0) if is_speech2 else (255, 100, 0)
        
        # Background color: darker when silent
        bg_alpha1 = 0.3 if is_speech1 else 0.1
        bg_alpha2 = 0.3 if is_speech2 else 0.1
        
        # Draw background panels
        top_panel = int(height * 0.05)
        mid_panel = int(height * 0.5)
        panel_height = int(height * 0.4)
        
        # Top panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, top_panel), (width, top_panel + panel_height), 
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, bg_alpha1, frame, 1 - bg_alpha1, 0, frame)
        
        # Bottom panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, mid_panel), (width, mid_panel + panel_height), 
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, bg_alpha2, frame, 1 - bg_alpha2, 0, frame)
        
        # Draw waveforms with envelope-based scaling
        draw_waveform(frame, window1, y_center=top_panel + panel_height // 2, 
                     color=color1, width=width, height=panel_height,
                     envelope_scale=env1, is_speech=is_speech1)
        draw_waveform(frame, window2, y_center=mid_panel + panel_height // 2, 
                     color=color2, width=width, height=panel_height,
                     envelope_scale=env2, is_speech=is_speech2)
        
        # Add time indicator line in center
        center_x = width // 2
        cv2.line(frame, (center_x, top_panel), (center_x, top_panel + panel_height), 
                (255, 255, 255), 2)
        cv2.line(frame, (center_x, mid_panel), (center_x, mid_panel + panel_height), 
                (255, 255, 255), 2)
        
        # Add labels with speech indicators
        status1 = "SPEAKING" if is_speech1 else "Silence"
        status2 = "SPEAKING" if is_speech2 else "Silence"
        
        label_color1 = (0, 255, 0) if is_speech1 else (150, 150, 150)
        label_color2 = (0, 255, 0) if is_speech2 else (150, 150, 150)
        
        cv2.putText(frame, f"Audio 1 - {status1}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color1, 2)
        cv2.putText(frame, f"Audio 2 - {status2}", (20, mid_panel - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color2, 2)
        
        # Add timestamp
        cv2.putText(frame, f"Time: {current_time:.2f}s / {duration:.2f}s", 
                   (width - 300, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        video.write(frame)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Progress: {frame_idx + 1}/{total_frames}")
    
    video.release()
    print("Video frames written.")
    
    # Save audio file
    print("Saving audio...")
    sf.write(temp_audio, stereo_audio, sr)
    print(f"Audio saved to: {temp_audio}")
    
    # Check if ffmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              timeout=5)
        ffmpeg_available = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        ffmpeg_available = False
    
    if ffmpeg_available:
        # Combine video and audio using ffmpeg with highly compatible settings
        print("Combining video and audio with ffmpeg...")
        final_output = str(output_path)
        
        # Use H.264 baseline profile and AAC-LC for maximum compatibility
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video, 
            '-i', temp_audio,
            '-c:v', 'libx264',
            '-profile:v', 'baseline',  # Most compatible H.264 profile
            '-level', '3.0',
            '-pix_fmt', 'yuv420p',  # Ensure compatible pixel format
            '-c:a', 'aac',
            '-b:a', '192k',  # Higher audio bitrate for quality
            '-ar', '48000',  # Standard sample rate
            '-ac', '2',  # Stereo
            '-movflags', '+faststart',  # Enable streaming/preview
            '-shortest',
            final_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            print("Falling back to moviepy...")
            ffmpeg_available = False
        else:
            # Clean up temp files
            try:
                os.remove(temp_video)
                os.remove(temp_audio)
            except OSError:
                pass
            print(f"Video saved to: {final_output}")
            print("Note: Audio should work when downloaded. VS Code preview may not support audio playback.")
            return final_output
    
    if not ffmpeg_available:
        # Fallback: use moviepy
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            print("Using moviepy to combine video and audio...")
            
            video_clip = VideoFileClip(temp_video)
            audio_clip = AudioFileClip(temp_audio)
            
            final_clip = video_clip.set_audio(audio_clip)
            final_output = str(output_path)
            
            # Use compatible encoding settings
            final_clip.write_videofile(
                final_output, 
                codec='libx264',
                audio_codec='aac',
                audio_bitrate='192k',
                preset='medium',
                ffmpeg_params=['-movflags', '+faststart', '-pix_fmt', 'yuv420p']
            )
            
            video_clip.close()
            audio_clip.close()
            
            # Clean up temp files
            try:
                os.remove(temp_video)
                os.remove(temp_audio)
            except OSError:
                pass
            
            print(f"Video saved to: {final_output}")
            print("Note: Audio should work when downloaded. VS Code preview may not support audio playback.")
            return final_output
            
        except ImportError:
            print("ERROR: Neither ffmpeg nor moviepy available.")
            print(f"Video saved without audio to: {temp_video}")
            print(f"Audio saved separately to: {temp_audio}")
            print("Please combine them manually or install ffmpeg/moviepy")
            return temp_video


def draw_waveform(frame, waveform, y_center, color, width, height, envelope_scale, is_speech):
    """Draw a waveform on the frame with envelope-based scaling"""
    if len(waveform) == 0:
        return
    
    # Downsample waveform to match display width
    num_points = width
    step = max(1, len(waveform) // num_points)
    
    # Use RMS in chunks for smoother, more visible waveform
    waveform_display = []
    for i in range(0, len(waveform), step):
        chunk = waveform[i:i+step]
        if len(chunk) > 0:
            # Use RMS instead of raw samples for better visibility
            rms_val = np.sqrt(np.mean(chunk**2))
            waveform_display.append(rms_val)
    
    waveform_display = np.array(waveform_display[:num_points])
    
    # Envelope scaling with a floor so quiet passages stay visible
    scale_factor = max(0.3, envelope_scale * 2.0)
    
    # Apply scaling
    waveform_display = waveform_display / (scale_factor + 1e-8)
    waveform_display = np.clip(waveform_display, -1, 1)
    
    # Draw waveform with thicker lines
    line_thickness = 2 if is_speech else 1
    
    for i in range(len(waveform_display) - 1):
        y1 = int(y_center - waveform_display[i] * height * 0.4)
        y2 = int(y_center - waveform_display[i + 1] * height * 0.4)
        
        # Clamp to panel bounds
        y1 = max(y_center - height // 2, min(y_center + height // 2, y1))
        y2 = max(y_center - height // 2, min(y_center + height // 2, y2))
        
        cv2.line(frame, (i, y1), (i + 1, y2), color, line_thickness)
    
    # Draw center line
    cv2.line(frame, (0, y_center), (width, y_center), (80, 80, 80), 1)
