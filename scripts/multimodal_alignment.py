import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
import librosa
import soundfile as sf
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel
import whisper
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalAlignment:
    """Multimodal alignment system for audio, image, and gesture data."""

    def __init__(self, base_dir: str, output_dir: str = "multimodal_aligned"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize models
        self.whisper_model = None
        self.image_model = None
        self.image_processor = None
        self.audio_model = None

        # Data storage
        self.aligned_data = []
        self.audio_features = []
        self.image_features = []
        self.gesture_features = []
        self.temporal_alignments = []

        # Configuration
        self.config = {
            'audio_sample_rate': 16000,
            'image_size': (224, 224),
            'feature_dim': 768,
            'temporal_window': 5.0,  # seconds
            'similarity_threshold': 0.7
        }

        # Initialize models
        self.initialize_models()

    def initialize_models(self):
        """Initialize all required models for multimodal processing."""
        logger.info("Initializing multimodal models...")

        try:
            # Initialize Whisper for speech recognition
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")

        try:
            # Initialize image feature extractor
            self.image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.image_model = AutoModel.from_pretrained("google/vit-base-patch16-224")
            logger.info("✅ Image model loaded")
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")

        try:
            # Initialize audio feature extractor (using same model for consistency)
            self.audio_model = AutoModel.from_pretrained("facebook/wav2vec2-base")
            logger.info("✅ Audio model loaded")
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")

    def extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract features from audio file."""
        logger.info(f"Extracting audio features from: {audio_path}")

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config['audio_sample_rate'])

            # Extract basic features
            features = {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'rms_energy': np.sqrt(np.mean(audio**2)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
                'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
                'mfcc': np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1).tolist()
            }

            # Speech recognition if Whisper is available
            if self.whisper_model:
                try:
                    result = self.whisper_model.transcribe(audio_path)
                    features['transcript'] = result['text']
                    features['segments'] = result['segments']
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")
                    features['transcript'] = ""
                    features['segments'] = []

            # Extract embeddings if audio model is available
            if self.audio_model:
                try:
                    # Convert audio to tensor format expected by wav2vec2
                    audio_tensor = torch.tensor(audio).unsqueeze(0)
                    with torch.no_grad():
                        outputs = self.audio_model(audio_tensor)
                        features['embedding'] = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
                except Exception as e:
                    logger.warning(f"Audio embedding extraction failed: {e}")

            return features

        except Exception as e:
            logger.error(f"Failed to extract audio features: {e}")
            return {}

    def extract_image_features(self, image_path: str) -> Dict[str, Any]:
        """Extract features from image file."""
        logger.info(f"Extracting image features from: {image_path}")

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.config['image_size'])

            # Extract basic features
            features = {
                'size': image.size,
                'mode': image.mode,
                'format': image.format
            }

            # Convert to numpy array for OpenCV features
            img_array = np.array(image)

            # Extract color features
            features['mean_color'] = np.mean(img_array, axis=(0, 1)).tolist()
            features['std_color'] = np.std(img_array, axis=(0, 1)).tolist()

            # Extract texture features using GLCM
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            features['brightness'] = np.mean(gray)
            features['contrast'] = np.std(gray)

            # Extract embeddings if image model is available
            if self.image_processor and self.image_model:
                try:
                    inputs = self.image_processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.image_model(**inputs)
                        features['embedding'] = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
                except Exception as e:
                    logger.warning(f"Image embedding extraction failed: {e}")

            # Extract face detection features
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                features['face_count'] = len(faces)
                features['face_locations'] = faces.tolist()
            except Exception as e:
                logger.warning(f"Face detection failed: {e}")
                features['face_count'] = 0
                features['face_locations'] = []

            return features

        except Exception as e:
            logger.error(f"Failed to extract image features: {e}")
            return {}

    def extract_gesture_features(self, gesture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from gesture data (simplified implementation)."""
        logger.info("Extracting gesture features")

        try:
            features = {
                'gesture_type': gesture_data.get('type', 'unknown'),
                'confidence': gesture_data.get('confidence', 0),
                'duration': gesture_data.get('duration', 0),
                'hand_positions': gesture_data.get('hand_positions', []),
                'body_pose': gesture_data.get('body_pose', {}),
                'facial_expression': gesture_data.get('facial_expression', {})
            }

            # Calculate gesture complexity
            if features['hand_positions']:
                positions = np.array(features['hand_positions'])
                features['movement_magnitude'] = np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                features['gesture_complexity'] = len(positions) / max(features['duration'], 1)
            else:
                features['movement_magnitude'] = 0
                features['gesture_complexity'] = 0

            return features

        except Exception as e:
            logger.error(f"Failed to extract gesture features: {e}")
            return {}

    def align_temporal_data(self, conversation_data: Dict[str, Any],
                          audio_data: List[Dict[str, Any]] = None,
                          image_data: List[Dict[str, Any]] = None,
                          gesture_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Align multimodal data temporally with conversation."""
        logger.info("Aligning multimodal data temporally")

        aligned_result = {
            'conversation_id': conversation_data.get('conversation_id', ''),
            'messages': [],
            'audio_segments': [],
            'image_segments': [],
            'gesture_segments': [],
            'temporal_mapping': []
        }

        messages = conversation_data.get('processed_messages', [])

        for msg_idx, message in enumerate(messages):
            msg_timestamp = message.get('timestamp', '')
            msg_start_time = self.parse_timestamp(msg_timestamp)

            if msg_start_time is None:
                continue

            # Find corresponding audio segments
            audio_segments = []
            if audio_data:
                for audio_item in audio_data:
                    audio_start = audio_item.get('start_time', 0)
                    audio_end = audio_item.get('end_time', 0)

                    if self.temporal_overlap(msg_start_time, msg_start_time + 30, audio_start, audio_end):
                        audio_segments.append({
                            'audio_id': audio_item.get('audio_id', ''),
                            'start_time': audio_start,
                            'end_time': audio_end,
                            'features': audio_item.get('features', {}),
                            'overlap_ratio': self.calculate_overlap_ratio(
                                msg_start_time, msg_start_time + 30, audio_start, audio_end
                            )
                        })

            # Find corresponding image segments
            image_segments = []
            if image_data:
                for image_item in image_data:
                    image_timestamp = image_item.get('timestamp', '')
                    image_time = self.parse_timestamp(image_timestamp)

                    if image_time and abs(image_time - msg_start_time) < self.config['temporal_window']:
                        image_segments.append({
                            'image_id': image_item.get('image_id', ''),
                            'timestamp': image_timestamp,
                            'features': image_item.get('features', {}),
                            'time_diff': abs(image_time - msg_start_time)
                        })

            # Find corresponding gesture segments
            gesture_segments = []
            if gesture_data:
                for gesture_item in gesture_data:
                    gesture_start = gesture_item.get('start_time', 0)
                    gesture_end = gesture_item.get('end_time', 0)

                    if self.temporal_overlap(msg_start_time, msg_start_time + 30, gesture_start, gesture_end):
                        gesture_segments.append({
                            'gesture_id': gesture_item.get('gesture_id', ''),
                            'start_time': gesture_start,
                            'end_time': gesture_end,
                            'features': gesture_item.get('features', {}),
                            'overlap_ratio': self.calculate_overlap_ratio(
                                msg_start_time, msg_start_time + 30, gesture_start, gesture_end
                            )
                        })

            # Create aligned message
            aligned_message = {
                'message_id': message.get('message_id', ''),
                'speaker': message.get('speaker', ''),
                'content': message.get('content', ''),
                'timestamp': msg_timestamp,
                'audio_segments': audio_segments,
                'image_segments': image_segments,
                'gesture_segments': gesture_segments,
                'multimodal_features': self.combine_multimodal_features(
                    message, audio_segments, image_segments, gesture_segments
                )
            }

            aligned_result['messages'].append(aligned_message)

            # Add to temporal mapping
            aligned_result['temporal_mapping'].append({
                'message_id': message.get('message_id', ''),
                'timestamp': msg_timestamp,
                'audio_count': len(audio_segments),
                'image_count': len(image_segments),
                'gesture_count': len(gesture_segments)
            })

        return aligned_result

    def parse_timestamp(self, timestamp: str) -> Optional[float]:
        """Parse timestamp string to seconds."""
        if not timestamp:
            return None

        try:
            # Try different timestamp formats
            if 'T' in timestamp:
                # ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.timestamp()
            elif ':' in timestamp:
                # Time format (HH:MM:SS)
                parts = timestamp.split(':')
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            else:
                # Unix timestamp
                return float(timestamp)
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp}: {e}")
            return None

    def temporal_overlap(self, start1: float, end1: float, start2: float, end2: float) -> bool:
        """Check if two time intervals overlap."""
        return max(start1, start2) < min(end1, end2)

    def calculate_overlap_ratio(self, start1: float, end1: float, start2: float, end2: float) -> float:
        """Calculate overlap ratio between two time intervals."""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start >= overlap_end:
            return 0.0

        overlap_duration = overlap_end - overlap_start
        interval1_duration = end1 - start1
        interval2_duration = end2 - start2

        return overlap_duration / min(interval1_duration, interval2_duration)

    def combine_multimodal_features(self, message: Dict[str, Any],
                                  audio_segments: List[Dict[str, Any]],
                                  image_segments: List[Dict[str, Any]],
                                  gesture_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine features from all modalities."""
        combined_features = {
            'text_features': {
                'complexity': message.get('complexity', {}).get('score', 0),
                'domains': message.get('domains', []),
                'intents': message.get('intents', []),
                'has_metaphor': message.get('has_metaphor', False)
            },
            'audio_features': {},
            'image_features': {},
            'gesture_features': {},
            'multimodal_scores': {}
        }

        # Aggregate audio features
        if audio_segments:
            audio_features = []
            for segment in audio_segments:
                features = segment.get('features', {})
                if features:
                    audio_features.append(features)

            if audio_features:
                # Calculate average features
                avg_audio = {}
                for key in audio_features[0].keys():
                    if key in ['mfcc', 'embedding']:
                        # Handle list features
                        all_values = [f[key] for f in audio_features if key in f]
                        if all_values:
                            avg_audio[key] = np.mean(all_values, axis=0).tolist()
                    else:
                        # Handle scalar features
                        all_values = [f[key] for f in audio_features if key in f]
                        if all_values:
                            avg_audio[key] = np.mean(all_values)

                combined_features['audio_features'] = avg_audio

        # Aggregate image features
        if image_segments:
            image_features = []
            for segment in image_segments:
                features = segment.get('features', {})
                if features:
                    image_features.append(features)

            if image_features:
                # Calculate average features
                avg_image = {}
                for key in image_features[0].keys():
                    if key in ['embedding', 'mean_color', 'std_color', 'mfcc']:
                        # Handle list features
                        all_values = [f[key] for f in image_features if key in f]
                        if all_values:
                            avg_image[key] = np.mean(all_values, axis=0).tolist()
                    else:
                        # Handle scalar features
                        all_values = [f[key] for f in image_features if key in f]
                        if all_values:
                            avg_image[key] = np.mean(all_values)

                combined_features['image_features'] = avg_image

        # Aggregate gesture features
        if gesture_segments:
            gesture_features = []
            for segment in gesture_segments:
                features = segment.get('features', {})
                if features:
                    gesture_features.append(features)

            if gesture_features:
                # Calculate average features
                avg_gesture = {}
                for key in gesture_features[0].keys():
                    if key in ['hand_positions']:
                        # Handle list features
                        all_values = [f[key] for f in gesture_features if key in f]
                        if all_values:
                            avg_gesture[key] = all_values  # Keep all positions
                    else:
                        # Handle scalar features
                        all_values = [f[key] for f in gesture_features if key in f]
                        if all_values:
                            avg_gesture[key] = np.mean(all_values)

                combined_features['gesture_features'] = avg_gesture

        # Calculate multimodal similarity scores
        combined_features['multimodal_scores'] = self.calculate_multimodal_scores(
            combined_features['text_features'],
            combined_features['audio_features'],
            combined_features['image_features'],
            combined_features['gesture_features']
        )

        return combined_features

    def calculate_multimodal_scores(self, text_features: Dict, audio_features: Dict,
                                  image_features: Dict, gesture_features: Dict) -> Dict[str, float]:
        """Calculate similarity scores between different modalities."""
        scores = {
            'text_audio_similarity': 0.0,
            'text_image_similarity': 0.0,
            'text_gesture_similarity': 0.0,
            'audio_image_similarity': 0.0,
            'audio_gesture_similarity': 0.0,
            'image_gesture_similarity': 0.0,
            'overall_multimodal_coherence': 0.0
        }

        # Calculate text-audio similarity (based on emotion/intonation)
        if text_features and audio_features:
            # Simple heuristic: check if text sentiment aligns with audio energy
            text_sentiment = 1 if text_features.get('complexity', 0) > 0.5 else 0
            audio_energy = audio_features.get('rms_energy', 0)
            audio_energy_norm = min(audio_energy / 0.1, 1.0)  # Normalize to 0-1

            scores['text_audio_similarity'] = 1.0 - abs(text_sentiment - audio_energy_norm)

        # Calculate text-image similarity (based on content)
        if text_features and image_features:
            # Simple heuristic: check if text mentions visual elements
            text_content = text_features.get('domains', [])
            has_visual_domain = any(domain in ['creative', 'design', 'art'] for domain in text_content)
            has_images = len(image_features) > 0

            scores['text_image_similarity'] = 1.0 if (has_visual_domain and has_images) else 0.0

        # Calculate overall coherence
        valid_scores = [score for score in scores.values() if score > 0]
        if valid_scores:
            scores['overall_multimodal_coherence'] = np.mean(valid_scores)

        return scores

    def process_conversation_multimodal(self, conversation_file: str,
                                      audio_dir: str = None,
                                      image_dir: str = None,
                                      gesture_file: str = None) -> Dict[str, Any]:
        """Process a single conversation with multimodal data."""
        logger.info(f"Processing multimodal conversation: {conversation_file}")

        # Load conversation data
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return {}

        # Process audio data
        audio_data = []
        if audio_dir:
            audio_dir_path = Path(audio_dir)
            if audio_dir_path.exists():
                for audio_file in audio_dir_path.glob("*.wav"):
                    audio_id = str(uuid.uuid4())
                    features = self.extract_audio_features(str(audio_file))

                    if features:
                        audio_data.append({
                            'audio_id': audio_id,
                            'file_path': str(audio_file),
                            'features': features,
                            'start_time': 0,  # Simplified - would need proper timestamp extraction
                            'end_time': features.get('duration', 0)
                        })

        # Process image data
        image_data = []
        if image_dir:
            image_dir_path = Path(image_dir)
            if image_dir_path.exists():
                for image_file in image_dir_path.glob("*.png"):
                    image_id = str(uuid.uuid4())
                    features = self.extract_image_features(str(image_file))

                    if features:
                        image_data.append({
                            'image_id': image_id,
                            'file_path': str(image_file),
                            'features': features,
                            'timestamp': datetime.now().isoformat()  # Simplified
                        })

        # Process gesture data
        gesture_data = []
        if gesture_file:
            try:
                with open(gesture_file, 'r', encoding='utf-8') as f:
                    gesture_raw = json.load(f)

                for gesture_item in gesture_raw:
                    gesture_id = str(uuid.uuid4())
                    features = self.extract_gesture_features(gesture_item)

                    if features:
                        gesture_data.append({
                            'gesture_id': gesture_id,
                            'features': features,
                            'start_time': gesture_item.get('start_time', 0),
                            'end_time': gesture_item.get('end_time', 0)
                        })
            except Exception as e:
                logger.error(f"Failed to load gesture data: {e}")

        # Align all data
        aligned_result = self.align_temporal_data(
            conversation_data, audio_data, image_data, gesture_data
        )

        return aligned_result

    def save_aligned_data(self, aligned_data: Dict[str, Any], output_file: str):
        """Save aligned multimodal data to file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(aligned_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Saved aligned data to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save aligned data: {e}")

    def process_directory(self, conversation_dir: str, audio_dir: str = None,
                         image_dir: str = None, gesture_dir: str = None):
        """Process all conversations in a directory with multimodal data."""
        conversation_dir_path = Path(conversation_dir)

        if not conversation_dir_path.exists():
            logger.error(f"Conversation directory not found: {conversation_dir}")
            return

        # Create output directory
        output_dir = self.output_dir / f"aligned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(exist_ok=True)

        # Process each conversation file
        conversation_files = list(conversation_dir_path.glob("*.json"))
        logger.info(f"Found {len(conversation_files)} conversation files to process")

        for conv_file in conversation_files:
            try:
                # Find corresponding multimodal data
                conv_id = conv_file.stem

                # Look for corresponding audio directory
                conv_audio_dir = None
                if audio_dir:
                    potential_audio_dir = Path(audio_dir) / conv_id
                    if potential_audio_dir.exists():
                        conv_audio_dir = str(potential_audio_dir)

                # Look for corresponding image directory
                conv_image_dir = None
                if image_dir:
                    potential_image_dir = Path(image_dir) / conv_id
                    if potential_image_dir.exists():
                        conv_image_dir = str(potential_image_dir)

                # Look for corresponding gesture file
                conv_gesture_file = None
                if gesture_dir:
                    potential_gesture_file = Path(gesture_dir) / f"{conv_id}_gestures.json"
                    if potential_gesture_file.exists():
                        conv_gesture_file = str(potential_gesture_file)

                # Process conversation
                aligned_data = self.process_conversation_multimodal(
                    str(conv_file), conv_audio_dir, conv_image_dir, conv_gesture_file
                )

                if aligned_data:
                    # Save aligned data
                    output_file = output_dir / f"{conv_id}_aligned.json"
                    self.save_aligned_data(aligned_data, str(output_file))

                    # Store for summary
                    self.aligned_data.append({
                        'conversation_id': conv_id,
                        'output_file': str(output_file),
                        'message_count': len(aligned_data.get('messages', [])),
                        'audio_segments': sum(len(msg.get('audio_segments', [])) for msg in aligned_data.get('messages', [])),
                        'image_segments': sum(len(msg.get('image_segments', [])) for msg in aligned_data.get('messages', [])),
                        'gesture_segments': sum(len(msg.get('gesture_segments', [])) for msg in aligned_data.get('messages', []))
                    })

            except Exception as e:
                logger.error(f"Failed to process {conv_file}: {e}")

        # Save summary
        summary_file = output_dir / "alignment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_conversations': len(self.aligned_data),
                'conversations': self.aligned_data,
                'config': self.config
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Multimodal alignment completed. Results saved to: {output_dir}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Multimodal alignment for conversation data')
    parser.add_argument('--conversation-dir', required=True, help='Directory containing conversation files')
    parser.add_argument('--audio-dir', help='Directory containing audio files (organized by conversation)')
    parser.add_argument('--image-dir', help='Directory containing image files (organized by conversation)')
    parser.add_argument('--gesture-dir', help='Directory containing gesture files')
    parser.add_argument('--output-dir', default='multimodal_aligned', help='Output directory')
    parser.add_argument('--base-dir', default='.', help='Base directory')

    args = parser.parse_args()

    # Initialize multimodal alignment
    aligner = MultimodalAlignment(args.base_dir, args.output_dir)

    # Process directory
    aligner.process_directory(
        args.conversation_dir,
        args.audio_dir,
        args.image_dir,
        args.gesture_dir
    )

    print(f"✅ Multimodal alignment completed!")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
