import numpy as np


class VoiceActivityDetector:
    """
    Detects voice activity in raw audio streams.
    
    Uses an energy-based RMS calculation over sliding windows to identify
    speech onset and determine the end-of-speech timeout (1.5 seconds).
    Suitable for streaming operations over websockets.
    """

    def __init__(self, threshold: float = 300.0) -> None:
        """
        Initializes the VoiceActivityDetector.
        
        Args:
            threshold: RMS energy threshold above which speech is detected.
        """
        self.threshold = threshold
        self.speech_detected = False
        self.silence_duration_samples = 0
        self.last_buffer_len = 0

    def is_speech(self, audio_chunk: bytes) -> bool:
        """Backward compatibility helper to check if a single chunk contains speech."""
        if not audio_chunk:
            return False
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        if len(audio_data) == 0:
            return False
        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        return rms >= self.threshold

    def is_end_of_speech(self, audio_buffer: bytes, sample_rate: int = 16000) -> bool:
        """
        Statefully determines if the user has completed speaking.
        
        Args:
            audio_buffer: Cumulative 16-bit signed PCM mono audio bytes.
            sample_rate: Sample rate of the audio buffer (default: 16000).
            
        Returns:
            True if the user has completed speaking, False otherwise.
        """
        if not audio_buffer:
            return False

        # Reset state if buffer was cleared/reset by the caller
        if len(audio_buffer) < self.last_buffer_len:
            self.speech_detected = False
            self.silence_duration_samples = 0

        audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
        total_samples = len(audio_data)

        # 1. Compute RMS energy of the last 500ms segment
        num_samples_500ms = int(sample_rate * 0.5)
        if total_samples < num_samples_500ms:
            segment = audio_data
        else:
            segment = audio_data[-num_samples_500ms:]

        if len(segment) == 0:
            rms = 0.0
        else:
            rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2))

        # 2. Count new samples added since last call (2 bytes per sample)
        new_samples = (len(audio_buffer) - self.last_buffer_len) // 2
        self.last_buffer_len = len(audio_buffer)

        # 3. Transition VAD states
        if rms >= self.threshold:
            # Active speech detected
            self.speech_detected = True
            self.silence_duration_samples = 0
        else:
            # Silence or background noise detected
            if self.speech_detected:
                if new_samples > 0:
                    self.silence_duration_samples += new_samples

        # 4. Check if silence duration exceeds 1.5 seconds
        silence_limit_samples = int(sample_rate * 1.5)
        if self.speech_detected and self.silence_duration_samples >= silence_limit_samples:
            return True

        return False
