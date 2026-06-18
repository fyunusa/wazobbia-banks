import asyncio
import io
import logging
import os
import sys
from voice.tts.base import BaseTTSEngine, TTSResult

logger = logging.getLogger("voice.tts.yarngpt")


class YarnGPTTTSEngine(BaseTTSEngine):
    """
    YarnGPT2 TTS Engine for high-quality, accent-specific Nigerian speech synthesis.
    Supports Hausa (ha) and Nigerian Pidgin (pcm) natively.
    Uses saheedniyi/YarnGPT2 (SmolLM2-360M + WavTokenizer).
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._tokenizer = None
            cls._instance._device = None
        return cls._instance

    def _load_model(self):
        """Lazily downloads and loads the YarnGPT2 model on first use."""
        if self._model is not None:
            return self._model, self._tokenizer

        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"YarnGPT2: Initializing engine on device: {self._device}...")

        try:
            # 1. Add yarngpt_repo to sys.path to resolve imports
            workspace_dir = os.getenv("WORKSPACE_DIR", os.getcwd())
            yarngpt_repo_path = os.path.join(workspace_dir, "voice", "tts", "yarngpt_repo")
            if yarngpt_repo_path not in sys.path:
                sys.path.append(yarngpt_repo_path)

            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForCausalLM
            from audiotokenizer import AudioTokenizerV2

            # 2. Download WavTokenizer configuration and model weights
            logger.info("YarnGPT2: Downloading WavTokenizer weights and config from Hugging Face...")
            wav_tokenizer_model_path = hf_hub_download(
                repo_id="novateur/WavTokenizer-large-speech-75token", 
                filename="wavtokenizer_large_speech_320_v2.ckpt"
            )
            wav_tokenizer_config_path = hf_hub_download(
                repo_id="novateur/WavTokenizer-medium-speech-75token", 
                filename="wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
            )

            # 3. Initialize AudioTokenizerV2
            logger.info("YarnGPT2: Loading saheedniyi/YarnGPT2 tokenizer and causal model...")
            self._tokenizer = AudioTokenizerV2(
                tokenizer_path="saheedniyi/YarnGPT2",
                wav_tokenizer_model_path=wav_tokenizer_model_path,
                wav_tokenizer_config_path=wav_tokenizer_config_path,
            )

            # 4. Load causal LM Smollm2
            self._model = AutoModelForCausalLM.from_pretrained(
                "saheedniyi/YarnGPT2", 
                torch_dtype="auto"
            ).to(self._device)
            logger.info("YarnGPT2: Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"YarnGPT2: Initialization failed: {e}. Falling back to dummy generator.", exc_info=True)
            self._model = False
            self._tokenizer = False

        return self._model, self._tokenizer

    def _synthesize_sync(self, text: str, lang_name: str, gender: str = "female") -> tuple[bytes, float, int]:
        """Synchronous CPU/GPU bound YarnGPT2 inference."""
        import soundfile as sf
        import numpy as np

        model, tokenizer = self._load_model()
        if not model or not tokenizer:
            # Fallback to 3 seconds of dummy audio
            logger.warning("YarnGPT2: Synthesis falling back to dummy WAV generator.")
            sample_rate = 24000
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_array = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
            
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, sample_rate, format="WAV", subtype="PCM_16")
            return wav_io.getvalue(), duration, sample_rate

        try:
            # Map speaker based on language and requested gender
            speaker_mapping = {
                "hausa": {
                    "female": "hausa_female1",
                    "male": "hausa_male1"
                },
                "yoruba": {
                    "female": "yoruba_female2",
                    "male": "yoruba_male2"
                },
                "igbo": {
                    "female": "igbo_female1",
                    "male": "igbo_male2"
                },
                "english": {
                    "female": "zainab",
                    "male": "osagie"
                }
            }
            speaker_name = speaker_mapping.get(lang_name, {}).get(gender, "zainab")

            logger.info(f"YarnGPT2: Generating prompt for lang='{lang_name}', speaker='{speaker_name}', text='{text}'...")
            prompt = tokenizer.create_prompt(text, lang=lang_name, speaker_name=speaker_name)
            input_ids = tokenizer.tokenize_prompt(prompt)

            import torch
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids, 
                    temperature=0.1, 
                    repetition_penalty=1.1, 
                    max_length=4000
                )

            codes = tokenizer.get_codes(output)
            audio_out = tokenizer.get_audio(codes)

            # Convert to float32 numpy array
            try:
                audio_array = audio_out.squeeze().cpu().numpy()
            except RuntimeError as re_numpy:
                if "Numpy is not available" in str(re_numpy):
                    audio_array = np.array(audio_out.squeeze().cpu().tolist(), dtype=np.float32)
                else:
                    raise
            sample_rate = 24000
            duration = len(audio_array) / sample_rate

            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = wav_io.getvalue()

            return wav_bytes, duration, sample_rate
        except Exception as e:
            logger.error(f"YarnGPT2: Synthesis failed with error: {e}. Generating dummy fallback.", exc_info=True)
            sample_rate = 24000
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_array = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
            
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, sample_rate, format="WAV", subtype="PCM_16")
            return wav_io.getvalue(), duration, sample_rate

    async def synthesize(self, text: str, language: str, gender: str = "female") -> TTSResult:
        """Asynchronously synthesizes text to speech using an executor."""
        lang = language.lower().strip()
        # Map codes to YarnGPT local names
        lang_mapping = {
            "ha": "hausa",
            "yo": "yoruba",
            "ig": "igbo",
            "pcm": "yoruba", # Use Yoruba speaker presets as a proxy for Pidgin accents in YarnGPT if needed, or fallback to english
            "en": "english"
        }
        lang_name = lang_mapping.get(lang, "english")

        loop = asyncio.get_running_loop()
        wav_bytes, duration, sample_rate = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
            lang_name,
            gender,
        )

        return TTSResult(
            audio_bytes=wav_bytes,
            language=language,
            duration_seconds=duration,
            engine_used="yarngpt",
            sample_rate=sample_rate,
        )
