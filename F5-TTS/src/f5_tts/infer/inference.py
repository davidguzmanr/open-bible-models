"""
Inference script for Jupyter notebooks.
Loads the model/checkpoint once and allows generating multiple texts without reloading.
"""
import random
import sys
from importlib.resources import files
from pathlib import Path

import soundfile as sf
import tqdm
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model.utils import seed_everything


class F5TTSInference:
    """
    F5-TTS Inference class for Jupyter notebooks.
    Loads model and vocoder once, then allows generating multiple texts.
    
    Example:
        # Load once
        tts = F5TTSInference(
            model="F5TTS_v1_Base",
            ref_audio="path/to/ref_audio.wav",
            ref_text="Reference text here."
        )
        
        # Generate multiple texts
        audio1, sr = tts.generate("First text to generate")
        audio2, sr = tts.generate("Second text to generate")
    """
    
    def __init__(
        self,
        model="F5TTS_v1_Base",
        ckpt_file="",
        vocab_file="",
        ref_audio=None,
        ref_text=None,
        ode_method="euler",
        use_ema=True,
        vocoder_local_path=None,
        device=None,
        hf_cache_dir=None,
        **inference_kwargs
    ):
        """
        Initialize F5-TTS inference engine.
        
        Args:
            model: Model name (e.g., "F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base")
            ckpt_file: Path to checkpoint file. If empty, downloads from HuggingFace.
            vocab_file: Path to vocab file. If empty, uses default.
            ref_audio: Path to reference audio file (can be set later with set_reference)
            ref_text: Transcript for reference audio (can be set later with set_reference)
            ode_method: ODE solver method (default: "euler")
            use_ema: Whether to use EMA model (default: True)
            vocoder_local_path: Local path to vocoder (default: None, downloads from HuggingFace)
            device: Device to run on (default: auto-detect)
            hf_cache_dir: HuggingFace cache directory
            **inference_kwargs: Default inference parameters (nfe_step, cfg_strength, speed, etc.)
        """
        # Load model configuration
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self.model_name = model
        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
        self.ode_method = ode_method
        self.use_ema = use_ema

        # Set device
        if device is not None:
            self.device = device
        else:
            import torch
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        print(f"Loading model on device: {self.device}")
        
        # Load vocoder
        print("Loading vocoder...")
        self.vocoder = load_vocoder(
            self.mel_spec_type,
            vocoder_local_path is not None,
            vocoder_local_path,
            self.device,
            hf_cache_dir
        )

        # Determine checkpoint path
        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

        # Override for previous models
        if model == "F5TTS_Base":
            if self.mel_spec_type == "vocos":
                ckpt_step = 1200000
            elif self.mel_spec_type == "bigvgan":
                model = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif model == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000

        if not ckpt_file:
            print(f"Downloading checkpoint from HuggingFace...")
            ckpt_file = str(
                cached_path(
                    f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}",
                    cache_dir=hf_cache_dir
                )
            )

        # Load TTS model
        print("Loading TTS model...")
        self.ema_model = load_model(
            model_cls,
            model_arc,
            ckpt_file,
            self.mel_spec_type,
            vocab_file,
            self.ode_method,
            self.use_ema,
            self.device
        )
        print("Model loaded successfully!")

        # Set reference audio/text if provided
        self.ref_audio_processed = None
        self.ref_text_processed = None
        if ref_audio is not None and ref_text is not None:
            self.set_reference(ref_audio, ref_text)

        # Default inference parameters
        self.default_kwargs = {
            "target_rms": 0.1,
            "cross_fade_duration": 0.15,
            "nfe_step": 32,
            "cfg_strength": 2.0,
            "sway_sampling_coef": -1.0,
            "speed": 1.0,
            "fix_duration": None,
            **inference_kwargs
        }

    def set_reference(self, ref_audio, ref_text=""):
        """
        Set or update reference audio and text.
        This preprocesses the reference audio (can be time-consuming on first call).
        
        Args:
            ref_audio: Path to reference audio file
            ref_text: Transcript for reference audio (empty string to auto-transcribe)
        """
        print(f"Preprocessing reference audio: {ref_audio}")
        self.ref_audio_processed, self.ref_text_processed = preprocess_ref_audio_text(
            ref_audio, ref_text
        )
        print(f"Reference text: {self.ref_text_processed}")

    def generate(
        self,
        gen_text,
        ref_audio=None,
        ref_text=None,
        seed=None,
        remove_silence=False,
        output_file=None,
        **kwargs
    ):
        """
        Generate speech from text.
        
        Args:
            gen_text: Text to generate speech from
            ref_audio: Optional reference audio (if None, uses previously set reference)
            ref_text: Optional reference text (if None, uses previously set reference)
            seed: Random seed for generation (None for random)
            remove_silence: Whether to remove silence from output
            output_file: Optional path to save output audio file
            **kwargs: Override default inference parameters (nfe_step, cfg_strength, speed, etc.)
        
        Returns:
            Tuple of (audio_array, sample_rate, spectrogram)
            - audio_array: numpy array of audio samples
            - sample_rate: sample rate of audio (typically 24000)
            - spectrogram: numpy array of mel spectrogram
        """
        # Check if reference is set
        if ref_audio is not None or ref_text is not None:
            if ref_audio is None or ref_text is None:
                raise ValueError("Both ref_audio and ref_text must be provided together, or use set_reference() first")
            self.set_reference(ref_audio, ref_text)
        elif self.ref_audio_processed is None or self.ref_text_processed is None:
            raise ValueError(
                "Reference audio/text not set. Provide ref_audio and ref_text in __init__ or generate(), "
                "or call set_reference() first."
            )

        # Set random seed
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)

        # Merge inference parameters
        infer_params = {**self.default_kwargs, **kwargs}

        # Generate audio
        wav, sr, spec = infer_process(
            self.ref_audio_processed,
            self.ref_text_processed,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=print,
            progress=tqdm,
            device=self.device,
            **infer_params
        )

        # Save to file if requested
        if output_file is not None:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_file, wav, sr)
            
            if remove_silence:
                remove_silence_for_generated_wav(output_file)
            
            print(f"Audio saved to: {output_file}")

        return wav, sr, spec

    def save_spectrogram(self, spectrogram, file_path):
        """Save spectrogram to file."""
        save_spectrogram(spectrogram, file_path)
        print(f"Spectrogram saved to: {file_path}")


# Convenience function for quick usage
def create_inference_engine(
    model="F5TTS_v1_Base",
    ref_audio=None,
    ref_text=None,
    ckpt_file="",
    **kwargs
):
    """
    Convenience function to create an inference engine.
    
    Example:
        tts = create_inference_engine(
            model="F5TTS_v1_Base",
            ref_audio="path/to/ref.wav",
            ref_text="Reference text here."
        )
        audio, sr, spec = tts.generate("Text to generate")
    """
    return F5TTSInference(
        model=model,
        ref_audio=ref_audio,
        ref_text=ref_text,
        ckpt_file=ckpt_file,
        **kwargs
    )

