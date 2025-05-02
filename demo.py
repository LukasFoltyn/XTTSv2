# F5 imports
from pydub import AudioSegment, silence
import argparse
import os
import sys
import librosa
from pathlib import Path
from omegaconf import OmegaConf

sys.path.append(os.path.abspath('./F5-TTS/src'))


from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT

######################## F5-TTS BASE EN (pretrained) ########################
CKPT_FILE = "/homes/eva/xl/xluner01/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_Base/model_1200000.safetensors"
MODEL_CFG = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/configs/F5TTS_Base_train.yaml"
VOCAB_FILE = "/mnt/matylda4/xluner01/F5-TTS/src/f5_tts/infer/examples/vocab.txt"

LANG = "en"

SPEED = speed # 1.0
CROSSFADE = cross_fade_duration
VOCODER_LOCAL_PATH = "/mnt/matylda4/xluner01/F5-TTS/checkpoints/vocos-mel-24khz"
VOCODER_NAME = mel_spec_type # vocos
LOAD_VOCODER_FROM_LOCAL = True

FS = 24000

ARGS = None

def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio using the ??? model.")
    
    # reference audio and text will be the same for all generated audio
    parser.add_argument(
        "--audio_ref_file", 
        type=str, 
        required=True, 
        help="Path to the original reference audio."
    )
    parser.add_argument(
        "--text_ref_file", 
        type=str, 
        required=True, 
        help="Path to the original reference text."
    )
    parser.add_argument(
        "--audio_gen_output_folder", 
        type=str, 
        default="./generated_audio",
        help="Path to the folder where the generated audio files will be stored."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="If set, print additional information during evaluation."
    )
    parser.add_argument(
        "--model",
        type=str, 
        choices=["f5", "xtts"],
        help="Available options: F5 (f5), XTTS-V2 (xtts)."
    )

    return parser.parse_args()

def load_f5_model():
    global vocoder, model_cls, model_cfg, ema_model
    
    vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=True, local_path=VOCODER_LOCAL_PATH)
    
    model_cls = DiT
    model_cfg = OmegaConf.load(MODEL_CFG).model.arch
  
    print("Initializing model: F5-TTS")
    print(f"Initializing vocoder: {VOCODER_NAME}")
    
    ema_model = load_model(model_cls, model_cfg, CKPT_FILE, mel_spec_type=VOCODER_NAME, vocab_file=VOCAB_FILE)

def run_inference(ref_audio, ref_text, gen_text, gen_audio_path, fix_dur=fix_duration):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = {"main": main_voice}
    voices["main"]["ref_audio"], voices["main"]["ref_text"] = preprocess_ref_audio_text(
        voices["main"]["ref_audio"], voices["main"]["ref_text"]
    )

    ref_audio_ = voices["main"]["ref_audio"]
    ref_text_ = voices["main"]["ref_text"]
    gen_text_ = gen_text.strip()
    
    # fix_duration experiment (if fix_dur is set and not equal to the default fix_duration, use it)
    if fix_dur != fix_duration:
        final_fix_duration = fix_dur
    else:
        final_fix_duration = fix_duration
    
    wave, sample_rate, spectrogram = infer_process(
        ref_audio_,
        ref_text_,
        gen_text_,
        ema_model,
        vocoder,
        mel_spec_type=VOCODER_NAME,
        target_rms=target_rms,
        cross_fade_duration=CROSSFADE,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=SPEED,
        fix_duration=final_fix_duration,
    )

def remove_silence_and_get_duration_from_wav(src_file):
    aseg = AudioSegment.from_file(src_file)
    non_silent_segs = silence.split_on_silence(aseg, min_silence_len=200, silence_thresh=-50, seek_step=10)
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export("tmp_without_silence.wav", format="wav")
    y, sr = librosa.load(src_file, sr=None)  # `sr=None` preserves original sampling rate
    duration = librosa.get_duration(y=y, sr=sr)
    os.remove("tmp_without_silence.wav")
    return duration

def experiment_per_chunk_fix_duration(text_to_gen):
    ref_audio = ARGS.audio_ref_file
    ref_text = ARGS.text_ref_file
    
    # Load reference text
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()
            
    # Calculate the duration of the reference audio without silence
    ref_audio_duration = remove_silence_and_get_duration_from_wav(ref_audio)
    
    output_wav_path = Path(ARGS.audio_gen_output_folder, f'output_chunk.wav')
    
    # Based on the length of the word, set the fix_duration value
    if len(text_to_gen) <= 5:
        audio_duration = ref_audio_duration + 0.7
    elif len(text_to_gen) <= 10:
        audio_duration = ref_audio_duration + 1.0
    else:
        audio_duration = ref_audio_duration + 1.5
        
    # Run inference
    # run_inference(ref_audio, ref_text, chunk, output_wav_path)
    run_inference(ref_audio, ref_text, text_to_gen, output_wav_path, fix_dur=audio_duration)
    
    # Load generated audio
    generated_audio = AudioSegment.from_wav(output_wav_path)

    return generated_audio

if __name__ == "__main__":   
    ARGS = parse_args()
    
    # Load the F5-TTS model
    load_f5_model()
    
    text = "Hello"
    experiment_per_chunk_fix_duration(text_to_gen=text)


# import torch
# import sounddevice as sd
# import queue
# import torchaudio
# import os

# import sys
# import tty
# import termios

# sys.path.append(os.path.abspath('./XTTSv2'))

# from TTS.config.shared_configs import BaseDatasetConfig
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig

# torch.serialization.add_safe_globals([XttsConfig])
# torch.serialization.add_safe_globals([XttsAudioConfig])
# torch.serialization.add_safe_globals([BaseDatasetConfig])
# torch.serialization.add_safe_globals([XttsArgs])


# def load_xttsv2(ckpt_path="./XTTSv2/original_model_files/best_model.pth"):
#   xtts_checkpoint = ckpt_path
#   xtts_config = "./XTTSv2/original_model_files/config.json"
#   xtts_vocab = "./XTTSv2/original_model_files/vocab.json"
  
#   config = XttsConfig()
#   config.load_json(xtts_config)
#   xtts_model = Xtts.init_from_config(config)
  
#   print("Loading XTTS model... from checkpoint: ", xtts_checkpoint)
  
#   xtts_model.load_checkpoint(config, checkpoint_dir="./XTTSv2/original_model_files/"  ,checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
  
#   if torch.cuda.is_available():
#       xtts_model.cuda()

#   print("Model Loaded.")
#   return xtts_model


# def getchar():
#   fd = sys.stdin.fileno()
#   old_settings = termios.tcgetattr(fd)
#   try:
#     tty.setraw(fd)
#     ch = sys.stdin.read(1)
#   finally:
#     termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#   return ch


# # ------------------------------------------------------------------------------ #
# SAMPLING_RATE = 24000

# text_queue = queue.Queue()

# refs = [
#   "./XTTSv2/DF_E_2374016.wav",
#   "./XTTSv2/DF_E_2374585.wav",
#   "./XTTSv2/DF_E_2381388.wav",
# ]

# if __name__ == "__main__":
#   model = load_xttsv2()

#   gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=refs)

#   wav_chuncks = []

#   for t in ["Michael", "is attracted ", "to blond ", "women and", "he likes", "them", "very", "much."]:
#     chunk = model.inference_stream_hokus_pokus(
#               t,
#               "en",
#               gpt_cond_latent,
#               speaker_embedding,
#             )
    
#     wav_chuncks.append(chunk)

#   wav = torch.cat(wav_chuncks, dim=0)
#   torchaudio.save(f"xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
