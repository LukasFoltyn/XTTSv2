import torch
import glob
import os
import tqdm
import sounddevice as sd
import tty
import sys
import termios
import queue
import threading
import string
import torchaudio
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig

torch.serialization.add_safe_globals([XttsConfig])
torch.serialization.add_safe_globals([XttsAudioConfig])
torch.serialization.add_safe_globals([BaseDatasetConfig])
torch.serialization.add_safe_globals([XttsArgs])

def clear_gpu_cache():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

def load_model(ckpt_path="./original_model_files/best_model.pth"):
  clear_gpu_cache()

  xtts_checkpoint = ckpt_path
  xtts_config = "./original_model_files/config.json"
  xtts_vocab = "./original_model_files/vocab.json"
  
  config = XttsConfig()
  config.load_json(xtts_config)
  xtts_model = Xtts.init_from_config(config)
  
  print("Loading XTTS model... from checkpoint: ", xtts_checkpoint)
  
  xtts_model.load_checkpoint(config, checkpoint_dir="./original_model_files/"  ,checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
  
  if torch.cuda.is_available():
      xtts_model.cuda()

  print("Model Loaded.")
  return xtts_model

# ------------------------------------------------------------------------------ #
SAMPLING_RATE = 24000

text_queue = queue.Queue()

refs = [
    "./DF_E_2374016.wav",
    "./DF_E_2374585.wav",
    "./DF_E_2381388.wav",
  ]

model = load_model()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=refs)

def get_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def in_thread_fn():
  word = ""
  print("Start typing (press ~ to quit):")
  while True:
    ch = get_char()
    print(ch, end='', flush=True)
    if ch == '~':
      if word:
        text_queue.put(word)
      text_queue.put(None)
      break
    if ch == '.':
      if word:
          text_queue.put(word+"_")
          word = ""
      text_queue.put(".")
    elif ch in string.whitespace:
      if word:
        # if len(word) >= 5:
        text_queue.put(word)
        word = ""
        # else:
          # word += ch
    else:
        word += ch

def out_thread_fn():
  stream = sd.OutputStream(samplerate=SAMPLING_RATE, channels=1, dtype='float32')
  stream.start()

  prev_tokens =[]
  buffer=""

  while True:
    text = text_queue.get()
    if text is None:
      break
    if text == ".":
      buffer = ""
      prev_tokens = []
      continue

    buffer = (text if buffer == ""  else buffer + " " + text).lower() 
      
    chunk, prev_tokens = model.inference_stream_hokus_pokus(
      buffer,
      "en",
      gpt_cond_latent,
      speaker_embedding,
      context_tokens=prev_tokens
    )

    stream.write(chunk.squeeze().cpu().numpy())

    text_queue.task_done()
  
  stream.stop()
  stream.close()


if __name__ == "__main__":
  
  # EXPERIMENT_LANGS = ["en"]
  # EXPERIMENT_REF_DIR = "/mnt/matylda5/xfolty17/audio_playground"
  # EXPERIMENT_OUT_DIR = "/mnt/matylda5/xfolty17/knn_experiments"

  # EXPERIMENTS = ["dtw"]


  # for iference_id, experiment_name in enumerate(EXPERIMENTS):
  #   for lang in EXPERIMENT_LANGS: 
  #     texts_to_gen = glob.glob(os.path.join(EXPERIMENT_REF_DIR, lang, "to_generate", "*.txt"))
  #     ref_audios = glob.glob(os.path.join(EXPERIMENT_REF_DIR, lang, "reference", "*.wav"))

  #     ckpt = "./checkpoints/parczech-balanced/best_model.pth" if lang == "cz" else "./checkpoints/original/best_model.pth"

  #     model = load_model(ckpt_path=ckpt)

  #     inferences = [model.inference_stream_dtw]

  #     for spk_ref in [ref_audios[3]]:
  #       print(f"Processing {spk_ref}...")
  #       spk_id = spk_ref.split("_")[-1].replace(".wav", "")

  #       spk_dir = os.path.join(EXPERIMENT_OUT_DIR, lang, spk_id, f"experiment_{experiment_name}")
  #       os.makedirs(spk_dir, exist_ok=True)

  #       gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=spk_ref)

  #       for text_file in tqdm.tqdm(texts_to_gen):
  #         with open(text_file, "r") as f:
  #           wav_path = os.path.join(spk_dir, os.path.basename(text_file).replace(".txt", ".wav"))
  #           text = f.read().strip()
            
  #           chunks = inferences[iference_id](
  #             text,
  #             "cs" if lang == "cz" else "en",
  #             gpt_cond_latent,
  #             speaker_embedding,
  #           )

  #           if experiment_name == "whole_sentence":
  #             torchaudio.save(wav_path, torch.tensor(chunks["wav"]).unsqueeze(0).cpu(), 24000)
  #           else:
  #             wav_chuncks = []
  #             for i, chunk in enumerate(chunks):
  #               wav_chuncks.append(chunk)

  #             wav = torch.cat(wav_chuncks, dim=0)
  #             torchaudio.save(wav_path, wav.squeeze().unsqueeze(0).cpu(), 24000)

  in_t = threading.Thread(target=in_thread_fn)
  out_t = threading.Thread(target=out_thread_fn)

  in_t.start()
  out_t.start()
  
  in_t.join()
  out_t.join()


  

    

    # wav_chuncks = []
    # for i, chunk in enumerate(chunks):
    #   # torchaudio.save(f"chunk{i}.wav", chunk.squeeze().unsqueeze(0).cpu(), 24000)
    #   wav_chuncks.append(chunk)



    # wav = torch.cat(wav_chuncks, dim=0)

    # print("Saving final wav of shape:", wav.shape)
    # torchaudio.save(f"xtts_streaming_{inference_idx}.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
