from safe_gpu import safe_gpu
from time import sleep

no_gpu = True
while no_gpu:
    try:
        safe_gpu.claim_gpus()
        no_gpu = False
    except:
        print("Waiting for free GPU")
        sleep(5)
        pass

import os
from glob import glob
import pandas as pd
import torch
import torchaudio
import argparse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

torch.set_num_threads(16)

def clear_gpu_cache():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

def load_model(xtts_checkpoint_dir):
  clear_gpu_cache()

  xtts_checkpoint = os.path.join(xtts_checkpoint_dir, "best_model.pth")
  xtts_config = os.path.join(xtts_checkpoint_dir, "config.json")
  xtts_vocab = os.path.join(xtts_checkpoint_dir, "vocab.json")
  
  config = XttsConfig()
  config.load_json(xtts_config)
  xtts_model = Xtts.init_from_config(config)
  
  print("Loading XTTS model... from checkpoint: ", xtts_checkpoint)
  
  xtts_model.load_checkpoint(config, checkpoint_dir=xtts_checkpoint_dir  ,checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
  
  if torch.cuda.is_available():
      xtts_model.cuda()

  print("Model Loaded.")
  return xtts_model

# ------------------------------------------------------------------------------ #

SAMPLING_RATE = 24000

if __name__ == "__main__":
  parser = argparse.ArgumentParser (
    description="XTTSv2 inference\n\n",
    formatter_class=argparse.RawTextHelpFormatter,
  )
  parser.add_argument(
    "--model_path",
    "-m",
    type=str,
    default="/mnt/matylda5/xmihol00/xTTSv2/data/run/training/XTTS_v2.0_original_model_files",
    help="Folder where the model files are stored.",
  )
  parser.add_argument(
    "--out_path",
    "-o",
    type=str,
    required=True,
    help="Output path where the generated audio files will be saved.",
  )
  parser.add_argument(
    "--reference_dir",
    "-r",
    type=str,
    required=True,
    help="Path to the directory containing speaker's original recordings.",
  )
  parser.add_argument(
    "--sentences_file",
    "-s",
    type=str,
    required=True,
    help="Path to the file containing sentences to be generated.",
  )
  parser.add_argument(
     "--language",
      "-l",
      type=str,
      default="en",
      help="Language of the generated audio files.",
  )
  parser.add_argument(
     "--num_samples",
      "-n",
      type=int,
      default=3,
      help="Number of samples to generate per sentence.",
  )

  args = parser.parse_args()

  MODEL_PATH = args.model_path
  LANGUAGE = args.language
  SENTENCES_FILE = args.sentences_file
  SPEAKER_REF_DIR = args.reference_dir
  OUT_PATH = args.out_path
  NUM_SAMPLES = args.num_samples
  
  EXTENSIONS = ["wav", "flac", "mp3"]
  
  reference_audio_files = []

  # get all the audio files
  for ext in EXTENSIONS:
    speaker_audio_files = glob(os.path.join(SPEAKER_REF_DIR, f"*.{ext}"))
    reference_audio_files.extend(speaker_audio_files)

  # read the sentences file - each line is a sentence
  with open(SENTENCES_FILE, "r") as read:
    sentences = read.readlines()

  xtts_model = load_model(MODEL_PATH)

  # create output directory
  os.makedirs(OUT_PATH, exist_ok=True)

  gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
      audio_path=reference_audio_files, gpt_cond_len=xtts_model.config.gpt_cond_len, max_ref_length=xtts_model.config.max_ref_len, sound_norm_refs=xtts_model.config.sound_norm_refs)
  
  with open(os.path.join(OUT_PATH,"metadata.txt"), "w") as metadata:
    for i, sentence in enumerate(sentences):
      # sentence_dir = os.path.join(OUT_PATH, f"sentence_{i}")
      # os.makedirs(sentence_dir, exist_ok=True)

      # generate more samples for each sentence
      # for s in range(NUM_SAMPLES):
      out = xtts_model.inference(
            text=sentence,
            language=LANGUAGE,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=xtts_model.config.temperature,
            length_penalty=xtts_model.config.length_penalty,
            repetition_penalty=xtts_model.config.repetition_penalty,
            top_k=xtts_model.config.top_k,
            top_p=xtts_model.config.top_p,
        )
      
      # LINE_TEMPLATE = {0:SPK_ID} | {1:FILE_NAME} | {2:FILE_PATH} | {3:DURATION} | {4:TOOL} | {5:LANG} | {6:GENDER} | {7:TYPE} | {8:ORIGIN} | {9:AUGMENTATION}
      # eg. danuse|danuse_sentence_1.wav/danuse/danuse_sentence_1.wav|2.1|xttsv2|cs|female|spoof|none|none


      audio = torch.tensor(out["wav"])
      
      file_name = f"sample_{i}.wav"
      file_save_path = os.path.join(OUT_PATH, file_name)

      # writing metadata
      metadata.write(f"petrpavel|{file_name}|{file_name}|{round(audio.size(0)/SAMPLING_RATE, 2)}|xttsv2|cs|male|spoof|none|none\n")
      
      print(f"Saving generated audio file as {file_save_path}")
      torchaudio.save(file_save_path, audio.unsqueeze(0), SAMPLING_RATE)
