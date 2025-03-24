import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def clear_gpu_cache():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

def load_model():
  clear_gpu_cache()

  xtts_checkpoint = "./original_model_files/best_model.pth"
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

if __name__ == "__main__":
  

  refs = [
    "./DF_E_2374016.wav",
    "./DF_E_2374585.wav",
    "./DF_E_2381388.wav",
  ]
  model = load_model()  

  print("Computing speaker latents...")
  gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=refs)

  print("Inference...")
  chunks = model.inference_stream(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    # stream_chunk_size=0,
  )

  wav_chuncks = []
  for i, chunk in enumerate(chunks):
    torchaudio.save(f"chunk{i}.wav", chunk.squeeze().unsqueeze(0).cpu(), 24000)
    wav_chuncks.append(chunk)
  
  wav = torch.cat(wav_chuncks, dim=0)
  torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
