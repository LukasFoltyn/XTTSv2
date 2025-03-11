import numpy as np
import argparse
import torch
import gc

torch.set_num_threads(16)

from trainer.trainer import Trainer, TrainerArgs

from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

def load_tts_samples(metadata_file, language, eval_split_size=0.02, eval_split_max_size=128):
  items = []

  with open(metadata_file, "r", encoding="utf-8") as f:
    for line in f:
      audio, transcript = line.strip().split("|")
      items.append({"audio_file": audio, "text": transcript, "language": language})
  
  np.random.shuffle(items)

  eval_size = min(int(len(items) * eval_split_size), eval_split_max_size)

  return items[eval_size:], items[:eval_size]

def clear_gpu_cache():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

def train_gpt(language, num_epochs, batch_size, grad_acumm, metadata_path, output_path, checkpoint_path, trainer_args):
  RUN_NAME = "GPT_XTTS_FT"
    
  #  Logging parameters
  PROJECT_NAME = "XTTS_trainer"
  DASHBOARD_LOGGER = "tensorboard"
  LOGGER_URI = None
    
  # Training Parameters
  OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False

  START_WITH_EVAL = False  # if True it will star with evaluation
  BATCH_SIZE = batch_size  # set here the batch size
  GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps

  # DVAE files
  DVAE_CHECKPOINT = "./original_model_files/dvae.pth"
  MEL_NORM_FILE = "./original_model_files/mel_stats.pth"
    
  # Paths to the downloaded XTTS v2.0.1 files
  TOKENIZER_FILE = "./original_model_files/vocab.json"
  XTTS_CHECKPOINT = checkpoint_path
  # init args and config
  model_args = GPTArgs(
    max_conditioning_length=132300,  # 6 secs
    min_conditioning_length=66150,  # 3 secs
    debug_loading_failures=False,
    max_wav_length=255995,  # ~11.6 seconds, seconds * sample_rate
    max_text_length=200,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
  )
 
  # define audio config
  audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

  # load training samples
  train_samples, eval_samples = load_tts_samples(metadata_path, language)
  
  # training parameters config
  config = GPTTrainerConfig(
    epochs=num_epochs,
    output_path=output_path,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=PROJECT_NAME,
    run_description="""
      GPT XTTS training
      """,
    dashboard_logger=DASHBOARD_LOGGER,
    logger_uri=LOGGER_URI,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=128,
    print_step=50,
    plot_step=100,
    log_model_step=100,
    save_step=1000,
    save_n_checkpoints=1,
    save_checkpoints=False,
    # target_loss="loss",
    print_eval=False,
    # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
    optimizer="AdamW",
    optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06,  # learning rate
    lr_scheduler="MultiStepLR",
    # it was adjusted accordly for the new step scheme
    lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[
      {
        "text": eval_samples[0]["text"],
        "speaker_wav": eval_samples[0]["audio_file"],
        "language": eval_samples[0]["language"],
      },
      {
        "text": eval_samples[1]["text"],
        "speaker_wav": eval_samples[1]["audio_file"],
        "language": eval_samples[1]["language"],
      }
    ],
  )

  # init the model from config
  model = GPTTrainer.init_from_config(config)

  # init the trainer and 🚀
  trainer = Trainer(
    TrainerArgs(
        **trainer_args,
        skip_train_epoch=False,
        start_with_eval=START_WITH_EVAL,
        grad_accum_steps=GRAD_ACUMM_STEPS,
    ),
    config,
    output_path=output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
  )

  trainer.fit()

  # deallocate VRAM and RAM
  del model, trainer, train_samples, eval_samples
  gc.collect()

  return  XTTS_CHECKPOINT, TOKENIZER_FILE

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="xTTSv2 fine-tuning script\n\n.",
    formatter_class=argparse.RawTextHelpFormatter,
  )
  parser.add_argument(
    "--checkpoint_path",
    "-c",
    type=str,
    help="Path to the checkpoint file to be used as a starting point for fine-tuning.",
    default="./original_model_files/best_model.pth"
  )
  parser.add_argument(
    "--num_epochs",
    "-e",
    type=int,
    help="Number of epochs to train. Default: 5",
    default=5,
  )
  parser.add_argument(
    "--batch_size",
    "-b",
    type=int,
    help="Batch size. Default: 4", # getting error for batch of 16
    default=4,
  )
  parser.add_argument(
    "--grad_acumm",
    "-g",
    type=int,
    help="Grad accumulation steps. Default: 1",
    default=1,
  )
  parser.add_argument(
    "--out_path",
    "-o",
    type=str,
    required=True,
    help="Output path (where data and checkpoints will be saved).",
  )
  parser.add_argument(
    "--metadata_path",
    "-m",
    type=str,
    required=True,
    help="Path to the metadata file containing information about the dataset to be used for fine-tuning.",
  )
  parser.add_argument(
    "--language",
    "-l",
    type=str,
    default="cs",
    help="Language of the dataset to be used for fine-tuning. Default: cs",
  )
  
  parser.add_argument(
    "--continue_path",
    type=str,
    default="",
    help="Path to continue training from a checkpoint.",
  )
  parser.add_argument(
    "--restore_path",
    type=str,
    default="",
    help="Path to restore a model from a checkpoint.",
  )
  parser.add_argument(
    "--group_id",
    type=str,
    default="",
    help="Group ID for distributed training.",
  )
  parser.add_argument(
    "--use_ddp",
    type=str,
    default="false",
    help="Whether to use Distributed Data Parallel (DDP).",
  )
  parser.add_argument(
    "--rank",
    type=int,
    default=0,
    help="Rank of the current process in distributed training.",
  )

  args = parser.parse_args()

  trainer_args = {
    "continue_path": args.continue_path,
    "restore_path": args.restore_path,
    "group_id": args.group_id,
    "use_ddp": args.use_ddp.lower() == "true",
    "rank": args.rank,
}

  print("Training model...")
  clear_gpu_cache()
  try:
      train_gpt(
          args.language, args.num_epochs, args.batch_size, args.grad_acumm, args.metadata_path, args.out_path, args.checkpoint_path, trainer_args)
  except Exception as e:
      print(f"An error occurred during training: {e}")
      exit(1)
  finally:
      clear_gpu_cache()

  print("Model training done.")
