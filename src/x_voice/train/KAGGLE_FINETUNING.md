# Fine-tune X-Voice on Kaggle with two GPUs

This guide runs the stage 1 fine-tuning workflow in a Kaggle Notebook with
Accelerate Distributed Data Parallel (DDP). Kaggle GPU choices and availability
depend on the account and session; select **GPU T4 x2** (or another two-GPU
option) in Notebook **Settings → Accelerator** if it is offered. Kaggle's
notebook documentation describes this setting and the attached input/output
directories: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

## 1. Attach the inputs and enable the session

In the notebook's **Add Input** panel, attach:

- This X-Voice repository, or make it available by cloning it in a cell.
- Your prepared dataset directory, containing `raw/` (saved with
  `datasets.save_to_disk`) or `raw.arrow`, `duration.json`, and `vocab.txt`.
- The X-Voice stage 1 checkpoint (`.safetensors` or supported `.pt` file).
- A local Vocos vocoder folder if you use the example config's
  `vocoder.is_local: true` option.

Set **Internet** on if you need to clone/install the project or download any
inputs. Kaggle input mounts are read-only. Write checkpoints and logs under
`/kaggle/working`, then save a Notebook version or copy the output to durable
storage before the session ends. A live session's working directory is not a
durable checkpoint store.

## 2. Check the available GPUs

Run this notebook cell before installing or starting training:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

assert torch.cuda.is_available(), "Enable a GPU accelerator in Notebook Settings."
assert torch.cuda.device_count() == 2, "This guide expects a two-GPU Kaggle session."
```

If Kaggle assigned one GPU, the two-process launch below will fail. Change the
notebook accelerator if a two-GPU option is available, or use a one-process
launch instead.

## 3. Install the project

If the repository is not already attached as a Kaggle input, clone it:

```bash
%cd /kaggle/working
!git clone https://github.com/sunnyxrxrx/X-Voice.git
```

Install the project and its Python dependencies. Kaggle's image often already
includes CUDA-enabled PyTorch; check the GPU cell again after installation if
pip changed the PyTorch installation.

```bash
%cd /kaggle/working/X-Voice
!apt-get update -qq
!apt-get install -y -qq ffmpeg espeak-ng
!python -m pip install -e .
!python -m pip install accelerate
```

For a repository attached under `/kaggle/input`, use its actual path in `%cd`.
Do not run `src/x_voice/prepare_ipa.sh` on Kaggle: it builds and replaces
system libraries. The provided custom-data training path expects `text` to
already be tokenized/prepared, so training does not need that preparation
script.

## 4. Set paths and check the data

Copy `src/x_voice/configs/XVoice_Custom_Finetune.yaml` into the repository
folder as `XVoice_Kaggle.yaml`. Edit these settings in that copy:

```yaml
datasets:
  name: /kaggle/input/my-data/xvoice-data
  dataset_type: CustomDatasetPath
  audio_type: raw
  root_dir: /kaggle/input/my-data/audio
  num_workers: 2
  language_sampling_alpha: 0.5

model:
  tokenizer: custom
  tokenizer_path: /kaggle/input/my-data/xvoice-data/vocab.txt
  arch:
    languages: ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "id", "it", "ja", "ko", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "th", "vi", "zh"]  # append every dataset language_id
  vocoder:
    is_local: true
    local_path: /kaggle/input/my-models/vocos-mel-24khz

ckpts:
  save_dir: /kaggle/working/xvoice-finetune
  pretrained_path: /kaggle/input/my-models/xvoice-stage1.safetensors
  pretrained_languages: ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "id", "it", "ja", "ko", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "th", "vi", "zh"]
  continue_training: true
```

Set `root_dir: null` if each `audio_path` is already absolute. Each row needs
`audio_path`, `text`, `duration`, and `language_id`; `duration.json` must have
a `duration` list in the same row order. Use `language_id` values listed in
`model.arch.languages`. The trainer accepts audio from 0.3 to 30 seconds and
resamples it to `target_sample_rate`.

If your source has only WAV files, you still need one transcript and one
correct language ID per recording. Assign IDs from the known dataset source or
language folder, and put them in the Arrow rows. A WAV file does not contain a
reliable transcript or language label by itself.

The vocabulary file has one token per line. Preserve the old checkpoint
vocabulary's entries at their original indices and append new tokens. The
trainer then loads the pretrained embedding rows into those existing
positions, leaving newly added rows at their random initialization. The
current `custom` tokenizer uses the repository's IPA v6 text splitting rules;
make sure your `text` strings match those rules. Keep the model architecture
and mel settings consistent with the checkpoint.

Keep the full original checkpoint language list and its order in
`model.arch.languages`, then append your dataset's new language IDs. Set
`ckpts.pretrained_languages` to the original checkpoint list exactly. The
trainer maps existing language embedding rows by ID and initializes added
language rows randomly.

`language_sampling_alpha: 0.5` uses a square-root mix based on the summed row
durations per language. It gives your six languages roughly 29.9% Bolo, 10.5%
Manipuri, 16.1% Santali, 16.1% Chakma, 15.7% Garo, and 11.7% Nagamese. The
sampler draws with replacement, shuffles by epoch, and prints its expected
language probabilities at startup. Use `1.0` for proportions closer to the
raw audio hours, or `null` to turn this balancing off.

Before launching, confirm Kaggle can read all inputs and the output directory
is writable:

```bash
!ls -la /kaggle/input/my-data/xvoice-data
!mkdir -p /kaggle/working/xvoice-finetune
!df -h /kaggle/working
```

## 5. Launch on both GPUs

Accelerate starts one training process per GPU when `--num_processes=2` is
provided. See [Accelerate's launch guide](https://huggingface.co/docs/accelerate/main/basic_tutorials/launch).
Run this from the repository root:

```bash
%cd /kaggle/working/X-Voice
!accelerate launch --multi_gpu --num_processes=2 \
  src/x_voice/train/train.py \
  --config-name XVoice_Kaggle.yaml
```

The repo uses Accelerate's distributed training setup, and `batch_size_per_gpu`
is a per-process budget. The example config uses frame batching with
`batch_size_per_gpu: 19200`; lower that value if either GPU runs out of memory.
Start with fewer dataloader workers (for example, 2 per process) to fit Kaggle's
CPU and RAM limits. The trainer currently does not use `accelerator.autocast`
around its model forward pass, so this launch intentionally leaves mixed
precision disabled. Both GPUs still participate in distributed training.

## 6. Keep and resume checkpoints

The config writes checkpoints under `/kaggle/working/xvoice-finetune`. The
trainer saves numbered checkpoints and `model_last.pt` according to
`save_per_updates` and `last_per_updates`. When a Kaggle session restarts,
attach or copy the previous output into the working directory, then set
`ckpts.save_dir` to the directory containing the training checkpoint. The
trainer checks that directory and resumes from its latest checkpoint. Keep
the expanded `vocab.txt` alongside the checkpoint and use the same language
list and model configuration when resuming.

Before ending the session, use **Save Version** and ensure the checkpoint
directory is included in the notebook output, or copy it to persistent storage.
Kaggle's output storage has quotas, so choose `keep_last_n_checkpoints` and
save intervals with the expected checkpoint size in mind.
