# Training

## Fine-tune on a custom multilingual dataset

For a two-GPU Kaggle run, follow the [Kaggle fine-tuning guide](KAGGLE_FINETUNING.md).

For a custom dataset, use the stage 1 training script and the example
[`XVoice_Custom_Finetune.yaml`](../configs/XVoice_Custom_Finetune.yaml). The dataset directory must contain either a Hugging Face dataset saved as
`raw/` or an Arrow file named `raw.arrow`, plus `duration.json` with a
`duration` array in the same row order. Each row needs:

```text
audio_path: path to audio (mono or multichannel; it will be converted to mono)
text: text already converted to the representation used by your tokenizer
duration: audio length in seconds
language_id: language code listed in model.arch.languages
```

If the source data contains only WAV files, make a metadata manifest before
training. A WAV file alone does not provide the transcript needed for TTS;
obtain or create a transcript for each recording, and attach its language ID
from trusted dataset metadata or the known language folder. Do not infer the
language ID from the speaker or filename unless your dataset defines that
convention.

Audio should be between 0.3 and 30 seconds. Relative `audio_path` values are
resolved against `datasets.root_dir`. Use 24 kHz-compatible audio; the loader
resamples it to the configured `target_sample_rate`.

Set `model.tokenizer: custom` and point `model.tokenizer_path` to a UTF-8
`vocab.txt` with one token per line. To expand a pretrained vocabulary, keep
every old token in its original line/ID and append new tokens after them. The
trainer copies all pretrained text embedding rows (including the padding row)
into the same indices; extra rows retain the model's random initialization.
Reordering or removing old tokens will give IDs the wrong pretrained
embeddings. The vocabulary must tokenize `text` with the same rules used by
`x_voice.model.utils.str_to_list_ipa_v*`; `custom` selects the vocabulary file
but does not define new tokenization rules: `custom` currently uses the IPA v6
text splitting rules. Update `model.arch.languages` to cover all `language_id`
values.

For new languages, keep every language in the pretrained checkpoint in the
same order in `model.arch.languages`, then append the new `language_id` values.
Set `ckpts.pretrained_languages` to the checkpoint's original language list
and order. Checkpoint loading maps the old language embedding rows by language
ID, copies the unknown/dropout row to the new final row, and leaves only added
language rows randomly initialized. The custom fine-tuning config starts with
the repository's base language list in both settings; append your data IDs to
`model.arch.languages` only.

To control the language mix, set `datasets.language_sampling_alpha` in the
config. It uses the sum of each language's row `duration` values to set its
sampling probability, then shuffles and samples examples with replacement
each epoch. `0.5` samples languages proportional to the square root of their
hours, `1.0` follows the raw hour proportions, and `0.0` gives each language
an equal share. The custom fine-tuning example uses `0.5`; for your six
languages this is approximately Bolo 29.9%, Manipuri 10.5%, Santali 16.1%,
Chakma 16.1%, Garo 15.7%, and Nagamese 11.7%. Probabilities are printed when
training starts. Since sampling is with replacement, some rows may repeat and
others may not appear in a given epoch. Set the value to `null` to disable
language reweighting and use the existing shuffle behavior.

Adjust the example model dimensions and mel settings to match the checkpoint.
Set the pretrained checkpoint and output directory, then run from the repo
root:

```bash
accelerate config
accelerate launch src/x_voice/train/train.py --config-name XVoice_Custom_Finetune.yaml
```

For a stage 1 checkpoint that should be fine-tuned immediately, leave
`ckpts.continue_training: true`; this trains the loaded backbone instead of
freezing it. Keep `false` only when you intend to use the repository's
pretrained freeze behavior.

- Check your FFmpeg installation:
  ```bash
  ffmpeg -version
  ```
  If not found, install it first (or skip this if you know other backends are available).

- Check your ESpeak-ng installation:
  ```bash
  espeak-ng --version 
  ```
  If not found, run `bash x_voice/prepare_ipa.sh` first.

- Download our dataset from [Hugging Face](https://huggingface.co/datasets/XRXRX/X-Voice-Dataset-Train) or [ModelScope](https://modelscope.cn/datasets/sunnyxrxrx/X-Voice-Dataset-Train) and organize it in the following structure:
  ```
  x_voice/
  ├── wavs/
  │   ├── bg/
  │   ├── ...
  │   └── zh/
  ├── csvs/
  │   ├── metadata_bg_voxpopuli.csv
  │   ├── ...
  │   └── metadata_zh_emilia.csv
  └── csvs_stage2/
      ├── metadata_bg_voxpopuli.csv
      ├── ...
      └── metadata_zh_emilia.csv
  ```
- Optional: Download the [original F5-TTS checkpoint](https://huggingface.co/SWivid/F5-TTS/blob/main/F5TTS_v1_Base/model_1250000.safetensors), so that we can load the pretrained DiT module during training and make convergence faster.

## Stage 1  Training
### 1. Data Preparation
In this part, we first convert transcripts into the unified IPA representation.
```bash
python src/x_voice/train/datasets/prepare_ipa.py \
    --dataset_name XVoice_Dataset \
    --inp_dir path/to/train-set/of/x-voice/ \
    --dnsmos 2.0 --check_exists
```
| Type | Argument | Description |
| --- | --- | --- |
| Required | `--dataset_name` | The dataset name to use (for example, `XVoice_Dataset`). This value is used in later training configurations. |
| Required | `--inp_dir` | The **absolute path** to the downloaded X-Voice dataset, for example `/home/datasets/x_voice/`. |
| Optional | `--dnsmos` | A DNSMOS threshold. Entries below this value are excluded from training. |
| Optional | `--check_exists` | Checks whether audio files exist under `x_voice/wavs` to avoid file-not-found errors during training. |



### 2. Configure training parameters
Once your datasets are prepared, we can start the training process.

Create a `.yaml` file under `./src/x_voice/config`, for example `XVoice_Base_Stage1.yaml`, refer to [this example](https://github.com/sunnyxrxrx/X-Voice/blob/main/src/x_voice/configs/XVoice_Base_Stage1.yaml)

Modify the following parameters:
- `datasets.name`: the dataset name just prepared
- `model.name`: your custom model name
- `vocoder.is_local` and `vocoder.local_path`: specify local vocoder path if you have downloaded it in advance
- `ckpts.pretrained_path`: provide the path to the downloaded original F5-TTS checkpoints

### 2. Training script used for pretrained model

```bash
# setup accelerate config, e.g. use multi-gpu ddp, mixed-precision. bf16 mixed-precision is recommended.
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config

# pass in the .yaml file you just configured
accelerate launch src/x_voice/train/train.py  --config-name XVoice_Base_Stage1.yaml

# possible to overwrite accelerate and hydra config
accelerate launch --mixed_precision=bf16 src/x_voice/train/train.py --config-name XVoice_Base_Stage1.yaml ++datasets.batch_size_per_gpu=19200
```



### 3. W&B Logging

The `wandb/` dir will be created under path you run training/finetuning scripts.

By default, the training script does NOT use logging (assuming you didn't manually log in using `wandb login`).

To turn on wandb logging, you can either:

1. Manually login with `wandb login`: Learn more [here](https://docs.wandb.ai/ref/cli/wandb-login)
2. Automatically login programmatically by setting an environment variable: Get an API KEY at https://wandb.ai/authorize and set the environment variable as follows:

On Mac & Linux:

```
export WANDB_API_KEY=<YOUR WANDB API KEY>
```

On Windows:

```
set WANDB_API_KEY=<YOUR WANDB API KEY>
```
Moreover, if you couldn't access W&B and want to log metrics offline, you can set the environment variable as follows:

```
export WANDB_MODE=offline
```

## Inferencing for Stage 2
### 1. Data preparation
Similarly, convert transcripts into the unified IPA representation.
```
python src/x_voice/train/datasets/prepare_ipa_stage2_gen_data.py \
    --tokenizer ipa_v6 \
    --dataset_name XVoice_Dataset \
    --inp_dir path/to/train-set/of/x-voice/ \
    --check_exists
```
Command-line arguments are the same as [data preparation for stage 1 training](#1-data-preparation), but the `--dnsmos` threshold is not provided because this step already uses the dnsmos-filtered subset.
### 2. Prepare inference config
Then, create a `.yaml` file under `./src/x_voice/config`, for example `XVoice_Base_Infer.yaml`, refer to [this example](https://github.com/sunnyxrxrx/X-Voice/blob/main/src/x_voice/configs/XVoice_Base_Infer.yaml).

### 3. Run inference
```
accelerate launch src/x_voice/train/inference_gp.py --config-name XVoice_Base_Infer.yaml
```
The `.pt` and `.json` files generated by inference are stored by default in `./[dataset_name]_gp/`. The internal path follows those in `metadata_*.csv`, for example `bg/2009/xxx.pt` and `bg/2009/xxx.json`.
## Stage 2 Training
### 1. Data preparation
```
python src/x_voice/train/datasets/prepare_ipa_sft.py  \
    --tokenizer ipa_v6 \
    --dataset_name XVoice_Dataset \
    --inp_dir path/to/train-set/of/x-voice/ \
    --sft_gen_dir path/to/generated/pt/and/json/ 
    --check_exists
```
Here you need to explicitly provide `sft_gen_dir`, which is where the `.pt` and `.json` files generated by inference are stored.

### 2. Prepare training config
Create a `.yaml` file under `./src/x_voice/config`, for example `XVoice_Base_Stage2.yaml`, refer to [this example](https://github.com/sunnyxrxrx/X-Voice/blob/main/src/x_voice/configs/XVoice_Base_Stage2.yaml).

> [!NOTE]
>- `model.sft` must be set to `True`
>- `model.use_total_text` set to `False`
>- `ckpts.pretrained_path` pass the path to Stage 1 model weights

### 3. Run training
```
accelerate launch src/x_voice/train/train_sft.py  --config-name XVoice_Base_Stage2.yaml
```
