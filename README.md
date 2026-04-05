# Deep Multimodal Personality Prediction (Ouarka et al. 2024 Reimplementation)

This repository contains a full, training-ready PyTorch reimplementation of the multimodal personality prediction pipeline inspired by:

Ouarka et al. (2024), A Deep Multimodal Fusion Method for Personality Traits Prediction.

The implementation is adapted for modern libraries, CPU-only execution, and reproducible ablation experiments on a 400/80/80 ChaLearn First Impressions V2 subset.

## 1. What is implemented

Implemented model variants:

- scene
- face
- audio
- text
- early_fusion
- model_fusion
- self_attention
- cross_attention

Implemented objectives and metrics:

- Loss: Mean Squared Error (MSE)
- Main metric: Mean Absolute Error (MAE)
- Accuracy: (1 - MAE) * 100

## 2. Repository structure

- prepare_data.py: full preprocessing and cache generation pipeline
- train.py: training entrypoint for all ablation variants
- evaluate.py: standalone checkpoint evaluation
- verify_cache_usage.py: utility to verify cache comes from real videos
- src/config.py: constants and dataclass configs
- src/data.py: annotation parsing, transcript parsing, feature extraction, cache dataset
- src/models.py: unimodal and fusion architectures
- src/engine.py: train loop, validation/test loop, metrics, checkpoint saving
- requirements.txt: Python dependencies
- cache/: generated data tensors and metadata
- checkpoints/: best model checkpoints
- histories/: loss curves and final metrics JSON files

## 3. Environment and dependencies

Python environment: local virtual environment at .venv

Dependencies (requirements.txt):

- numpy
- opencv-python
- torch
- tqdm
- scikit-learn
- librosa
- soundfile

Install:

```bash
pip install -r requirements.txt
```

## 4. Dataset assumptions used in this run

Local directory layout:

- train_data/ -> 400 videos
- valid_data/ -> 80 videos
- test_data/ -> 80 videos

Annotation files:

- annotation_training.pkl is required
- annotation_validation.pkl and annotation_test.pkl are optional

In this project run, annotation_training.pkl was used as the annotation source for all splits because it already contained complete labels across train/valid/test video IDs.

Transcription files:

- transcription_training.pkl
- transcription_validation.pkl
- transcription_test.pkl

These were missing in this run, so the pipeline used empty transcript fallback.

## 5. Data preprocessing pipeline (prepare_data.py + src/data.py)

### 5.1 Annotation parsing

The code supports two schemas:

1) Trait dictionary schema (preferred):

- openness
- conscientiousness
- extraversion
- agreeableness
- neuroticism

Each trait key maps to a dict: video_name.mp4 -> float score.

2) Fallback schema:

- video_name.mp4 -> [o, c, e, a, n]

### 5.2 Transcript parsing

If transcript pickle files exist, a mapping video_name.mp4 -> text is loaded.
If not, empty string is used for each sample.

### 5.3 Video features

For each video:

- Uniformly sample 10 frames
- Resize scene frames to 96x96 RGB
- Detect face with OpenCV Haar cascade
- If face detection fails, use center square crop
- Resize face crop to 96x96 RGB

### 5.4 Audio features

Audio is decoded from each video using librosa at 16kHz mono and converted to log-mel features:

- n_mels = 128
- output temporal steps = 15
- if shorter, pad with repeated last frame
- if extraction fails, return zeros (15, 128)

### 5.5 Text features

Vocabulary is built from training transcripts:

- special tokens: <pad>, <unk>
- max vocab size: 12000
- sequence length: 50

In this run, transcripts were absent, so vocab size ended up very small (special tokens only).

### 5.6 Cached tensors

Saved to cache directory:

- train.npz
- valid.npz
- test.npz
- vocab.json
- meta.json

Tensor shapes in cache:

- scene: [N, 10, 96, 96, 3]
- face: [N, 10, 96, 96, 3]
- audio: [N, 15, 128]
- text: [N, 50]
- targets: [N, 5]

## 6. Model implementation details (src/models.py)

### 6.1 Visual branch (scene/face)

Dual-stream design inspired by VGG-like + ViT-like fusion:

- Stream A: CNN frame encoder -> LSTM(128) -> LSTM(64) -> FC(1024, 512)
- Stream B: Lightweight transformer frame encoder -> LSTM(128) -> LSTM(64) -> FC(1024, 512)
- Stream merge: average of stream features
- Output head: FC(256) + dropout + sigmoid(5)

Returns:

- per-modality prediction [B, 5]
- sequence embedding for attention fusion [B, T, 64]

### 6.2 Audio branch

- Conv1d(128->32, k=2) + dropout
- Conv1d(32->64, k=2) + dropout
- LSTM(128) -> LSTM(64)
- FC(256) + dropout + sigmoid(5)

Returns prediction and sequence embedding.

### 6.3 Text branch

- Embedding(vocab_size, 100)
- Two parallel Conv1d stacks:
	- path x: 100->16->8
	- path y: 100->32->16
- Flatten + dense per path -> concatenate -> FC -> sigmoid(5)
- Additional projection to sequence embedding for attention fusion

### 6.4 Fusion variants

1) early_fusion

- Average of scene, face, audio, text predictions

2) model_fusion

- Concatenate the 4 modality predictions (20 dims) -> FC(100) -> FC(5)

3) self_attention

- Self-attention per modality sequence
- Concatenate attended sequences
- Global average + FC head

4) cross_attention

- Build video sequence from scene+face
- Pairwise cross-attention between video, audio, text
- Concatenate 6 attended outputs
- Global average + FC head

## 7. Training and evaluation logic (src/engine.py + train.py + evaluate.py)

### 7.1 Training

- Optimizer: AdamW
- Default hyperparameters:
	- batch_size = 4
	- epochs = 10
	- lr = 1e-3
	- weight_decay = 1e-4
	- device = cpu
- Checkpoint policy: save best checkpoint by lowest validation MAE

Saved training outputs:

- checkpoints/<run_name>.pt
- histories/<run_name>.json
- histories/<run_name>_metrics.json

### 7.2 Evaluation

Evaluates on valid and test split from cached tensors.

Metrics include:

- loss
- mae
- accuracy
- per-trait mae and per-trait accuracy

## 8. Commands used in this project

### 8.1 Prepare cache

```bash
python prepare_data.py \
	--root-dir . \
	--cache-dir cache \
	--train-dir train_data \
	--valid-dir valid_data \
	--test-dir test_data \
	--annotation-train annotation_training.pkl \
	--annotation-valid annotation_training.pkl \
	--annotation-test annotation_training.pkl \
	--train-limit 400 \
	--valid-limit 80 \
	--test-limit 80 \
	--image-size 96 \
	--num-frames 10 \
	--audio-steps 15 \
	--audio-bins 128 \
	--text-seq-len 50
```

### 8.2 Train all fusion ablations

```bash
python train.py --model-type model_fusion --cache-dir cache --checkpoint-dir checkpoints --history-dir histories --batch-size 4 --epochs 10 --learning-rate 1e-3 --device cpu --num-workers 0
python train.py --model-type early_fusion --cache-dir cache --checkpoint-dir checkpoints --history-dir histories --batch-size 4 --epochs 10 --learning-rate 1e-3 --device cpu --num-workers 0
python train.py --model-type self_attention --cache-dir cache --checkpoint-dir checkpoints --history-dir histories --batch-size 4 --epochs 10 --learning-rate 1e-3 --device cpu --num-workers 0
python train.py --model-type cross_attention --cache-dir cache --checkpoint-dir checkpoints --history-dir histories --batch-size 4 --epochs 10 --learning-rate 1e-3 --device cpu --num-workers 0
```

### 8.3 Evaluate any checkpoint

```bash
python evaluate.py --cache-dir cache --checkpoint checkpoints/model_fusion_20260404_120118.pt --model-type model_fusion --batch-size 4 --device cpu
```

### 8.4 Verify cache is built from real videos

```bash
python verify_cache_usage.py
```

## 9. Experimental results (completed runs)

All results below are from the 400/80/80 subset and CPU settings above.

| Model | Valid Loss | Valid MAE | Valid Acc (%) | Test Loss | Test MAE | Test Acc (%) |
|---|---:|---:|---:|---:|---:|---:|
| early_fusion | 0.024774 | 0.124775 | 87.5225 | 0.016640 | 0.104642 | 89.5358 |
| cross_attention | 0.024817 | 0.124056 | 87.5944 | 0.017183 | 0.107092 | 89.2908 |
| model_fusion | 0.025272 | 0.127001 | 87.2999 | 0.018012 | 0.107855 | 89.2145 |
| self_attention | 0.025606 | 0.127630 | 87.2370 | 0.018852 | 0.109674 | 89.0326 |

Best test MAE in this run: early_fusion (0.104642).

Metric artifact files:

- histories/early_fusion_20260404_140916_metrics.json
- histories/cross_attention_20260404_184849_metrics.json
- histories/model_fusion_20260404_120118_metrics.json
- histories/self_attention_20260404_162425_metrics.json

## 10. Reproducibility notes and limitations

1) This run is a CPU-friendly subset setting, not the full-dataset full-compute paper setup.

2) Transcript files were not available during this run, so text modality used empty input fallback.

3) Because of (2), these scores should be treated as partial multimodal baselines.

4) To approach stronger paper-style reproduction, add transcription pickle files and rerun prepare_data.py + train.py.

## 11. Completion summary

The implementation is complete and functional end-to-end:

- data preparation
- multimodal model training
- ablation runs
- checkpointing
- metric export
- final model comparison

