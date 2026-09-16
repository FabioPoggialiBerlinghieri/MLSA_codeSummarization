# Python Code Summarization

A Machine Learning project that analyzes small Python code snippets and generates natural-language descriptions in English.

## Features

* **Train** — train the model on the provided dataset.
* **Evaluate** — evaluate model performance using **BLEU** and **ROUGE**.
* **Summarize** — generate an English description from a Python code snippet.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the pre-trained model weights from [Google Drive](https://drive.google.com/drive/folders/1E1HfQHBSiO5K4AIzRP2F-0FgrUa2qnPg?usp=drive_link) and place them in:

```text
/models
```

## Usage

Run the following commands from the `/src` directory.

### Summarize

Generate a description for a Python snippet:

```bash
python3 summarize.py \
    --input 'python snippet' \
    --checkpoint '../models/[model_weights.pt]'
```

Example snippets to use with the **Summarize** functionality are available in the `snippets_test/` directory.

### Evaluate

Evaluate the model on the validation set:

```bash
python3 evaluate.py \
    --checkpoint '../models/[model_weights.pt]' \
    --split val
```

The batch size can be changed using `--batch` (default: `128`).

## Training

The tokenized dataset and vocabularies are available on [Google Drive](https://drive.google.com/drive/folders/1E1HfQHBSiO5K4AIzRP2F-0FgrUa2qnPg?usp=drive_link) and should be placed in:

```text
/data
```

Start training with:

```bash
python3 train.py --config config/config.yaml
```

The main model parameters are defined in `config/config.yaml`.

If the dataset has not been tokenized yet, set:

```yaml
first_time: true
```

The dataset will then be processed automatically.

### Command-line parameters

| Parameter        | Default | Description                       |
| ---------------- | ------: | --------------------------------- |
| `--max-code-len` |     512 | Maximum input code length         |
| `--max-sum-len`  |     128 | Maximum output description length |
| `--save-every`   |     500 | Model checkpoint saving frequency |
