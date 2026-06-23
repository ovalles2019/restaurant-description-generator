# Restaurant Description Generator

A sequence-to-sequence (GRU encoder-decoder) web app that generates natural-language
restaurant descriptions from structured **meaning representations**. Built from a
CS 6320 (NLP) Homework 5 implementation and wrapped in a simple Gradio UI.

## Example

**Input (meaning representation):**

```
name[The Eagle], eatType[coffee shop], food[French], priceRange[moderate],
customer rating[3 out of 5], area[riverside], familyFriendly[yes], near[Burger King]
```

**Output (generated description):**

> The three star coffee shop, The Eagle, gives families a mid-priced dining
> experience near Burger King.

(Exact output depends on how many epochs the model was trained for.)

## How it works

The model is a single-layer GRU **encoder-decoder** trained on the
[E2E Challenge dataset](http://www.macs.hw.ac.uk/InteractionLab/E2E/). The encoder
reads the meaning-representation tokens; the decoder generates the description one
word at a time using greedy decoding. The architecture lives in `model.py`; the
Gradio interface lives in `app.py`.

## Project structure

```
restaurant-description-generator/
|-- app.py             # Gradio UI + inference glue
|-- model.py           # Encoder/Decoder, vocab building, generate()
|-- requirements.txt
|-- .gitignore
|-- README.md
|-- hw5.encoder        # trained encoder weights  (ADD THIS YOURSELF)
|-- hw5.decoder        # trained decoder weights  (ADD THIS YOURSELF)
`-- train.txt          # E2E training data        (ADD THIS YOURSELF)
```

## Files you need to add yourself

Three files are **not** in this repo and must be placed in the project folder before
running:

- **`hw5.encoder`** and **`hw5.decoder`** - your trained model parameters
  (saved from the notebook via `torch.save(model.state_dict(), ...)`).
- **`train.txt`** - the E2E training file. It is required because the vocabularies
  are rebuilt from it at startup, exactly as in the notebook, so the word indices
  line up with the trained weights.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Gradio prints a local URL (usually `http://127.0.0.1:7860`). Open it in your browser,
fill in the restaurant fields, and click **Generate Description**.

## Notes

- The app runs on **CPU**. GPU-trained weights load fine thanks to
  `map_location='cpu'` in `model.py`.
- Output quality scales with training. A single training epoch produces rough,
  sometimes repetitive descriptions; more epochs improve fluency. Just drop in
  updated `hw5.encoder` / `hw5.decoder` files - no code changes needed.
- Price values use the £ symbol to match the E2E dataset
  (e.g. `less than £20`).

## Acknowledgements

Dataset: E2E NLG Challenge. Model architecture from CS 6320 Natural Language
Processing coursework.
# restaurant-description-generator
A seq2seq (GRU encoder-decoder) Gradio app that generates restaurant descriptions from structured meaning representations. Built from CS 6320 NLP HW5.
