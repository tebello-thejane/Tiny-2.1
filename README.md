# TINY — a teensy transformer to play with

A question-answering transformer with 366,528 parameters that reads and writes
one letter at a time. It is small enough to train on a laptop CPU, and built to
be poked at: chat with it, watch it think, switch parts of it off, retrain it,
grow it.

- 4 layers, 96 numbers wide, 4 attention heads per layer, 97 possible letters
- Trained on about 3,500 facts, each asked in many wordings. Outside those it
  makes things up confidently.
- 61.5% right on 780 questions it never saw during training
- A terminal chat with a live neural monitor and a "surgery" panel

TINY began as a DOS program in Turbo C; that version is in the git history
(commit `4934844`). Written by Fable 5.2 and trained by Opus 5.5.

Everything runs with [uv](https://docs.astral.sh/uv/). Each script declares
what it needs, and uv fetches Python, NumPy, CPU-only PyTorch and Textual on
first use. Nothing to install by hand.

## Play

```
tui/tiny.py                    # chat; F1 help, F4 creativity, Up/Down history, Esc quits
tui/tiny.py 0                  # creativity 0: always the surest answer
tui/tiny.py --off 4.3          # start with layer 4, head 3 switched off
tui/tiny.py --model my.pt      # chat with another checkpoint
tui/tiny.py --demo             # TINY asks itself sample questions
tui/tiny.py < questions.txt    # plain text when piped
```

The right-hand panel shows each layer's activity, how sure TINY is of every
letter it writes, and the surgery panel: one row per layer, one dot per head.
Click a dot or a layer name to switch it off and on, or type `off 2.3`,
`off L4`, `on 2.3` or `on all` in the question box. F5 switches everything back
on. Switching parts off never changes the checkpoint file.

## Measure, retrain, grow

```
cd training
./gen2.py                                    # see sample questions from the facts
./train2.py info tiny2.pt                    # sizes and where the parameters are
./train2.py eval tiny2.pt                    # score on 780 held-back questions
./train2.py eval tiny2.pt --off 4.3          # 49.0%: what losing one head costs
./train2.py train 60 more.pt --resume tiny2.pt --lr 5e-4 --decay 0.3 --eval-every 20
./train2.py train 30 wide.pt --grow tiny2.pt --config models/wide.toml
./train2.py train 60 gelu.pt --config models/gelu.toml
./train2.py --help                           # every command and option
```

- `train MINUTES` trains for that much wall-clock time, then saves and scores
  the model.
- `--resume` continues training a model as it is.
- `--grow` puts an existing model inside a bigger one (more width `D`, heads
  `H`, middle width `HID` or layers `L`). The new parts start silent, so the
  grown model gives the same answers as the old one until training gives the
  new parts something to do.
- Without either, `train` starts a new model from random numbers.

Add your own facts to `training/facts/` (format in `training/TRAINING.TXT`).

## Design your own model

A model's design is a small text file in `training/models/`. Start from
`models/tiny2.toml`, which describes the shipped model with a comment on every
setting:

```toml
D = 96                  # width: numbers per letter (must divide by H)
L = 4                   # layers
H = 4                   # attention heads per layer
HID = 256               # width of each layer's middle part
activation = "relu"     # "relu", "gelu" or "silu"
norm = "rms"            # "rms" or "layer"
tied_output = true      # reuse the letter embedding to score the next letter
positions = "learned"   # "learned" or "none"
attention = "all"       # "all", or the layers that keep it, e.g. [1, 3]
```

Copy it, change what you like (anything you leave out keeps the value above),
and train it with `--config`. The design is saved inside the checkpoint, so
`eval`, `info` and the chat rebuild the right model automatically.
`./train2.py info MODEL.pt` prints any model's design in this format. The other
files in `models/` are ready-made experiments: `gelu`, `small`, `wide` (grow
tiny2 into it), `no-attention` and `blind-order`.

For changes the settings don't cover, edit the `Norm`, `Block` and `Model`
classes near the top of `training/train2.py`. The chat program runs that same
code.

## Licence

Code: MIT (`LICENSE`). Three data files derived from public datasets keep
their own licences (ODbL and CC BY-SA); see `DATA_LICENSES.md`.
