# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

TINY: a 366,528-parameter character-level transformer that answers questions, kept small so it can be trained, grown and taken apart on a CPU. It started as a Turbo C DOS program; that version (and the 8-bit `TINY.BIN` format) was removed and lives only in git history before the current work. Do not reintroduce a second implementation of the network.

- `training/train2.py`: the single definition of the network (`Cfg`, `Norm`, `Block`, `Model`), plus training, growing (`grow`), evaluation, lobotomy switches (`switch_off`/`switch_on`/`switched_off`) and `info`.
- `training/models/*.toml`: model designs for `train --config`; `tiny2.toml` documents every setting and matches `tiny2.pt`.
- `training/gen2.py`: builds the training data from `training/facts/` at import time.
- `tui/tiny.py`: Textual chat UI. It imports `train2` from `../training` and runs `train2.Model` on a `.pt` checkpoint (default `training/tiny2.pt`), so any change to the model is automatically what the UI runs.

`README.md` is the user-facing guide; `training/TRAINING.TXT` has the fact-file formats and training history.

## Commands

Every `.py` is a PEP 723 uv script (`#!/usr/bin/env -S uv run --script`), executable, run as `./script.py`. `train2.py` and `tui/tiny.py` pin torch to the CPU wheel index (`download.pytorch.org/whl/cpu`) through `[tool.uv.sources]`; keep that block on any script that imports torch, or uv pulls the multi-gigabyte CUDA build. `torch.set_num_threads(2)` is set in `train2.py`.

```bash
cd training
./train2.py info tiny2.pt                        # sizes and parameter breakdown
./train2.py eval tiny2.pt [N_SHOWN] [--off 4.3]  # 780 held-out questions; tiny2.pt scores 61.5%
./train2.py train MINUTES OUT.pt --resume M.pt   # continue as-is
./train2.py train MINUTES OUT.pt --grow M.pt --config models/wide.toml
./train2.py train MINUTES OUT.pt --config models/gelu.toml [--D 64 ...]   # fresh; flags override the file
./train2.py --help                               # all commands, options and design settings
./gen2.py                                        # fact counts, generator mix, sample per generator
../tui/tiny.py [0-9] [--model M.pt] [--off SPEC] [--demo] [--plain] [--delay S]
```

`train MINUTES` is wall-clock, not epochs. Without `--decay` the schedule is cosine; `--decay F` holds the LR then decays linearly over the last fraction F. `--resume` refuses size options; use `--grow` to change sizes.

There is no test suite or linter. Regression checks that worked when this layout was built:

- `./train2.py eval tiny2.pt` must stay at 61.5% after any refactor that is meant to preserve behaviour (the switches multiply by 1.0 when on, so they must not change it).
- A grown model must score exactly what its source scores before training: `./train2.py train 0 g.pt --grow tiny2.pt --D 120 --H 5` gives 61.5%, category for category. Compare raw logits too, not only eval scores (an untrained model scores 0% either way): growing H/HID/L gave a max difference of 0.0; growing D gave about 4e-3 on logits of size 36, from the unscaled norm eps.
- Each design in `training/models/*.toml` should build, train for a few seconds (`train 0.1 x.pt --config models/NAME.toml`), save, and reload in `eval` and `tui/tiny.py --plain`.
- `tui/tiny.py 0` in plain mode (stdin piped) prints `>   answer` lines that equal `train2.answer_many` greedy answers once questions are cleaned the same way (the UI lower-cases ALL-CAPS questions, `answer_many` does not). 1,000 `gen2.sample` questions matched exactly.
- The UI is tested headless with Textual's `App.run_test()` pilot: build the app with `ui_main(model, creativity, demo, delay)` and drive it with key presses and clicks on the `.pill` widgets.

## Things that are easy to break

- **Vocabulary**: 97 tokens. 0 = pad, 1 = newline, 2..96 = ASCII 32..126 (`token = ord(c) - 30`, `encode`/`decode` in `train2.py`). Changing this invalidates every checkpoint.
- **Sequence layout**: `question \n answer \n`, at most `CTX` = 128 letters. Loss is masked to the answer part only. `gen2.py` caps questions at 80 letters and question+answer at `MAXLEN` = 126; the UI caps typed questions at `MAXQ` = 72.
- **Design (`Cfg`)** holds sizes (`D`, `L`, `H`, `HID`) and wiring switches (`activation` relu/gelu/silu, `norm` rms/layer, `tied_output`, `positions` learned/none, `attention` "all" or a 1-based layer list). `Cfg.__init__` validates every field with a readable `ValueError`; `read_config` loads a TOML file and rejects unknown keys; `Cfg.toml()` writes one back (`info` prints it). Precedence in `train`: `--grow` model's design, then `--config` file, then size flags.
- **Checkpoints** (`save_model`/`load_model`) hold `vars(cfg)` and the `state_dict`. New `Cfg` fields must have defaults equal to the old behaviour, so checkpoints saved before the field existed still load (`tiny2.pt` has only the four sizes). Wiring that `Cfg` does not describe is not saved: editing `Block`/`Model` code can make old checkpoints fail to load or answer nonsense. Parameter names (`emb`, `pos`, `blocks.N.{n1,q,k,v,o,n2,f1,f2}`, `nf`, `out`) are what the files key on; a layer without attention has no `n1`/`q`/`k`/`v`/`o`, and `pos`/`out` exist only when the design asks for them.
- **Default architecture** (what `tiny2.pt` uses): pre-norm blocks, RMS norm with gain only (eps 1e-5), no biases, ReLU MLP, learned positional embeddings, output logits tied to the letter embedding (`nf(x) @ emb.weight.t()`, so the output adds no parameters). "layer" norm is the same norm after subtracting the mean, still gain-only.
- **Lobotomy switches** live on each `Block`: `on` (skip the whole block) and the `head_on` buffer (multiplies each head's attention output). `head_on` is registered with `persistent=False` so it is never saved and checkpoints always load fully on. `record`/`act` store the mean ReLU activity per position for the UI monitor; leave `record` off during training.
- **`grow()` refuses** to change wiring (activation, norm, tied_output, positions, attention of existing layers) and refuses to widen D under layer norm, because neither can preserve the function.
- **`grow()` is function-preserving** (exact for H/HID/L, within the norm eps for D), and that depends on: new embedding columns zero; every write into the residual stream from new parts (`o` columns for new heads or new per-head dims, `o`/`f2` rows for new width, `f2` columns for new middle units, all of `o` and `f2` in new layers) zero; norm gains scaled by sqrt(oldD/newD); queries scaled by sqrt(newHD/oldHD); and `k` rows for new per-head dims of old heads zero. New `q`/`k`/`v`/`f1` weights stay random so training can break symmetry; zeroing them as well leaves new heads stuck at zero gradient.
- **Spec strings** for switches: `2.3` (layer 2 head 3), `L4` or `4` (whole layer), comma-separated, counted from 1. Parsed by `train2.switch`; used by `eval --off`, `tiny.py --off` and the UI's `off`/`on` commands.

## Training data pipeline

`gen2.py` builds everything at import time from `training/facts/` and exposes `sample(rng, test=False, kind=None)`. `train2.py` imports it and draws an endless stream; there is no stored dataset.

- `facts/*.txt`: `question | question | ... => answer | answer`. When a line has 3 or more questions, the last one is held back for testing.
- `facts/*.tsv`: `relation<TAB>subject<TAB>answer`. Phrasing templates per relation live in `REL_T`; the last template in each list is test-only. `geo.tsv` language/currency rows are skipped in favour of `geo2.tsv`; `borders` rows feed `g_borders` only.
- `GENS` lists each generator with its share of the training mix. Every generator honours the `test` flag by holding back phrasings or items (`pick_q`, `split(frac=0.1, seed=7)`), so `eval` measures generalisation, not recall. A new generator must do the same and be added to `GENS`; `eval_set()` picks it up automatically (60 questions per generator).
- `style()` adds typos, CAPS, "hey tiny," prefixes and similar noise, except for the `letters`, `yesno` and `numbers2` generators.
- `raw/` holds hand-written word lists already folded into `facts/words2.tsv`; they are not read at training time.

## Licensing constraint on data files

`training/facts/geo2.tsv`, `geo3.tsv` (ODbL 1.0) and `elements.tsv` (CC BY-SA 3.0) are derived from public datasets and are not MIT. They are rebuilt by `training/geo2.py` and `training/extra_facts.py` from source datasets placed in `training/mledoze/` and `training/ptable/` (not committed). Keep `DATA_LICENSES.md` accurate when adding derived data.
