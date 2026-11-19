#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy", "torch"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Train, grow, inspect and lobotomise TINY.

  ./train2.py info  MODEL.pt                     sizes and where the parameters are
  ./train2.py eval  MODEL.pt [N] [--off SPEC]    score on held-back questions (show N misses)
  ./train2.py train MINUTES OUT.pt [options]     train for MINUTES of wall-clock time
  ./train2.py bench                              speed test of a few model sizes

train options:
  --config FILE.toml   the design of a new or grown model: sizes and wiring
                       (see models/*.toml; `info` prints any model's design)
  --D n --L n --H n --HID n   sizes, overriding the config file
                       (width, layers, heads, middle width; default 96 4 4 256)
  --resume M.pt        continue training M.pt exactly as it is
  --grow M.pt          start from M.pt inside a bigger model (see grow());
                       settings not given keep M.pt's values
  --lr X               peak learning rate (default 0.002)
  --decay F            hold the rate, then fade it out over the last F of the time
                       (without it: a cosine curve)
  --eval-every MIN     save and score the model every MIN minutes while training

design settings (config file keys; all optional, defaults in brackets):
  D, L, H, HID         sizes [96, 4, 4, 256]; D must divide by H
  activation           "relu", "gelu" or "silu"  [relu]
  norm                 "rms" or "layer"  [rms]
  tied_output          true: score letters with the letter embedding;
                       false: a separate output table  [true]
  positions            "learned" or "none" (no idea of letter order)  [learned]
  attention            "all", or a list of layers that keep it, e.g. [1, 3]  [all]

--off SPEC switches parts off, e.g. "2.3" (layer 2, head 3), "L4" (all of layer 4),
"2.3,3.1,L4". Layers and heads count from 1.

The model reads and writes one letter at a time (ASCII 32..126 plus newline)."""
import sys, os, time, math, random, re, collections
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen2

VOCAB, CTX = 97, 128
torch.set_num_threads(2)


def encode(s):
    return [1 if c == '\n' else ord(c) - 30 for c in s]


def decode(ids):
    return ''.join('\n' if i == 1 else chr(i + 30) for i in ids if i > 0)


ACTIVATIONS = {'relu': F.relu, 'gelu': F.gelu, 'silu': F.silu}


class Cfg:
    """The model's design: sizes and wiring. Saved inside every checkpoint, and
    readable from a TOML file (see models/*.toml and read_config)."""
    CHOICES = dict(activation=tuple(ACTIVATIONS), norm=('rms', 'layer'), positions=('learned', 'none'))
    KEYS = ('D', 'L', 'H', 'HID', 'activation', 'norm', 'tied_output', 'positions', 'attention')

    def __init__(self, D=96, L=4, H=4, HID=256, activation='relu', norm='rms', tied_output=True,
                 positions='learned', attention='all'):
        self.D, self.L, self.H, self.HID = D, L, H, HID
        self.activation, self.norm, self.tied_output = activation, norm, tied_output
        self.positions, self.attention = positions, attention
        for k in ('D', 'L', 'H', 'HID'):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise ValueError('%s must be a whole number of at least 1, not %r' % (k, getattr(self, k)))
        if D % H:
            raise ValueError('D (%d) must divide evenly by H (%d): each head gets D/H numbers' % (D, H))
        for k, ok in self.CHOICES.items():
            if getattr(self, k) not in ok:
                raise ValueError('%s must be one of %s, not %r' % (k, ', '.join(ok), getattr(self, k)))
        if not isinstance(tied_output, bool):
            raise ValueError('tied_output must be true or false, not %r' % (tied_output,))
        if attention != 'all' and not (isinstance(attention, list)
                                       and all(isinstance(i, int) and 1 <= i <= L for i in attention)):
            raise ValueError('attention must be "all" or a list of layer numbers from 1 to %d, not %r' % (L, attention))

    def attends(self, i):
        """Does layer i (counted from 0) have attention?"""
        return self.attention == 'all' or i + 1 in self.attention

    def __repr__(self):
        s = 'D=%d L=%d H=%d HID=%d' % (self.D, self.L, self.H, self.HID)
        extra = [('%s=%s' % (k, getattr(self, k))) for k in self.KEYS[4:] if getattr(self, k) != Cfg().__dict__[k]]
        return s + (' ' + ' '.join(extra) if extra else '')

    def toml(self):
        val = lambda v: ('true' if v else 'false') if isinstance(v, bool) else ('"%s"' % v if isinstance(v, str) else str(v))
        return '\n'.join('%s = %s' % (k, val(getattr(self, k))) for k in self.KEYS) + '\n'


def read_config(path):
    """A TOML file of Cfg settings. Missing settings keep their defaults."""
    import tomllib
    with open(path, 'rb') as f:
        d = tomllib.load(f)
    unknown = sorted(set(d) - set(Cfg.KEYS))
    if unknown:
        raise ValueError('%s: unknown setting %s (known: %s)' % (path, ', '.join(unknown), ', '.join(Cfg.KEYS)))
    return d


class Norm(nn.Module):
    """RMS norm (kind "rms"), or the same after subtracting the mean (kind "layer").
    Gain only, no bias."""

    def __init__(self, d, kind='rms'):
        super().__init__()
        self.kind = kind
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        if self.kind == 'layer':
            x = x - x.mean(-1, keepdim=True)
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype) * self.g


class Block(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.c = c
        self.attn = c.attends(i)                         # False: this layer has only the middle part
        self.fn = ACTIVATIONS[c.activation]
        if self.attn:
            self.n1 = Norm(c.D, c.norm)
        self.n2 = Norm(c.D, c.norm)
        if self.attn:
            self.q = nn.Linear(c.D, c.D, bias=False)
            self.k = nn.Linear(c.D, c.D, bias=False)
            self.v = nn.Linear(c.D, c.D, bias=False)
            self.o = nn.Linear(c.D, c.D, bias=False)
        self.f1 = nn.Linear(c.D, c.HID, bias=False)
        self.f2 = nn.Linear(c.HID, c.D, bias=False)
        # Lobotomy switches. Not saved with the model (persistent=False), so a
        # checkpoint always loads with everything on.
        self.on = True                                    # False: the layer passes x through untouched
        self.register_buffer('head_on', torch.ones(c.H), persistent=False)   # 0 silences a head
        self.record = False                               # True: keep self.act for the TUI monitor
        self.act = None

    def forward(self, x):
        if not self.on:
            return x
        if self.attn:
            B, T, D = x.shape
            H = self.c.H
            h = self.n1(x)
            sp = lambda t: t.view(B, T, H, D // H).transpose(1, 2)
            a = F.scaled_dot_product_attention(sp(self.q(h)), sp(self.k(h)), sp(self.v(h)), is_causal=True)
            a = a * self.head_on.view(1, H, 1, 1)
            x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        m = self.fn(self.f1(self.n2(x)))
        if self.record:
            self.act = m.detach().mean(-1)                # (B, T): how busy the middle layer is
        return x + self.f2(m)


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(VOCAB, c.D)
        self.pos = nn.Embedding(CTX, c.D) if c.positions == 'learned' else None
        self.blocks = nn.ModuleList([Block(c, i) for i in range(c.L)])
        self.nf = Norm(c.D, c.norm)
        self.out = None if c.tied_output else nn.Linear(c.D, VOCAB, bias=False)
        for e in (self.emb, self.pos, self.out):
            if e is not None:
                nn.init.normal_(e.weight, std=0.05)
        for b in self.blocks:
            for m in ((b.q, b.k, b.v, b.o) if b.attn else ()) + (b.f1, b.f2):
                nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))

    def forward(self, idx):
        x = self.emb(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        x = self.nf(x)
        return x @ self.emb.weight.t() if self.out is None else self.out(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- lobotomy
def switch_off(model, spec):
    """Switch parts off: "2.3" = layer 2 head 3, "L4" or "4" = all of layer 4,
    comma-separated. Counts from 1. Returns the parts it switched off."""
    return switch(model, spec, False)


def switch_on(model, spec):
    """Undo switch_off for the parts named in spec (same format)."""
    return switch(model, spec, True)


def switched_off(model):
    """The parts currently off, in switch_off's format."""
    out = []
    for l, b in enumerate(model.blocks):
        if not b.on:
            out.append('L%d' % (l + 1))
        if b.attn:
            out += ['%d.%d' % (l + 1, h + 1) for h in range(model.c.H) if b.head_on[h] == 0]
    return out


def switch(model, spec, on):
    done = []
    for part in filter(None, (p.strip() for p in (spec or '').split(','))):
        m = re.fullmatch(r'[lL]?(\d+)(?:\.(\d+))?', part)
        if not m:
            raise ValueError('cannot read %r: use 2.3 for a head or L4 for a layer' % part)
        l, h = int(m.group(1)) - 1, m.group(2)
        if not 0 <= l < model.c.L:
            raise ValueError('there is no layer %d (this model has %d)' % (l + 1, model.c.L))
        if h is None:
            model.blocks[l].on = on
            done.append('L%d' % (l + 1))
        else:
            h = int(h) - 1
            if not model.blocks[l].attn:
                raise ValueError('layer %d has no attention, so no heads to switch' % (l + 1))
            if not 0 <= h < model.c.H:
                raise ValueError('there is no head %d (this model has %d per layer)' % (h + 1, model.c.H))
            model.blocks[l].head_on[h] = 1.0 if on else 0.0
            done.append('%d.%d' % (l + 1, h + 1))
    return done


def switch_all_on(model):
    for b in model.blocks:
        b.on = True
        b.head_on.fill_(1.0)


# ---------------------------------------------------------------- growing
def grow(old, c):
    """A bigger model that starts out computing what `old` computes, so training
    continues from the old answers instead of from scratch. Growing H, HID or L
    is exact; growing D is exact except for the norm's 1e-5 safety constant,
    which shifts the scores by about 1e-4 of their size (same answers in tests).

    Width D, heads H, middle width HID and layers L may each grow; none may
    shrink, and each head (D/H numbers) may not get narrower. New parts start
    "silent": their output weights are zero, so they add nothing until training
    gives them a job. Two rescalings keep the old numbers exact: RMS norm now
    averages over more numbers (gains x sqrt(oldD/newD)), and attention now
    divides by the square root of a wider head (queries x sqrt(newHD/oldHD)).

    The wiring (activation, norm, output, positions, and which old layers have
    attention) must stay the same: changing it would change the answers. Layer
    norm cannot widen exactly (its mean would include the new zeros), so a
    "layer"-norm model may grow in every way except D."""
    o = old.c
    ohd, nhd = o.D // o.H, c.D // c.H
    if c.D % c.H or min(c.D - o.D, c.H - o.H, c.HID - o.HID, c.L - o.L, nhd - ohd) < 0:
        raise SystemExit('grow: %s -> %s is not a growth (sizes may only grow, D must divide by H, '
                         'heads may not get narrower)' % (o, c))
    for k in ('activation', 'norm', 'tied_output', 'positions'):
        if getattr(o, k) != getattr(c, k):
            raise SystemExit('grow: %s must stay %r (asked for %r); train a new model to change it'
                             % (k, getattr(o, k), getattr(c, k)))
    if any(o.attends(i) != c.attends(i) for i in range(o.L)):
        raise SystemExit('grow: the old layers must keep their attention as it is')
    if o.norm == 'layer' and c.D != o.D:
        raise SystemExit('grow: a model with layer norm cannot get wider (D); grow H, HID or L instead')
    m = Model(c)
    od = o.D
    old_rows = torch.tensor([h * nhd + i for h in range(o.H) for i in range(ohd)])       # old heads, new places
    new_dims = torch.tensor([h * nhd + i for h in range(o.H) for i in range(ohd, nhd)], dtype=torch.long)
    gscale, qscale = math.sqrt(od / c.D), math.sqrt(nhd / ohd)
    with torch.no_grad():
        for t, s in ((m.emb, old.emb), (m.pos, old.pos)):
            if t is not None:
                t.weight.zero_()
                t.weight[:, :od] = s.weight               # new width starts at zero
        if m.out is not None:
            m.out.weight[:, :od] = old.out.weight
        m.nf.g.fill_(1.0)
        m.nf.g[:od] = old.nf.g * gscale
        for i, blk in enumerate(m.blocks):
            if blk.attn:
                blk.o.weight.zero_()
            blk.f2.weight.zero_()
            if i >= o.L:
                continue                                  # a new layer: adds nothing yet
            ob = old.blocks[i]
            for n in ('n1', 'n2') if blk.attn else ('n2',):
                g = getattr(blk, n).g
                g.fill_(1.0)
                g[:od] = getattr(ob, n).g * gscale
            if blk.attn:
                blk.q.weight[old_rows, :od] = ob.q.weight * qscale
                blk.k.weight[old_rows, :od] = ob.k.weight
                blk.k.weight[new_dims] = 0.0              # widened old heads: scores unchanged
                blk.v.weight[old_rows, :od] = ob.v.weight
                blk.o.weight[:od, old_rows] = ob.o.weight  # new heads and new dims: silent
            blk.f1.weight[:o.HID, :od] = ob.f1.weight
            blk.f2.weight[:od, :o.HID] = ob.f2.weight     # new middle units: silent
    return m


# ---------------------------------------------------------------- data
_POOL = []


def batch(rng, n):
    """Examples of similar length go together, so little time is spent on padding."""
    if not _POOL:
        ex = []
        for _ in range(n * 32):
            q, a, _ = gen2.sample(rng)
            ids = encode(q + '\n' + a + '\n')[:CTX]
            ex.append((ids, [0] * (len(q) + 1) + [1] * (len(ids) - len(q) - 1)))
        ex.sort(key=lambda e: len(e[0]))
        chunks = [ex[i:i + n] for i in range(0, len(ex), n)]
        rng.shuffle(chunks)
        _POOL.extend(chunks)
    seqs, masks = zip(*_POOL.pop())
    T = max(len(s) for s in seqs)
    x = torch.zeros(n, T, dtype=torch.long)
    m = torch.zeros(n, T)
    for i, (s, k) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(k)] = torch.tensor(k, dtype=torch.float)
    return x, m


def loss_fn(model, x, m):
    logits = model(x[:, :-1])
    l = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1), reduction='none')
    w = m[:, 1:].reshape(-1)
    return (l * w).sum() / w.sum()


# ---------------------------------------------------------------- evaluation
STOP = set('a an the it is are was of in on and to its it\'s they there people speak use money capital city '
           'called comes after before has have says say lives eats one two answer number letter letters starts '
           'ends with month day the'.split())


def words_of(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def eval_set(n_per_kind=60, seed=123):
    rng = random.Random(seed)
    out = []
    for f in [f for f in gen2.TXT if f['test_q']]:
        out.append((f['test_q'][0], f['answers'], 'facts'))
    rng.shuffle(out)
    out = out[:120]
    for name, _, _ in gen2.GENS:
        if name == 'facts':
            continue
        seen = set()
        while sum(1 for o in out if o[2] == name) < n_per_kind:
            q, a, _ = gen2.sample(rng, True, name)
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append((q, [a], name))
    return out


UNKNOWN_START = ('i do not', 'i have never', 'sorry', 'hmm')


def correct(q, gen, refs, kind=None):
    if kind == 'unknown' and refs[0].lower().startswith(UNKNOWN_START):
        if not gen.lower().startswith(UNKNOWN_START):
            return False
        names = [w for w in words_of(refs[0]) if w in words_of(q)]
        return all(w in words_of(gen) for w in names)
    g = set(words_of(gen))
    qw = set(words_of(q))
    for r in refs:
        rw = words_of(r)
        if rw and rw[0] in ('yes', 'no') and (not words_of(gen) or words_of(gen)[0] != rw[0]):
            continue
        key = [w for w in rw if w not in STOP and w not in qw]
        if not key:
            key = [w for w in rw if w not in STOP] or rw
        if all(w in g for w in key):
            return True
    return False


@torch.no_grad()
def answer_many(model, qs):
    """Greedy answers, batched by question length."""
    model.eval()
    res = {}
    bylen = collections.defaultdict(list)
    for q in qs:
        bylen[len(q)].append(q)
    for n, group in bylen.items():
        x = torch.tensor([encode(q + '\n') for q in group])
        done = [False] * len(group)
        outs = [[] for _ in group]
        while x.shape[1] < CTX and not all(done):
            nxt = model(x)[:, -1].argmax(-1)
            for i, t in enumerate(nxt.tolist()):
                if not done[i]:
                    if t <= 1:
                        done[i] = True
                    else:
                        outs[i].append(t)
            x = torch.cat([x, nxt[:, None]], 1)
        for q, o in zip(group, outs):
            res[q] = decode(o)
    model.train()
    return res


def evaluate(model, show=0):
    es = eval_set()
    ans = answer_many(model, [e[0] for e in es])
    by = collections.defaultdict(lambda: [0, 0])
    bad = []
    for q, refs, kind in es:
        ok = correct(q, ans[q], refs, kind)
        by[kind][0] += ok
        by[kind][1] += 1
        if not ok:
            bad.append((kind, q, ans[q], refs[0]))
    tot = sum(v[0] for v in by.values()) / sum(v[1] for v in by.values())
    line = '  '.join('%s %d%%' % (k, 100 * v[0] / v[1]) for k, v in by.items())
    if show:
        random.Random(0).shuffle(bad)
        for b in bad[:show]:
            print('   x [%s] %s -> %s   (want: %s)' % b)
    return tot, line


# ---------------------------------------------------------------- saving
def info(m):
    c = m.c
    na = sum(b.attn for b in m.blocks)
    count = lambda pat: sum(p.numel() for n, p in m.named_parameters() if re.search(pat, n))
    rows = [('letter embedding', '%d letters x %d' % (VOCAB, c.D), count(r'^emb\.')),
            ('position embedding', ('%d positions x %d' % (CTX, c.D)) if m.pos is not None else 'none', count(r'^pos\.')),
            ('attention', '%d layer(s) x 4 x %d x %d' % (na, c.D, c.D), count(r'\.(q|k|v|o)\.')),
            ('middle layers', '%d x 2 x %d x %d (%s)' % (c.L, c.D, c.HID, c.activation), count(r'\.f[12]\.')),
            ('norm gains', '%s norm' % c.norm, count(r'(n1|n2|nf)\.g')),
            ('output', 'reuses the letter embedding' if m.out is None else '%d x %d' % (VOCAB, c.D), count(r'^out\.'))]
    total = n_params(m)
    print('model %s' % c)
    print('  %d layers, %d wide, %d heads of %d, middle %d; attention in layers %s'
          % (c.L, c.D, c.H, c.D // c.H, c.HID, ', '.join(str(i + 1) for i in range(c.L) if c.attends(i)) or 'none'))
    for name, how, n in rows:
        print('  %-20s %-34s %9s  %5.1f%%' % (name, how, format(n, ','), 100 * n / total))
    print('  %-20s %-34s %9s' % ('total', '', format(total, ',')))
    print('\nas a config file (save as models/NAME.toml, then train --config models/NAME.toml):\n')
    print(c.toml(), end='')


def load_model(path):
    if not path.endswith('.pt'):
        raise SystemExit('load_model: %s is not a .pt checkpoint' % path)
    ck = torch.load(path, map_location='cpu')
    m = Model(Cfg(**ck['cfg']))
    m.load_state_dict(ck['sd'])
    return m


def save_model(m, path):
    torch.save(dict(cfg=vars(m.c), sd=m.state_dict()), path)


# ---------------------------------------------------------------- main
def train(minutes, out, cfg=None, grow_from=None, resume=None, peak=2e-3, bs=64, log_every=200,
          decay_frac=None, eval_every=0):
    """New model (cfg), grown model (grow_from + cfg) or continued model (resume).
    decay_frac=None: cosine schedule. decay_frac=f: constant LR, then a linear
    decay to zero over the last fraction f of the time (lets us stop and polish)."""
    torch.manual_seed(0)
    rng = random.Random(int(time.time()))
    if resume:
        model = load_model(resume)
    elif grow_from:
        old = load_model(grow_from)
        model = grow(old, cfg or old.c)
        print('grown from', grow_from, old.c, flush=True)
    else:
        model = Model(cfg or Cfg())
    cfg = model.c
    print('model', cfg, 'parameters', n_params(model), flush=True)
    decay = [p for p in model.parameters() if p.dim() == 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.01), dict(params=other, weight_decay=0.0)],
                            lr=peak, betas=(0.9, 0.98))
    t0, step, total = time.time(), 0, minutes * 60
    run = None
    eval_no = 0
    while True:
        el = time.time() - t0
        if el > total:
            break
        warm = min(1.0, (step + 1) / 200)
        if decay_frac is None:
            lr = peak * warm * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * el / total)))
        else:
            left = (total - el) / (decay_frac * total)
            lr = peak * warm * min(1.0, max(0.0, left))
        if eval_every and el > (eval_no + 1) * eval_every * 60:
            eval_no += 1
            save_model(model, out)
            tot, line = evaluate(model)
            print('  EVAL at %.0f min: %.1f%%  %s' % (el / 60, 100 * tot, line), flush=True)
        for g in opt.param_groups:
            g['lr'] = lr
        x, m = batch(rng, bs)
        loss = loss_fn(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % log_every == 0:
            print('step %5d  %4.0fs  lr %.5f  loss %.3f' % (step, el, lr, run), flush=True)
    save_model(model, out)
    print('saved', out, 'steps', step, 'time %.0fs' % (time.time() - t0), flush=True)
    return model


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        sys.exit(__doc__)
    cmd = sys.argv[1]
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    if cmd == 'bench':
        rng = random.Random(0)
        for c in [Cfg(96, 3, 4, 192), Cfg(96, 4, 4, 256), Cfg(112, 4, 4, 224)]:
            m = Model(c)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            for i in range(13):
                if i == 3:
                    t = time.time()
                x, k = batch(rng, 96)
                l = loss_fn(m, x, k); opt.zero_grad(); l.backward(); opt.step()
            print(c, sum(p.numel() for p in m.parameters()), '%.3f s/step' % ((time.time() - t) / 10), flush=True)
    elif cmd == 'train':
        sizes = {k: int(arg('--' + k)) for k in ('D', 'L', 'H', 'HID') if arg('--' + k)}
        resume, grow_from, config = arg('--resume'), arg('--grow'), arg('--config')
        if resume and (sizes or grow_from or config):
            sys.exit('train: --resume keeps the model as it is; use --grow to change its design')
        try:
            base = vars(load_model(grow_from).c) if grow_from else {}
            cfg = Cfg(**{**base, **(read_config(config) if config else {}), **sizes})
        except (ValueError, OSError) as e:
            sys.exit('train: %s' % e)
        m = train(float(sys.argv[2]), sys.argv[3], cfg=cfg, grow_from=grow_from, resume=resume,
                  peak=float(arg('--lr', 2e-3)),
                  decay_frac=float(arg('--decay')) if arg('--decay') else None,
                  eval_every=float(arg('--eval-every', 0)))
        tot, line = evaluate(m, show=25)
        print('eval %.1f%%  %s' % (100 * tot, line))
    elif cmd == 'eval':
        m = load_model(sys.argv[2])
        try:
            off = switch_off(m, arg('--off'))
        except ValueError as e:
            sys.exit('eval: %s' % e)
        n = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].isdigit() else 20
        tot, line = evaluate(m, show=int(n))
        print('eval %.1f%%%s  %s' % (100 * tot, ' (off: %s)' % ','.join(off) if off else '', line))
    elif cmd == 'info':
        info(load_model(sys.argv[2]))
    else:
        sys.exit('unknown command %r\n%s' % (cmd, __doc__))
