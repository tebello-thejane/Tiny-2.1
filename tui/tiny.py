#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.15"
# dependencies = ["numpy", "torch", "textual"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""TINY in the terminal: chat with a .pt checkpoint, watch it think, switch parts off.

  ./tiny.py                 full-screen chat with the live neural monitor
  ./tiny.py 0               creativity 0 (always the surest answer) .. 9
  ./tiny.py --model M.pt    another checkpoint (default: ../training/tiny2.pt)
  ./tiny.py --off 2.3,L4    start with layer 2 head 3 and all of layer 4 switched off
  ./tiny.py --demo          TINY asks itself sample questions; any key takes over
  ./tiny.py --plain         plain text (automatic when stdin is not a terminal)
  ./tiny.py --delay S       seconds per answer letter on screen (default 0.025, 0 = instant)

The network is the Model class in training/train2.py, so whatever you change
there (sizes, layers, the wiring itself) is what this program runs."""
import os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'training'))

import numpy as np
import torch

import train2 as tr

CTX, MAXQ = tr.CTX, 72


# ---------------------------------------------------------------- thinking
def pick(score, creativity, rng):
    """Choose the next letter; returns (letter, confidence).
    Creativity 0 always takes the best letter. Otherwise it picks at random
    among the letters that fit nearly as well, so answers vary in wording."""
    best = int(np.argmax(score))
    tok = best
    if creativity > 0:
        z = (score - score[best]) / (creativity / 5.0)
        bs = np.where(z < -60.0, 0.0, np.exp(z))
        bs[bs < 0.15] = 0.0                       # never pick unlikely letters
        r = bs.sum() * rng.random()
        for i in np.flatnonzero(bs):
            r -= bs[i]
            if r < 0.0:
                tok = int(i)
                break
    d = score - score[tok]
    return tok, 1.0 / float(np.where(d < -60.0, 0.0, np.exp(d)).sum())


def activity(model):
    """How busy each layer's middle part was for the newest letter (0 when off)."""
    return [float(b.act[0, -1]) if b.on and b.act is not None else 0.0 for b in model.blocks]


@torch.inference_mode()
def think(model, q, creativity, rng, watch=True):
    """Answer q, yielding ('read', i, n, activity) per question letter (only when
    watch: it re-runs the network on each prefix so the monitor can show it reading),
    ('letter', c, confidence, activity) per answer letter, then ('done', answer, seconds)."""
    model.eval()
    for b in model.blocks:
        b.record = True
    t0 = time.perf_counter()
    ids = tr.encode(q + '\n')
    for i in range(len(q) if watch else 0):
        model(torch.tensor([ids[:i + 1]]))
        yield 'read', i, len(q), activity(model)
    out = []
    while len(ids) < CTX:
        score = model(torch.tensor([ids]))[0, -1].float().numpy()
        tok, conf = pick(score, creativity, rng)
        if tok <= 1:
            break                                 # end of the answer
        ids.append(tok)
        out.append(chr(tok + 30))
        yield 'letter', out[-1], conf, activity(model)
    yield 'done', ''.join(out), time.perf_counter() - t0


def clean(line):
    """Printable letters only, at most MAXQ, no trailing spaces, CAPS LOCK to lower case."""
    q = ''.join(c for c in line if 32 <= ord(c) <= 126)[:MAXQ].rstrip(' ')
    if any(c.isupper() for c in q) and not any(c.islower() for c in q):
        q = q.lower()
    return q


def load(path):
    if not os.path.exists(path):
        raise SystemExit('TINY: cannot find %s (use --model FILE.pt)' % path)
    return tr.load_model(path)


# ---------------------------------------------------------------- plain text
def plain_main(model, creativity):
    rng = random.Random()
    c = model.c
    off = tr.switched_off(model)
    print('TINY - a neural network that answers questions.')
    print('Ready: %d layers, %d wide, %d heads. Creativity %d.%s' % (
        c.L, c.D, c.H, creativity, ' Off: %s.' % ','.join(off) if off else ''))
    print('Type QUIT to stop.')
    while True:
        print('\n> ', end='', flush=True)
        line = sys.stdin.readline()
        if not line:
            break
        q = clean(line.rstrip('\n'))
        if not q:
            continue
        if q.lower() in ('quit', 'exit'):
            break
        print('  ', end='')
        for ev in think(model, q, creativity, rng, watch=False):
            if ev[0] == 'letter':
                print(ev[1], end='', flush=True)
            elif ev[0] == 'done':
                print('\n   [%d letters in %.1f seconds]' % (len(ev[1]), ev[2]))
    print('Goodbye!')


# ---------------------------------------------------------------- full screen
DEMO_Q = ["what is the capital of estonia", "is a whale a fish", "what is 7 times 8",
          "which countries border estonia", "spell computer", "what is the symbol for gold",
          "how do you say thank you in spanish", "what is a group of lions called",
          "what comes after 199", "what time is it", "tell me a joke", "name the continents"]

HELP = ["Ask a question and press Enter.",
        "Try: capital of peru · is a whale a fish · what is 7 times 8",
        "Surgery: click a head (●) or a layer name to switch it off and on,",
        "     or type  off 2.3  ·  off L4  ·  on 2.3  ·  on all",
        "F4 / Shift+F4 creativity · F5 everything back on",
        "Up/Down earlier questions · HELP · CLS (Ctrl+L) · Esc or QUIT"]

MINT, CYAN, PINK, AMBER, RED, DIM, GLOW = '#3cff9e', '#00e5ff', '#ff4fd8', '#ffbf3c', '#ff4d5e', '#4a5578', '#ffffff'


def conf_colour(v):
    return MINT if v > 0.8 else AMBER if v > 0.5 else RED


def ui_main(model, creativity, demo, delay):
    from rich.text import Text
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Footer, Header, Input, Static

    c = model.c
    weights = tr.n_params(model)

    def meter(v, width, colour):
        n = max(0, min(2 * width, int(v * width * 2)))
        t = Text('█' * (n // 2), style=colour)
        if n % 2:
            t.append('▌', style=colour)
        t.append('·' * (width - (n + 1) // 2), style=DIM)
        return t

    class Monitor(Static):
        def on_mount(self):
            self.act = np.zeros(c.L)
            self.act_max = np.zeros(c.L)
            self.conf, self.heat, self.speed, self.status = 0.0, [], 0.0, 'ready'
            self.border_title = 'NEURAL MONITOR'
            self.redraw()

        def feed(self, act=None, conf=None):
            if act is not None:
                self.act = np.array(act)
            if conf is not None:
                self.conf = conf
                self.heat = (self.heat + [conf])[-18:]
            self.redraw()

        def redraw(self):
            w = max(8, (self.size.width or 26) - 8)
            t = Text('LAYER ACTIVITY\n', style=DIM)
            self.act_max = np.maximum(self.act_max, self.act)
            for l in range(c.L):
                t.append('L%-2d ' % (l + 1), style=CYAN if model.blocks[l].on else RED)
                if not model.blocks[l].on:
                    t.append('switched off\n', style=RED)
                    continue
                v = float(self.act[l] / self.act_max[l]) if self.act_max[l] > 0 else 0.0
                t.append_text(meter(v, w, CYAN if l % 2 else MINT))
                t.append('\n')
            self.act_max *= 0.995
            t.append('\nCONFIDENCE\n', style=DIM)
            t.append_text(meter(self.conf, w, conf_colour(self.conf)))
            t.append(' %3d%%\n' % int(self.conf * 100), style='bold ' + GLOW)
            t.append('\nLAST LETTERS\n', style=DIM)
            for i in range(18):
                if i < len(self.heat):
                    v = self.heat[i]
                    t.append('█' if v > 0.9 else '▓' if v > 0.7 else '▒' if v > 0.45 else '░', style=conf_colour(v))
                else:
                    t.append('·', style=DIM)
            t.append('\n\nSPEED  ', style=DIM)
            t.append('%.0f letters/s\n' % self.speed)
            t.append('MODE   ', style=DIM)
            t.append('creativity %d%s\n' % (app.creativity, ' (surest)' if app.creativity == 0 else ''))
            t.append(self.status, style=AMBER if self.status != 'ready' else DIM)
            self.update(t)

    class Pill(Static):
        """One clickable switch: a head (h >= 0) or a whole layer (h is None)."""

        def __init__(self, l, h):
            super().__init__(classes='pill layer' if h is None else 'pill')
            self.l, self.h = l, h

        def on_mount(self):
            self.show()

        def show(self):
            b = model.blocks[self.l]
            if self.h is None:
                self.update(Text('L%d' % (self.l + 1), style=CYAN if b.on else 'strike ' + RED))
            elif not b.attn:
                self.update(Text('·', style=DIM))          # this layer has no attention heads
            else:
                on = bool(b.head_on[self.h]) and b.on
                self.update(Text('●' if on else '○', style=MINT if on else RED))
            self.tooltip = ('layer %d' % (self.l + 1)) if self.h is None else ('layer %d, head %d' % (self.l + 1, self.h + 1))

        def on_click(self):
            if self.h is not None and not model.blocks[self.l].attn:
                app.say(Text('Layer %d has no attention, so no heads to switch.' % (self.l + 1)), 'info')
                return
            spec = 'L%d' % (self.l + 1) if self.h is None else '%d.%d' % (self.l + 1, self.h + 1)
            on = model.blocks[self.l].on if self.h is None else bool(model.blocks[self.l].head_on[self.h])
            app.surgery('off' if on else 'on', spec)

    class AskInput(Input):
        BINDINGS = [Binding('up', 'older', show=False), Binding('down', 'newer', show=False)]

        def action_older(self):
            app.browse(-1)

        def action_newer(self):
            app.browse(+1)

    class Tiny(App):
        TITLE = 'T I N Y'
        SUB_TITLE = '%s · %s weights' % (c, format(weights, ','))
        CSS = """
        Screen { background: #05060f; }
        Header { background: #0a1a3a; color: #00e5ff; }
        #chat { border: round #00e5ff; border-title-color: #00e5ff; padding: 0 1; background: #05060f; }
        #side { width: 32; }
        #monitor { border: round #0b7f90; border-title-color: #00e5ff; padding: 0 1; height: auto; }
        #surgery { border: round #0b7f90; border-title-color: #00e5ff; padding: 0 1; height: auto; }
        .row { height: 1; }
        .pill { width: 2; height: 1; }
        .pill.layer { width: 4; }
        .pill:hover { background: #0d1b2e; }
        #ask { border: round #0b7f90; border-title-color: #00e5ff; background: #05060f; color: #ffbf3c; }
        #ask:focus { border: round #ff4fd8; }
        .you { color: #ffbf3c; margin-top: 1; }
        .tiny { color: #3cff9e; }
        .info { color: #4a5578; }
        .cut { color: #ff4d5e; }
        .help { color: #c8d0e0; background: #0d1b2e; padding: 0 1; margin-top: 1; }
        Footer { background: #0b7f90; }
        """
        BINDINGS = [Binding('f1', 'help', 'Help'),
                    Binding('f4', 'creativity(1)', 'Creativity+'),
                    Binding('shift+f4', 'creativity(-1)', 'Creativity-', show=False),
                    Binding('f5', 'all_on', 'All on'),
                    Binding('ctrl+l', 'cls', 'Clear'),
                    Binding('escape', 'quit', 'Quit')]

        def __init__(self):
            super().__init__()
            self.creativity, self.busy, self.demo = creativity, False, demo
            self.history, self.hpos = [], 0
            self.rng = random.Random()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal():
                yield VerticalScroll(id='chat')
                with Vertical(id='side'):
                    yield Monitor(id='monitor')
                    with Vertical(id='surgery'):
                        for l in range(c.L):
                            with Horizontal(classes='row'):
                                yield Pill(l, None)
                                for h in range(c.H):
                                    yield Pill(l, h)
            yield AskInput(placeholder='ask TINY something, or: off 2.3 · on all · help',
                           max_length=MAXQ, id='ask')
            yield Footer()

        def on_mount(self):
            self.chat = self.query_one('#chat', VerticalScroll)
            self.monitor = self.query_one(Monitor)
            self.ask_box = self.query_one(AskInput)
            self.chat.border_title = 'CONVERSATION'
            self.ask_box.border_title = 'ASK TINY'
            self.query_one('#surgery').border_title = 'SURGERY  layer · heads'
            off = tr.switched_off(model)
            self.say(Text('Hello! I am TINY: %d layers of %d heads, %s weights.\n'
                          'Ask about countries, animals, space, words or sums. F1 for help.'
                          % (c.L, c.H, format(weights, ','))), 'info')
            if off:
                self.say(Text('Switched off: %s' % ', '.join(off)), 'cut')
            if self.demo:
                self.ask_box.disabled = True
                self.run_demo()
            else:
                self.ask_box.focus()

        # -- chat
        def say(self, text, cls):
            w = Static(text, classes=cls)
            self.chat.mount(w)
            self.chat.scroll_end(animate=False)
            return w

        def action_help(self):
            self.say(Text('HELP\n', style='bold ' + CYAN) + Text('\n'.join(HELP)), 'help')

        def action_cls(self):
            self.chat.remove_children()

        def action_creativity(self, d: int):
            self.creativity = (self.creativity + d) % 10
            self.monitor.redraw()

        def action_all_on(self):
            self.surgery('on', 'all')

        # -- surgery
        def surgery(self, verb, spec):
            if self.busy:
                self.say(Text('Wait for TINY to finish the answer first.'), 'info')
                return
            try:
                if spec.strip().lower() == 'all':
                    if verb == 'on':
                        tr.switch_all_on(model)
                    else:
                        tr.switch_off(model, ','.join('L%d' % (l + 1) for l in range(c.L)))
                    done = ['everything']
                else:
                    done = (tr.switch_on if verb == 'on' else tr.switch_off)(model, spec)
            except ValueError as e:
                self.say(Text(str(e)), 'cut')
                return
            for p in self.query(Pill):
                p.show()
            off = tr.switched_off(model)
            self.say(Text('Switched %s %s. Now off: %s' % (verb, ', '.join(done), ', '.join(off) or 'nothing')), 'cut')
            self.monitor.redraw()

        def browse(self, d):
            if not self.history:
                return
            self.hpos = max(0, min(len(self.history), self.hpos + d))
            self.ask_box.value = self.history[self.hpos] if self.hpos < len(self.history) else ''
            self.ask_box.cursor_position = len(self.ask_box.value)

        def on_input_submitted(self, event):
            q = clean(event.value)
            self.ask_box.value = ''
            if not q or self.busy:
                return
            word = q.lower()
            if word in ('quit', 'exit'):
                self.exit()
            elif word == 'cls':
                self.action_cls()
            elif word == 'help':
                self.action_help()
            elif word.split(' ', 1)[0] in ('off', 'on') and len(word.split()) >= 2:
                verb, spec = word.split(' ', 1)
                self.surgery(verb, spec.replace(' ', ''))
            else:
                if not self.history or self.history[-1] != q:
                    self.history.append(q)
                self.hpos = len(self.history)
                self.ask(q)

        def on_key(self, event):
            if self.demo:                        # any key takes over from the demo
                self.demo = False
                event.stop()

        # -- asking
        def ask(self, q):
            self.busy = True
            self.say(Text.assemble(('YOU ▸ ', 'bold ' + PINK), (q, AMBER)), 'you')
            self.answer = self.say(Text('TINY ▸ ', style='bold ' + CYAN), 'tiny')
            self.monitor.heat, self.monitor.status = [], 'reading...'
            self.monitor.redraw()
            self.run_think(q)

        def show_reading(self, i, n, act):
            head = i % 18
            head = 17 - head if head >= 9 else head
            t = Text('TINY ▸ ', style='bold ' + CYAN)
            for k in range(10):
                d = abs(head - k)
                t.append('█▓▒'[d] if d < 3 else '·', style=GLOW if d == 0 else PINK if d < 3 else DIM)
            t.append(' reading %d%%' % (100 * (i + 1) // n), style=DIM)
            self.answer.update(t)
            self.monitor.feed(act)

        def show_letters(self, text, conf, act):
            t = Text('TINY ▸ ', style='bold ' + CYAN)
            t.append(text[:-1], style=MINT)
            t.append(text[-1:], style='bold ' + GLOW)
            self.answer.update(t)
            self.chat.scroll_end(animate=False)
            self.monitor.status = 'thinking...'
            self.monitor.feed(act, conf)

        def finish(self, text, secs):
            self.answer.update(Text('TINY ▸ ', style='bold ' + CYAN) + Text(text or '.', style=MINT))
            off = tr.switched_off(model)
            self.say(Text('%d letters · %.2f s of thinking%s' % (
                len(text), secs, ' · off: %s' % ', '.join(off) if off else '')), 'info')
            self.monitor.speed = len(text) / secs if secs > 0 else 0.0
            self.monitor.status = 'ready'
            self.monitor.feed()
            self.busy = False

        @work(thread=True, exclusive=True)
        def run_think(self, q):
            out = ''
            for ev in think(model, q, self.creativity, self.rng):
                if ev[0] == 'read':
                    self.call_from_thread(self.show_reading, ev[1], ev[2], ev[3])
                    if delay:
                        time.sleep(delay / 4)
                elif ev[0] == 'letter':
                    out += ev[1]
                    self.call_from_thread(self.show_letters, out, ev[2], ev[3])
                    if delay:
                        time.sleep(delay)
                else:
                    self.call_from_thread(self.finish, ev[1], ev[2])

        @work(exclusive=False, group='demo')
        async def run_demo(self):
            import asyncio
            for q in DEMO_Q:
                if not self.demo:
                    break
                for i in range(len(q)):
                    if not self.demo:
                        break
                    self.ask_box.value = q[:i + 1]
                    await asyncio.sleep(0.03 + self.rng.random() * 0.04)
                if not self.demo:
                    break
                await asyncio.sleep(0.2)
                self.ask_box.value = ''
                self.history.append(q)
                self.hpos = len(self.history)
                self.ask(q)
                while self.busy:
                    await asyncio.sleep(0.05)
                await asyncio.sleep(1.2)
            self.demo = False
            self.ask_box.value = ''
            self.ask_box.disabled = False
            self.ask_box.focus()

    app = Tiny()
    return app


def parse(argv):
    o = dict(creativity=5, plain=False, demo=False, model=os.path.join(HERE, '..', 'training', 'tiny2.pt'),
             off='', delay=0.025)
    it = iter(argv)
    for a in it:
        if a.isdigit() and len(a) == 1:
            o['creativity'] = int(a)
        elif a.lower() in ('--plain', '/p', '-p'):
            o['plain'] = True
        elif a.lower() in ('--demo', '/demo'):
            o['demo'] = True
        elif a in ('--model', '--off', '--delay'):
            v = next(it, None)
            if v is None:
                sys.exit('TINY: %s needs a value (try --help)' % a)
            o[a[2:]] = float(v) if a == '--delay' else v
        elif a in ('-h', '--help'):
            print(__doc__)
            sys.exit(0)
        else:
            sys.exit('TINY: unknown option %s (try --help)' % a)
    return o


def main():
    o = parse(sys.argv[1:])
    model = load(o['model'])
    try:
        tr.switch_off(model, o['off'])
    except ValueError as e:
        sys.exit('TINY: --off: %s' % e)
    if o['plain'] or not (sys.stdin.isatty() and sys.stdout.isatty()):
        plain_main(model, o['creativity'])
    else:
        ui_main(model, o['creativity'], o['demo'], o['delay']).run()


if __name__ == '__main__':
    main()
