#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""TINY training data, version 2.

Turns the fact files in facts/ into an endless stream of (question, answer)
pairs. Compared with version 1 it adds:
  - many more phrasings for every table relation
  - reverse questions      (Paris is the capital of which country?)
  - yes/no questions       (is a whale a fish? -> No, a whale is a mammal.)
  - two-part questions     (capital and money of Japan, capitals of X and Y)
  - numbers                (sums, times tables, next number, odd/even,
                            bigger/smaller, number words, counting)
  - letters and spelling   (first letter, spell it, backwards, letter after)
  - "I do not know"        for made-up countries, animals and words
  - typing noise           (typos, CAPS LOCK, "hey tiny,", "please", ...)

Every generator keeps some phrasings and items back for testing, so the
test set measures whether the model generalises, not whether it memorised.

Run: ./gen2.py   -> statistics and samples."""
import glob, os, random, collections, string

HERE = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(HERE, 'facts')
MAXLEN = 126            # question + newline + answer + newline <= 128 letters


def cap(s):
    return s[0].upper() + s[1:] if s else s


def art(w):
    if w.lower().startswith(('ewe', 'uni', 'use', 'eu', 'one', 'ur')):
        return 'a ' + w
    return ('an ' if w[0].lower() in 'aeiou' or w.lower().startswith('hour') else 'a ') + w


# ---------------------------------------------------------------- loading
def load_txt():
    facts = []
    for fn in sorted(glob.glob(os.path.join(FACTS, '*.txt'))):
        theme = os.path.basename(fn)[:-4]
        for line in open(fn):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            qs, ans = line.split('=>')
            qs = [q.strip() for q in qs.split('|')]
            ans = [a.strip() for a in ans.split('|')]
            if len(qs) >= 3:
                train_q, test_q = qs[:-1], qs[-1:]
            else:
                train_q, test_q = qs, []
            facts.append(dict(theme=theme, train_q=train_q, test_q=test_q, answers=ans))
    return facts


def load_tsv():
    rel = collections.defaultdict(list)          # relation -> [(subject, answer)]
    for fn in sorted(glob.glob(os.path.join(FACTS, '*.tsv'))):
        base = os.path.basename(fn)
        for line in open(fn):
            r, s, a = line.rstrip('\n').split('\t')
            if base == 'geo.tsv' and r in ('language', 'currency'):
                continue                         # geo2.tsv has the fuller lists
            if base == 'elements.tsv':
                s = s.lower()
            if r == 'borders':
                continue                         # used by g_borders
            rel[r].append((s, a))
    return rel


TXT = load_txt()
REL = load_tsv()
BORDERS = {}
for _line in open(os.path.join(FACTS, 'geo3.tsv')):
    _r, _s, _a = _line.rstrip('\n').split('\t')
    if _r == 'borders':
        BORDERS[_s] = [] if _a == '-' else _a.split('|')
D = {r: dict(v) for r, v in REL.items()}         # relation -> {subject: answer}


def split(items, frac=0.1, seed=7):
    """Deterministic train/test split of a list of items."""
    items = sorted(items)
    r = random.Random(seed)
    test = set(r.sample(items, max(1, int(len(items) * frac))))
    return [x for x in items if x not in test], sorted(test)


def pick_q(qs, rng, test):
    """Last phrasing of every list is kept for testing only."""
    return qs[-1] if test else rng.choice(qs[:-1])


# ---------------------------------------------------------------- phrasings
# {s} subject, {a} answer, {A} answer capitalised. Last q of each list = test only.
REL_T = {
    'capital': (['what is the capital of {s}', 'what is the capital city of {s}', 'capital of {s}',
                 "what is {s}'s capital", 'which city is the capital of {s}', 'name the capital of {s}',
                 'tell me the capital of {s}', '{s} capital', 'do you know the capital of {s}',
                 'the capital of {s} is', 'what city is the capital of {s}', 'capital city of {s}',
                 'where is the government of {s}', 'what is the main city of {s}',
                 'which city is {s}\'s capital'],
                ['{a}.', 'It is {a}.', 'The capital is {a}.', 'The capital of {s} is {a}.']),
    'continent': (['which continent is {s} in', 'what continent is {s} in', 'on which continent is {s}',
                   'where is {s}', 'in which part of the world is {s}', '{s} is in which continent',
                   'continent of {s}', 'which continent does {s} belong to', 'is {s} in europe or asia',
                   'where in the world is {s}', 'what continent is {s} on'],
                  ['{a}.', 'In {a}.', 'It is in {a}.', '{s} is in {a}.']),
    'language': (['what language is spoken in {s}', 'what language do people speak in {s}',
                  'which language do they speak in {s}', 'what is the language of {s}',
                  'what do people speak in {s}', 'language of {s}', 'what language do they use in {s}',
                  'what language do people in {s} speak', 'which language is used in {s}',
                  'what is the main language of {s}'],
                 ['{a}.', 'They speak {a}.', 'People there speak {a}.', 'In {s} people speak {a}.']),
    'currency': (['what is the currency of {s}', 'what money do people use in {s}',
                  'what money is used in {s}', 'which currency is used in {s}', 'currency of {s}',
                  'what do people pay with in {s}', 'what is the money of {s} called', 'money of {s}',
                  'what money do they have in {s}', 'what do they pay with in {s}',
                  'what is the money in {s}'],
                 ['The {a}.', 'It is the {a}.', 'They use the {a}.', 'The money in {s} is the {a}.']),
    'people': (['what are people from {s} called', 'what do you call people from {s}',
                'what nationality are people from {s}', 'what is someone from {s} called',
                'people from {s} are called what', 'what is the nationality of {s}',
                'what do we call a person from {s}', 'what nationality is someone from {s}'],
               ['{a}.', 'They are called {a}.' , 'The nationality is {a}.']),
    'month_after': (['which month comes after {s}', 'what month comes after {s}', 'what is the month after {s}',
                     'what month follows {s}', 'month after {s}', 'after {s} comes which month',
                     'what is the next month after {s}', 'which month is after {s}',
                     'which month is next after {s}'],
                    ['{a}.', 'It is {a}.', '{a} comes after {s}.']),
    'month_before': (['which month comes before {s}', 'what month comes before {s}', 'what is the month before {s}',
                      'month before {s}', 'before {s} comes which month', 'which month is before {s}',
                      'what month is just before {s}', 'which month is just before {s}'],
                     ['{a}.', 'It is {a}.', '{a} comes before {s}.']),
    'month_days': (['how many days are in {s}', 'how many days does {s} have', 'how many days has {s}',
                    'how long is {s} in days', 'days in {s}', 'number of days in {s}', '{s} has how many days',
                    'how many days are there in {s}'],
                   ['{A}.', '{s} has {a} days.', 'It has {a} days.']),
    'month_number': (['what is the {s} month of the year', 'which month is the {s} month', 'what is the {s} month',
                      'name the {s} month of the year', 'the {s} month is called what', 'what month is {s}',
                      'which is the {s} month of the year'],
                     ['{a}.', 'It is {a}.', 'The {s} month is {a}.']),
    'day_after': (['which day comes after {s}', 'what day comes after {s}', 'what is the day after {s}',
                   'what day follows {s}', 'day after {s}', 'if today is {s} what is tomorrow',
                   'what day is after {s}', 'today is {s}, what day is tomorrow',
                   'which day is next after {s}'],
                  ['{a}.', 'It is {a}.', '{a} comes after {s}.']),
    'day_before': (['which day comes before {s}', 'what day comes before {s}', 'what is the day before {s}',
                    'day before {s}', 'if today is {s} what was yesterday', 'what day is before {s}',
                    'today is {s}, what was yesterday', 'which day is just before {s}'],
                   ['{a}.', 'It is {a}.', '{a} comes before {s}.']),
    'opposite': (['what is the opposite of {s}', 'opposite of {s}', 'what is the opposite word for {s}',
                  'tell me the opposite of {s}', 'what word means the opposite of {s}',
                  'the opposite of {s} is what', "what's the opposite of {s}", 'antonym of {s}',
                  'what is the antonym of {s}', 'give me the opposite of {s}'],
                 ['{A}.', 'It is {a}.', 'The opposite of {s} is {a}.']),
    'plural': (['what is the plural of {s}', 'plural of {s}', 'what do you call more than one {s}',
                'how do you say many of {s}', 'one {s} and two what', 'what is the plural form of {s}',
                'one {s}, two', "what's the plural of {s}", 'give me the plural of {s}'],
               ['{A}.', 'It is {a}.', 'One {s}, two {a}.']),
    'past': (['what is the past tense of {s}', 'past tense of {s}', 'what is the past of {s}',
              'how do you say {s} in the past', 'today i {s} and yesterday i what',
              'what is the past form of {s}', "what's the past tense of {s}", 'yesterday i ... ({s})',
              'give me the past tense of {s}'],
             ['{A}.', 'It is {a}.', 'The past of {s} is {a}.']),
    'class': (['what kind of animal is a {s}', 'what type of animal is a {s}', 'what group does a {s} belong to',
               'what is the animal group of a {s}', 'animal group of a {s}', 'is a {s} a mammal or what',
               'what is a {s}', 'a {s} is what kind of animal', 'what animal group is a {s} in'],
              ['A {a}.', 'It is a {a}.', 'A {s} is a {a}.']),
    'legs': (['how many legs does a {s} have', 'how many legs has a {s}', 'a {s} has how many legs',
              'number of legs of a {s}', 'what is the number of legs of a {s}', 'legs of a {s}',
              'how many legs do {s}s have', 'count the legs of a {s}', 'how many feet does a {s} have',
              'how many legs are on a {s}'],
             ['{A}.', 'A {s} has {a} legs.', 'It has {a} legs.']),
    'baby': (['what is a baby {s} called', 'what do you call a baby {s}', 'what is the baby of a {s} called',
              'a baby {s} is called what', 'what is the baby name of a {s}', 'baby name of a {s}',
              'what is a young {s} called', 'baby {s}', 'what is the name for a baby {s}',
              'what do we call a baby {s}'],
             ['{Art}.', 'It is called {art}.', 'A baby {s} is called {art}.']),
    'food': (['what does a {s} eat', 'what food does a {s} eat', 'what is the food of a {s}', 'food of a {s}',
              'what do {s}s eat', 'what does a {s} like to eat', 'what is a {s}\'s food',
              'tell me the food of a {s}', 'what does a {s} feed on'],
             ['{A}.', 'It eats {a}.', 'A {s} eats {a}.']),
    'home': (['where does a {s} live', 'what is the home of a {s}', 'where do {s}s live', 'home of a {s}',
              'where can you find a {s}', 'where is the home of a {s}', 'tell me the home of a {s}',
              'where does a {s} make its home', 'where would you find a {s}'],
             ['{A}.', 'In {a}.', 'A {s} lives in {a}.']),
    'sound': (['what sound does a {s} make', 'what does a {s} say', 'what noise does a {s} make',
               'how does a {s} sound', 'sound of a {s}', 'what sound do {s}s make', 'a {s} says what',
               'what is the sound of a {s}', 'what noise do {s}s make'],
              ['{A}!', 'A {s} says {a}.', 'It says {a}.']),
}

LANG_Q = ['how do you say {s} in {L}', 'what is {s} in {L}', '{s} in {L}', 'what is the {L} word for {s}',
          'translate {s} into {L}', 'how do you say {s} in {L} language', 'say {s} in {L}']
LANG_A = ['{A}.', 'In {L} it is {a}.', '{S} in {L} is {a}.']
REL_T.update({
    'subregion': (['what region is {s} in', 'in which region is {s}', 'which region of the world is {s} in',
                   'region of {s}', 'what is the region of {s}', 'which world region is {s} part of',
                   'where exactly is {s}'],
                  ['{a}.', 'It is in {a}.', '{s} is in {a}.']),
    'domain': (['what is the internet domain of {s}', 'what domain does {s} use', 'internet code of {s}',
                'what is the web domain of {s}', 'what do websites in {s} end with', 'domain of {s}',
                'what is the country code domain of {s}'],
               ['{a}.', 'It is {a}.', 'Websites in {s} end with {a}.']),
    'symbol': (['what is the symbol for {s}', 'what is the chemical symbol of {s}', 'symbol of {s}',
                'how do you write {s} in chemistry', 'what is the element symbol for {s}',
                'what letters stand for {s}', 'chemical symbol for {s}'],
               ['{a}.', 'It is {a}.', 'The symbol for {s} is {a}.']),
    'atomic': (['what is the atomic number of {s}', 'atomic number of {s}', 'how many protons does {s} have',
                'what number element is {s}', 'which number is {s} in the periodic table',
                'what is the element number of {s}'],
               ['{a}.', 'It is {a}.', '{S} has atomic number {a}.']),
    'kind': (['what kind of element is {s}', 'what type of element is {s}', 'is {s} a metal or a nonmetal',
              'what sort of element is {s}', 'which kind of element is {s}'],
             ['{A}.', '{S} is {a}.', 'It is {a}.']),
    'phase': (['is {s} a solid, liquid or gas', 'what state is {s} at room temperature', 'is {s} solid liquid or gas',
               'what is {s} like at room temperature', 'is {s} a solid or a gas', 'what state of matter is {s}'],
              ['{A}.', 'It is a {a}.', '{S} is a {a} at room temperature.']),
    'synonym': (['what is another word for {s}', 'what is a synonym for {s}', 'another word for {s}',
                 'synonym of {s}', 'what word means the same as {s}', 'what means the same as {s}',
                 'give me another word for {s}'],
                ['{A}.', 'It is {a}.', 'Another word for {s} is {a}.']),
    'comparative': (['what is the comparative of {s}', 'comparative of {s}', 'how do you say more {s}',
                     'what is the comparative form of {s}', '{s}, then what'],
                    ['{A}.', 'It is {a}.', 'The comparative of {s} is {a}.']),
    'superlative': (['what is the superlative of {s}', 'superlative of {s}', 'how do you say the most {s}',
                     'what is the superlative form of {s}', '{s}, ..., what is the last one'],
                    ['{A}.', 'It is {a}.', 'The superlative of {s} is {a}.']),
    'group': (['what is a group of {p} called', 'what do you call a group of {p}', 'a group of {p} is called what',
               'group of {p}', 'what is the word for a group of {p}', 'what is the name for a group of {p}'],
              ['{Art}.', 'It is called {art}.', 'A group of {p} is called {art}.']),
    'male': (['what is a male {s} called', 'what do you call a male {s}', 'male {s}', 'a boy {s} is called what',
              'what is the name for a male {s}'],
             ['{Art}.', 'A male {s} is called {art}.']),
    'female': (['what is a female {s} called', 'what do you call a female {s}', 'female {s}',
                'a girl {s} is called what', 'what is the name for a female {s}'],
               ['{Art}.', 'A female {s} is called {art}.']),
    'state_capital': (['what is the capital of the state of {s}', 'what is the capital of {s}', 'capital of {s}',
                       'which city is the capital of {s}', "what is {s}'s capital", 'state capital of {s}',
                       'what is the state capital of {s}'],
                      ['{a}.', 'It is {a}.', 'The capital of {s} is {a}.']),
})
for _L in ('French', 'German', 'Spanish', 'Italian'):
    REL_T['word_' + _L] = ([q.replace('{L}', _L) for q in LANG_Q], [a.replace('{L}', _L) for a in LANG_A])


# the old generic animal forms from version 1 (keep them working)
GENERIC = {'class': 'animal group', 'legs': 'number of legs', 'baby': 'baby name',
           'food': 'food', 'home': 'home'}
GENERIC_Q = ['what is the {l} of {x}', '{l} of {x}', 'tell me the {l} of {x}', '{x}: {l}',
             'do you know the {l} of {x}', 'what is the {l} for {x}']

THEME = {'capital': 'geography', 'continent': 'geography', 'language': 'geography',
         'currency': 'geography', 'people': 'geography', 'landlocked': 'geography'}
THEME.update({'subregion': 'geography', 'domain': 'geography', 'state_capital': 'geography',
              'symbol': 'science', 'atomic': 'science', 'kind': 'science', 'phase': 'science'})



def plural_of(w):
    pl = dict(REL.get('plural', []))
    if w in pl:
        return pl[w]
    if w.endswith('y') and w[-2] not in 'aeiou':
        return w[:-1] + 'ies'
    return w + ('es' if w.endswith(('s', 'x', 'ch', 'sh')) else 's')


def fill(t, s, a):
    f = dict(s=s, S=cap(s), a=a, A=cap(a), art=art(a), Art=cap(art(a)), p=plural_of(s))
    return t.format(**f)


def legs_answer(s, a, t):
    if a == 'none':
        return {'{A}.': 'None.', 'A {s} has {a} legs.': 'A {s} has no legs.',
                'It has {a} legs.': 'It has no legs.'}[t].format(s=s)
    if a == 'many':
        return {'{A}.': 'Very many.', 'A {s} has {a} legs.': 'A {s} has very many legs.',
                'It has {a} legs.': 'It has very many legs.'}[t].format(s=s)
    return fill(t, s, a)


def month_days_answer(s, a, t):
    if s == 'February':
        return {'{A}.': '28, or 29 in a leap year.', '{s} has {a} days.': 'February has 28 days, or 29 in a leap year.',
                'It has {a} days.': 'It has 28 days, or 29 in a leap year.'}[t]
    return fill(t, s, a)


def rel_qa(r, s, a, rng, test):
    qs, ans = REL_T[r]
    if r in GENERIC and not test and rng.random() < 0.2:
        x = art(s)
        q = rng.choice(GENERIC_Q).format(l=GENERIC[r], x=x)
    else:
        q = fill(pick_q(qs, rng, test), s, a)
    t = rng.choice(ans)
    if r == 'legs':
        a2 = legs_answer(s, a, t)
    elif r == 'month_days':
        a2 = month_days_answer(s, a, t)
    else:
        a2 = fill(t, s, a)
    return q, a2


# ---------------------------------------------------------------- generators
SWAPS = [('what is ', "what's "), ('which ', 'what '), ('what ', 'which '), ('biggest', 'largest'),
         ('largest', 'biggest'), ('smallest', 'tiniest'), ('colour', 'color'), ('color', 'colour'),
         ('how many', 'what number of'), ('do we ', 'do you '), ('do you ', 'do we '), ('what are ', 'name '),
         ('name the ', 'what are the '), ('what do we call ', 'what is the name for '), (' the ', ' a '),
         ('is called', 'is named'), ('who ', 'which person '), ('hottest', 'warmest'), ('coldest', 'chilliest'),
         ('tallest', 'highest'), ('highest', 'tallest'), ('closest', 'nearest'), ('nearest', 'closest'),
         ('make', 'create'), ('in the world', 'on earth'), ('on earth', 'in the world')]


def reword(q, rng):
    """Small automatic rewordings, so one-off facts get more phrasings."""
    for _ in range(rng.randint(1, 2)):
        opts = [sw for sw in SWAPS if sw[0] in q]
        if not opts:
            break
        a, b = rng.choice(opts)
        q = q.replace(a, b, 1)
    return q


def g_txt(rng, test):
    while True:
        f = rng.choice(TXT)
        qs = f['test_q'] if test else f['train_q']
        if qs:
            q = rng.choice(qs)
            if not test and rng.random() < 0.35:
                q = reword(q, rng)
            return q, rng.choice(f['answers']), f['theme']


REL_NAMES = [r for r in REL_T if r in REL]
REL_W = {r: len(REL[r]) ** 0.75 for r in REL_NAMES}


def g_rel(rng, test):
    r = rng.choices(REL_NAMES, weights=[REL_W[x] for x in REL_NAMES])[0]
    s, a = rng.choice(REL[r])
    q, a2 = rel_qa(r, s, a, rng, test)
    return q, a2, THEME.get(r, 'words' if r in ('opposite', 'plural', 'past', 'synonym', 'comparative',
                                                  'superlative') or r.startswith('word_') else
                             'calendar' if 'month' in r or 'day' in r else 'animals')


def g_landlocked(rng, test):
    s, a = rng.choice(REL['landlocked'])
    if rng.random() < 0.5:
        q = pick_q(['is {s} landlocked', 'is {s} a landlocked country', 'is {s} cut off from the sea',
                    'does {s} have no coast', 'is {s} surrounded by land'], rng, test).format(s=s)
        a2 = ('Yes, {s} is landlocked. It has no coast.' if a == 'yes' else 'No, {s} has a coast.').format(s=s)
    else:
        q = pick_q(['does {s} have a coast', 'does {s} have a coastline', 'is {s} by the sea',
                    'does {s} touch the sea', 'can you see the sea in {s}', 'does {s} have a sea coast'],
                   rng, test).format(s=s)
        a2 = ('No, {s} is landlocked.' if a == 'yes' else 'Yes, {s} has a coast.').format(s=s)
    return q, a2, 'geography'


def group(rel):
    g = collections.defaultdict(list)
    for s, a in REL[rel]:
        g[a].append(s)
    return g


def list_and(xs):
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]


CAP_REV = {a: ss[0] for a, ss in group('capital').items() if len(ss) == 1}
LANG_REV = group('language')
CUR_REV = group('currency')
SOUND_REV = group('sound')
BABY_REV = group('baby')
PLURAL_REV = {a: s for s, a in REL['plural'] if a != s}
PAST_REV = {a: s for s, a in REL['past'] if a != s}
MONTHNUM = {a: s for s, a in REL['month_number']}


def g_reverse(rng, test):
    k = rng.choice(['cap', 'cap', 'cap', 'lang', 'cur', 'sound', 'baby', 'plural', 'past', 'month'])
    if k == 'cap':
        a, s = rng.choice(sorted(CAP_REV.items()))
        q = pick_q(['{a} is the capital of which country', 'which country has the capital {a}',
                    'what country is {a} the capital of', '{a} is in which country',
                    'where is {a}', 'in which country is {a}', 'which country is {a} in',
                    '{a} is the capital of what', 'of which country is {a} the capital'], rng, test).format(a=a)
        ans = rng.choice(['{s}.', '{a} is in {s}.', '{a} is the capital of {s}.', 'It is in {s}.']).format(a=a, s=s)
        return q, ans, 'geography'
    if k == 'lang':
        a, ss = rng.choice(sorted(LANG_REV.items()))
        q = pick_q(['where do people speak {a}', 'in which country do people speak {a}',
                    'which country speaks {a}', 'where is {a} spoken', 'who speaks {a}',
                    'in which countries is {a} spoken'], rng, test).format(a=a)
        names = ss if len(ss) <= 3 else ss[:2] + ['other countries']
        return q, 'In ' + list_and(names) + '.', 'geography'
    if k == 'cur':
        a, ss = rng.choice(sorted(CUR_REV.items()))
        q = pick_q(['which country uses the {a}', 'where do people pay with the {a}', 'where is the {a} used',
                    'which country has the {a}', 'the {a} is the money of which country',
                    'what country uses the {a}'], rng, test).format(a=a)
        names = ss if len(ss) <= 3 else ss[:2] + ['many others']
        return q, list_and(names) + '.', 'geography'
    if k == 'sound':
        a, ss = rng.choice(sorted(SOUND_REV.items()))
        q = pick_q(['which animal says {a}', 'what animal says {a}', 'who says {a}', 'what animal goes {a}',
                    'which animal makes the sound {a}', 'what says {a}'], rng, test).format(a=a)
        return q, list_and([cap(art(x)) if i == 0 else art(x) for i, x in enumerate(ss)]) + '.', 'animals'
    if k == 'baby':
        a, ss = rng.choice(sorted(BABY_REV.items()))
        q = pick_q(['{x} is the baby of which animal', 'whose baby is called {x}',
                    'which animal has a baby called {x}', 'what animal does {x} grow up to be',
                    '{x} is a baby what', 'what is {x} the baby of'], rng, test).format(x=art(a))
        return q, list_and([cap(art(x)) if i == 0 else art(x) for i, x in enumerate(ss)]) + '.', 'animals'
    if k == 'plural':
        a, s = rng.choice(sorted(PLURAL_REV.items()))
        q = pick_q(['what is the singular of {a}', 'what is one of {a} called', 'singular of {a}',
                    '{a} is the plural of what', 'what is the singular form of {a}'], rng, test).format(a=a)
        return q, rng.choice(['{S}.', 'One {s}, two {a}.', 'It is {s}.']).format(s=s, S=cap(s), a=a), 'words'
    if k == 'past':
        a, s = rng.choice(sorted(PAST_REV.items()))
        q = pick_q(['{a} is the past tense of which word', 'what is the present tense of {a}',
                    '{a} is the past of what', 'which word has the past tense {a}',
                    'what is {a} the past tense of'], rng, test).format(a=a)
        return q, rng.choice(['{S}.', 'The past of {s} is {a}.', 'It is {s}.']).format(s=s, S=cap(s), a=a), 'words'
    a, s = rng.choice(sorted(MONTHNUM.items()))
    q = pick_q(['which month of the year is {a}', 'what number month is {a}', 'where is {a} in the year',
                'is {a} the first month', 'which month number is {a}'], rng, test).format(a=a)
    return q, '{a} is the {s} month.'.format(a=a, s=s), 'calendar'


def other(items, not_this, rng):
    while True:
        x = rng.choice(items)
        if x != not_this:
            return x


CLASSES = sorted(set(D['class'].values()))
CONTS = sorted(set(D['continent'].values()))
CAPS_BY_CONT = collections.defaultdict(list)
for _s, _c in REL['capital']:
    CAPS_BY_CONT[D['continent'].get(_s)].append(_c)
LANGS = sorted(set(l for v in D['language'].values() for l in v.replace(', ', ' and ').split(' and ')))
NAIVE_PLURAL = {s: s + ('es' if s.endswith(('s', 'x', 'ch', 'sh')) else 's') for s, a in REL['plural']}
NAIVE_PAST = {s: s + ('d' if s.endswith('e') else 'ed') for s, a in REL['past']}
COUNTRIES = sorted(D['capital'])
YN_COUNTRIES_TRAIN, YN_COUNTRIES_TEST = split(COUNTRIES, 0.08, seed=3)


def g_yesno(rng, test):
    k = rng.choice(['cap', 'cap', 'cap', 'cont', 'cont', 'lang', 'class', 'class', 'legs', 'sound',
                    'opp', 'plural', 'past', 'days'])
    yes = rng.random() < 0.5
    if k in ('cap', 'cont', 'lang'):
        pool = YN_COUNTRIES_TEST if test else YN_COUNTRIES_TRAIN
        if k == 'lang':
            pool = [c for c in pool if c in D['language']] or sorted(D['language'])
        s = rng.choice(pool)
    if k == 'cap':
        a = D['capital'][s]
        x = a if yes else other(CAPS_BY_CONT[D['continent'].get(s)] or [a], a, rng) \
            if len(CAPS_BY_CONT[D['continent'].get(s)]) > 1 else other(list(CAP_REV), a, rng)
        q = rng.choice(['is {x} the capital of {s}', 'is the capital of {s} {x}', 'is {x} the capital city of {s}',
                        '{x} is the capital of {s}, right', 'is {s}\'s capital {x}']).format(x=x, s=s)
        ans = ('Yes, {a} is the capital of {s}.' if x == a else 'No, the capital of {s} is {a}.').format(a=a, s=s)
        return q, ans, 'geography'
    if k == 'cont':
        a = D['continent'][s]
        x = a if yes else other(CONTS, a, rng)
        q = rng.choice(['is {s} in {x}', 'is {s} a country in {x}', 'is {s} part of {x}',
                        '{s} is in {x}, right', 'does {s} lie in {x}']).format(x=x, s=s)
        ans = ('Yes, {s} is in {a}.' if x == a else 'No, {s} is in {a}.').format(a=a, s=s)
        return q, ans, 'geography'
    if k == 'lang':
        a = D['language'][s]
        mine = a.replace(', ', ' and ').split(' and ')
        x = rng.choice(mine) if yes else other([l for l in LANGS if l not in mine], None, rng)
        q = rng.choice(['do people speak {x} in {s}', 'is {x} spoken in {s}', 'do they speak {x} in {s}',
                        'is the language of {s} {x}']).format(x=x, s=s)
        ans = ('Yes, people in {s} speak {a}.' if x in mine else 'No, people in {s} speak {a}.').format(a=a, s=s)
        return q, ans, 'geography'
    if k == 'class':
        s, a = rng.choice(REL['class'])
        x = a if yes else other(CLASSES, a, rng)
        q = rng.choice(['is a {s} a {x}', 'is a {s} {X}', 'are {s}s {x}s', 'is the {s} a {x}']).format(
            s=s, x=x, X=art(x))
        ans = ('Yes, a {s} is {A}.' if x == a else 'No, a {s} is {A}.').format(s=s, A=art(a))
        return q, ans, 'animals'
    if k == 'legs':
        s, a = rng.choice(REL['legs'])
        if a in ('none', 'many'):
            x = 'four' if a == 'none' else 'eight'
            q = 'does a {s} have {x} legs'.format(s=s, x=x)
            return q, ('No, a {s} has no legs.' if a == 'none' else 'No, a {s} has very many legs.').format(s=s), 'animals'
        x = a if yes else other(['two', 'four', 'six', 'eight', 'ten'], a, rng)
        q = rng.choice(['does a {s} have {x} legs', 'has a {s} got {x} legs', 'do {s}s have {x} legs']).format(s=s, x=x)
        ans = ('Yes, a {s} has {a} legs.' if x == a else 'No, a {s} has {a} legs.').format(s=s, a=a)
        return q, ans, 'animals'
    if k == 'sound':
        s, a = rng.choice(REL['sound'])
        x = a if yes else other(sorted(set(D['sound'].values())), a, rng)
        q = rng.choice(['does a {s} say {x}', 'does a {s} go {x}', 'do {s}s say {x}']).format(s=s, x=x)
        ans = ('Yes, a {s} says {a}.' if x == a else 'No, a {s} says {a}.').format(s=s, a=a)
        return q, ans, 'animals'
    if k == 'opp':
        s, a = rng.choice(REL['opposite'])
        x = a if yes else other(sorted(set(D['opposite'].values())), a, rng)
        q = rng.choice(['is {x} the opposite of {s}', 'is the opposite of {s} {x}', 'are {s} and {x} opposites'
                        ]).format(s=s, x=x)
        ans = ('Yes, {x} is the opposite of {s}.' if x == a else 'No, the opposite of {s} is {a}.').format(s=s, x=x, a=a)
        return q, ans, 'words'
    if k == 'plural':
        s, a = rng.choice(REL['plural'])
        x = a if yes else NAIVE_PLURAL[s] if NAIVE_PLURAL[s] != a else s + 'ies'
        q = rng.choice(['is the plural of {s} {x}', 'is {x} the plural of {s}', 'one {s}, two {x}, right']
                       ).format(s=s, x=x)
        ans = ('Yes, one {s}, two {a}.' if x == a else 'No, the plural of {s} is {a}.').format(s=s, a=a)
        return q, ans, 'words'
    if k == 'past':
        s, a = rng.choice(REL['past'])
        x = a if yes else NAIVE_PAST[s] if NAIVE_PAST[s] != a else s + 's'
        q = rng.choice(['is the past tense of {s} {x}', 'is {x} the past of {s}', 'yesterday i {x}, is that right'
                        ]).format(s=s, x=x)
        ans = ('Yes, the past of {s} is {a}.' if x == a else 'No, the past of {s} is {a}.').format(s=s, a=a)
        return q, ans, 'words'
    s, a = rng.choice(REL['month_days'])
    if s == 'February':
        return 'does february have 30 days', 'No, February has 28 days, or 29 in a leap year.', 'calendar'
    x = a if yes else ('30' if a == '31' else '31')
    q = rng.choice(['does {s} have {x} days', 'are there {x} days in {s}', 'has {s} got {x} days']).format(s=s, x=x)
    ans = ('Yes, {s} has {a} days.' if x == a else 'No, {s} has {a} days.').format(s=s, a=a)
    return q, ans, 'calendar'


COMBO_COUNTRIES = [c for c in COUNTRIES if c in D.get('currency', {})]


def g_combo(rng, test):
    pool = YN_COUNTRIES_TEST if test else YN_COUNTRIES_TRAIN
    k = rng.choice(['capcur', 'capcont', 'caplang', 'twocap', 'twocap', 'twocont', 'animal2', 'animal3'])
    s = rng.choice(pool)
    if k == 'capcur' and s in D.get('currency', {}):
        q = rng.choice(['what is the capital and the currency of {s}', 'tell me the capital and money of {s}',
                        'what are the capital and currency of {s}', 'capital and currency of {s}']).format(s=s)
        return q, 'The capital is {c} and the money is the {m}.'.format(c=D['capital'][s], m=D['currency'][s]), 'geography'
    if k == 'caplang' and s in D.get('language', {}):
        q = rng.choice(['what is the capital of {s} and what language do they speak',
                        'capital and language of {s}', 'tell me the capital and language of {s}']).format(s=s)
        return q, 'The capital is {c} and people speak {l}.'.format(c=D['capital'][s], l=D['language'][s]), 'geography'
    if k == 'twocap':
        t = other(pool, s, rng)
        q = rng.choice(['what are the capitals of {s} and {t}', 'capitals of {s} and {t}',
                        'what is the capital of {s} and of {t}', 'tell me the capitals of {s} and {t}']).format(s=s, t=t)
        return q, rng.choice(['{a} and {b}.', 'The capital of {s} is {a} and of {t} it is {b}.']).format(
            s=s, t=t, a=D['capital'][s], b=D['capital'][t]), 'geography'
    if k == 'twocont':
        t = other(pool, s, rng)
        q = rng.choice(['which continents are {s} and {t} in', 'where are {s} and {t}',
                        'on which continents are {s} and {t}']).format(s=s, t=t)
        return q, '{s} is in {a} and {t} is in {b}.'.format(s=s, t=t, a=D['continent'][s], b=D['continent'][t]), 'geography'
    if k == 'animal2':
        both = sorted(set(D['legs']) & set(D['sound']))
        x = rng.choice(both)
        n = D['legs'][x]
        q = rng.choice(['how many legs does {ax} have and what does it say',
                        'what does {ax} say and how many legs does it have']).format(ax=art(x))
        return q, '{AX} has {n} legs and says {s}.'.format(AX=cap(art(x)), n=n, s=D['sound'][x]), 'animals'
    if k == 'animal3':
        both = sorted(set(D['baby']) & set(D['class']))
        x = rng.choice(both)
        q = rng.choice(['what kind of animal is {ax} and what is its baby called',
                        'what is a baby {x} called and what kind of animal is it']).format(x=x, ax=art(x))
        return q, '{AX} is {c} and its baby is {b}.'.format(AX=cap(art(x)), c=art(D['class'][x]),
                                                          b=art(D['baby'][x])), 'animals'
    q = rng.choice(['where is {s} and what is its capital', 'which continent is {s} in and what is its capital',
                    'tell me the continent and capital of {s}']).format(s=s)
    return q, '{s} is in {c}. The capital is {k}.'.format(s=s, c=D['continent'][s], k=D['capital'][s]), 'geography'


# ---------------------------------------------------------------- numbers
ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen ' \
       'sixteen seventeen eighteen nineteen'.split()
TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()


def words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ('' if n % 10 == 0 else '-' + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + ' hundred' + ('' if n % 100 == 0 else ' and ' + words(n % 100))
    return 'one thousand'


ADD_TRAIN, ADD_TEST = split([(a, b) for a in range(21) for b in range(21)], 0.1, seed=11)
SUB_TRAIN, SUB_TEST = split([(a, b) for a in range(21) for b in range(a + 1)], 0.1, seed=12)
MUL_TRAIN, MUL_TEST = split([(a, b) for a in range(1, 13) for b in range(1, 13)], 0.08, seed=13)


def num(n, rng, wordy):
    return words(n) if wordy else str(n)


def g_numbers(rng, test):
    k = rng.choice(['add', 'add', 'add', 'sub', 'sub', 'mul', 'mul', 'div', 'next', 'prev', 'odd', 'big',
                    'small', 'words', 'fromwords', 'count', 'double', 'half'])
    wordy = rng.random() < 0.25
    if k in ('add', 'sub', 'mul', 'div'):
        if k == 'add':
            a, b = rng.choice(ADD_TEST if test else ADD_TRAIN); r = a + b
            op = rng.choice(['plus', 'plus', '+', 'add']); sym = 'plus'
        elif k == 'sub':
            a, b = rng.choice(SUB_TEST if test else SUB_TRAIN); r = a - b
            op = rng.choice(['minus', 'minus', '-', 'take away']); sym = 'minus'
        elif k == 'mul':
            a, b = rng.choice(MUL_TEST if test else MUL_TRAIN); r = a * b
            op = rng.choice(['times', 'times', 'x', '*', 'multiplied by']); sym = 'times'
        else:
            b, r = rng.choice(MUL_TEST if test else MUL_TRAIN); a = b * r
            op = rng.choice(['divided by', 'divided by', '/']); sym = 'divided by'
        A, B = num(a, rng, wordy), num(b, rng, wordy)
        if op == 'add':
            q = rng.choice(['add {A} and {B}', 'what is {A} add {B}', 'add {A} to {B}']).format(A=A, B=B)
        elif op in '+-x*/' and not wordy:
            q = rng.choice(['{A}{o}{B}', '{A} {o} {B}', 'what is {A} {o} {B}', 'what is {A}{o}{B}',
                            '{A} {o} {B} =', 'how much is {A} {o} {B}']).format(A=A, B=B, o=op)
        else:
            o = op if op not in '+-x*/' else sym
            q = rng.choice(['what is {A} {o} {B}', '{A} {o} {B}', 'how much is {A} {o} {B}',
                            'can you tell me what {A} {o} {B} is', 'what does {A} {o} {B} make']).format(A=A, B=B, o=o)
        if k == 'sub' and rng.random() < 0.2:
            q = rng.choice(['what is {B} less than {A}', 'take {B} from {A}', 'subtract {B} from {A}']).format(A=A, B=B)
        if k == 'mul' and rng.random() < 0.15:
            q = rng.choice(['what is {A} lots of {B}', '{A} groups of {B}']).format(A=A, B=B)
        ans = rng.choice(['{r}.', '{a} {o} {b} is {r}.', 'It is {r}.', 'The answer is {r}.']).format(
            r=r, a=a, b=b, o=sym)
        return q, ans, 'sums'
    pool_n = lambda lo, hi: rng.randint(lo, hi)
    if k == 'next':
        n = pool_n(0, 998) if rng.random() < 0.7 else rng.choice([9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 199, 299, 499, 999])
        n = min(n, 998)
        q = rng.choice(['what number comes after {n}', 'what comes after {n}', 'which number is after {n}',
                        'what is one more than {n}', 'what is {n} plus one', 'what is the next number after {n}',
                        'count on from {n}']).format(n=num(n, rng, wordy and n < 100))
        return q, rng.choice(['{r}.', '{r} comes after {n}.', 'It is {r}.']).format(r=n + 1, n=n), 'numbers'
    if k == 'prev':
        n = pool_n(1, 999) if rng.random() < 0.7 else rng.choice([10, 20, 30, 40, 50, 100, 200, 500, 1000 - 1])
        q = rng.choice(['what number comes before {n}', 'what comes before {n}', 'which number is before {n}',
                        'what is one less than {n}', 'what is {n} minus one']).format(n=num(n, rng, wordy and n < 100))
        return q, rng.choice(['{r}.', '{r} comes before {n}.', 'It is {r}.']).format(r=n - 1, n=n), 'numbers'
    if k == 'odd':
        n = pool_n(0, 999)
        q = rng.choice(['is {n} even or odd', 'is {n} odd or even', 'is {n} an even number', 'is {n} odd',
                        'is {n} even', 'even or odd: {n}']).format(n=num(n, rng, wordy and n < 100))
        return q, '{n} is {e}.'.format(n=n, e='even' if n % 2 == 0 else 'odd'), 'numbers'
    if k in ('big', 'small'):
        a = pool_n(0, 999)
        b = other(list(range(max(0, a - 30), min(999, a + 30) + 1)) if rng.random() < 0.5 else list(range(1000)), a, rng)
        if k == 'big':
            q = rng.choice(['which is bigger, {a} or {b}', 'which is more, {a} or {b}', 'is {a} bigger than {b}',
                            'which number is larger, {a} or {b}', '{a} or {b}, which is bigger']).format(a=a, b=b)
            ans = '{m} is bigger.'.format(m=max(a, b))
            if q.startswith('is '):
                ans = ('Yes, {a} is bigger than {b}.' if a > b else 'No, {a} is smaller than {b}.').format(a=a, b=b)
        else:
            q = rng.choice(['which is smaller, {a} or {b}', 'which is less, {a} or {b}', 'is {a} smaller than {b}',
                            'which number is smaller, {a} or {b}']).format(a=a, b=b)
            ans = '{m} is smaller.'.format(m=min(a, b))
            if q.startswith('is '):
                ans = ('Yes, {a} is smaller than {b}.' if a < b else 'No, {a} is bigger than {b}.').format(a=a, b=b)
        return q, ans, 'numbers'
    if k == 'words':
        n = pool_n(0, 999) if rng.random() < 0.4 else pool_n(0, 100)
        q = rng.choice(['how do you write {n} in words', 'write {n} in words', 'how do you say {n}',
                        'spell the number {n}', '{n} in words']).format(n=n)
        return q, cap(words(n)) + '.', 'numbers'
    if k == 'fromwords':
        n = pool_n(0, 100)
        q = rng.choice(['what number is {w}', 'write {w} in numbers', 'how do you write {w} as a number',
                        '{w} as a number']).format(w=words(n))
        return q, '{n}.'.format(n=n), 'numbers'
    if k == 'count':
        a = pool_n(0, 95); b = a + rng.randint(2, 9)
        q = rng.choice(['count from {a} to {b}', 'count up from {a} to {b}', 'what are the numbers from {a} to {b}'
                        ]).format(a=a, b=b)
        if rng.random() < 0.3:
            return 'count down from {b} to {a}'.format(a=a, b=b), ', '.join(str(i) for i in range(b, a - 1, -1)) + '.', 'numbers'
        return q, ', '.join(str(i) for i in range(a, b + 1)) + '.', 'numbers'
    if k == 'double':
        n = pool_n(0, 50)
        q = rng.choice(['what is double {n}', 'double {n}', 'what is twice {n}', 'what is {n} doubled']).format(
            n=num(n, rng, wordy))
        return q, rng.choice(['{r}.', 'Double {n} is {r}.']).format(r=2 * n, n=n), 'sums'
    n = 2 * pool_n(0, 50)
    q = rng.choice(['what is half of {n}', 'half of {n}', 'what is {n} halved', 'what is half {n}']).format(
        n=num(n, rng, wordy))
    return q, rng.choice(['{r}.', 'Half of {n} is {r}.']).format(r=n // 2, n=n), 'sums'


# ---------------------------------------------------------------- letters
def word_pool():
    w = set()
    for r in ('class', 'legs', 'baby', 'sound', 'food', 'home'):
        w.update(s for s, _ in REL.get(r, []))
    for r in ('opposite', 'plural', 'past'):
        for s, a in REL[r]:
            w.update([s, a])
    w.update(x.lower() for x in REL['capital'] for x in [x[1]])
    w.update('red blue green yellow black white orange purple pink brown apple banana bread milk water house '
             'school book pencil table chair window door tree flower sun moon star river mountain ocean '
             'computer keyboard mouse screen disk tiny hello friend'.split())
    return sorted(x for x in w if x.isalpha() and x.islower() and 2 <= len(x) <= 9)


WORDS = word_pool()
WORDS_TRAIN, WORDS_TEST = split(WORDS, 0.1, seed=21)
ALPHA = string.ascii_lowercase


def g_letters(rng, test):
    k = rng.choice(['first', 'last', 'spell', 'spell', 'back', 'back', 'caps', 'after', 'before', 'pos', 'count'])
    w = rng.choice(WORDS_TEST if test else WORDS_TRAIN)
    if k == 'first':
        q = rng.choice(['what is the first letter of {w}', 'what letter does {w} start with',
                        'first letter of {w}', 'which letter does {w} begin with']).format(w=w)
        return q, rng.choice(['{L}.', '{W} starts with {L}.']).format(L=w[0].upper(), W=cap(w)), 'letters'
    if k == 'last':
        q = rng.choice(['what is the last letter of {w}', 'what letter does {w} end with', 'last letter of {w}'
                        ]).format(w=w)
        return q, rng.choice(['{L}.', '{W} ends with {L}.']).format(L=w[-1].upper(), W=cap(w)), 'letters'
    if k == 'spell':
        q = rng.choice(['spell {w}', 'how do you spell {w}', 'can you spell {w}', 'spell the word {w}']).format(w=w)
        return q, '-'.join(w.upper()) + '.', 'letters'
    if k == 'back':
        q = rng.choice(['spell {w} backwards', 'what is {w} backwards', 'write {w} backwards',
                        'reverse the word {w}', 'say {w} backwards']).format(w=w)
        return q, cap(w[::-1]) + '.', 'letters'
    if k == 'caps':
        q = rng.choice(['write {w} in capital letters', 'write {w} in big letters', '{w} in upper case',
                        'make {w} uppercase']).format(w=w)
        return q, w.upper() + '.', 'letters'
    if k == 'count':
        q = rng.choice(['how many letters are in {w}', 'how many letters does {w} have', 'how long is the word {w}'
                        ]).format(w=w)
        n = words(len(w))
        return q, rng.choice([cap(n) + '.', '{W} has {n} letters.'.format(W=cap(w), n=n)]), 'letters'
    i = rng.randrange(26)
    L = ALPHA[i]
    if k == 'after' and i < 25:
        q = rng.choice(['what letter comes after {l}', 'which letter is after {l}', 'what comes after {l} in the alphabet',
                        'next letter after {l}']).format(l=L if rng.random() < 0.5 else L.upper())
        return q, '{N}.'.format(N=ALPHA[i + 1].upper()), 'letters'
    if k == 'before' and i > 0:
        q = rng.choice(['what letter comes before {l}', 'which letter is before {l}',
                        'what comes before {l} in the alphabet']).format(l=L if rng.random() < 0.5 else L.upper())
        return q, '{N}.'.format(N=ALPHA[i - 1].upper()), 'letters'
    q = rng.choice(['what number is the letter {l} in the alphabet', 'where is {l} in the alphabet',
                    'which letter of the alphabet is {l}', 'what is the {n} letter of the alphabet'])
    if '{n}' in q:
        return q.format(n=ORD[i + 1]), '{L}.'.format(L=L.upper()), 'letters'
    return q.format(l=L.upper()), '{L} is letter number {n}.'.format(L=L.upper(), n=i + 1), 'letters'


ORD = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
       'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth',
       'eighteenth', 'nineteenth', 'twentieth', 'twenty-first', 'twenty-second', 'twenty-third',
       'twenty-fourth', 'twenty-fifth', 'twenty-sixth']


# ---------------------------------------------------------------- don't know
SYL = ['ka', 'lo', 'mi', 'zu', 'ber', 'tan', 'vel', 'or', 'quo', 'shi', 'dra', 'pex', 'nim', 'gol', 'fra', 'wen',
       'tri', 'xa', 'bul', 'sno', 'ri', 'ya', 'mox', 'ple']
REAL = set(x.lower() for x in COUNTRIES) | set(WORDS) | set(x.lower() for x in CAP_REV)


ENDS = ['ia', 'land', 'stan', 'ovia', 'ania', 'istan', 'onia', 'ica', 'esia', 'burg', 'ar', 'o']


def fake(rng, country=False):
    while True:
        w = ''.join(rng.choice(SYL) for _ in range(rng.randint(2, 3) - (1 if country else 0)))
        if country and rng.random() < 0.7:
            w += rng.choice(ENDS)
        if w not in REAL and 4 <= len(w) <= 11:
            return w


def g_unknown(rng, test):
    k = rng.choice(['country', 'country', 'animal', 'word', 'person', 'gibberish', 'gibberish'])
    w = fake(rng, country=(k == 'country'))
    if k == 'country':
        q = rng.choice(['what is the capital of {W}', 'which continent is {W} in', 'what language is spoken in {W}',
                        'what is the currency of {W}', 'where is {W}']).format(W=cap(w) if rng.random() < 0.5 else w)
        return q, rng.choice(['I do not know a country called {W}.', 'I have never heard of {W}.']).format(W=cap(w)), 'unknown'
    if k == 'animal':
        q = rng.choice(['how many legs does a {w} have', 'what does a {w} eat', 'what is a baby {w} called',
                        'what sound does a {w} make', 'where does a {w} live']).format(w=w)
        return q, rng.choice(['I do not know an animal called a {w}.', 'I have never heard of a {w}.']).format(w=w), 'unknown'
    if k == 'word':
        q = rng.choice(['what is the opposite of {w}', 'what is the plural of {w}', 'what does {w} mean',
                        'what is a {w}']).format(w=w)
        return q, 'I do not know the word {w}.'.format(w=w), 'unknown'
    if k == 'person':
        q = rng.choice(['who is {W} {V}', 'who was {W} {V}', 'what did {W} {V} invent']).format(
            W=cap(w), V=cap(fake(rng)))
        return q, rng.choice(['I do not know who that is.', 'Sorry, I do not know that person.']), 'unknown'
    n = rng.randint(3, 14)
    q = ''.join(rng.choice(string.ascii_lowercase + '  ;,.') for _ in range(n)).strip() or 'xq'
    return q, rng.choice(['Sorry, I did not understand. Please ask a question.', 'I do not understand. Try again?',
                          'Hmm? Please ask me a question.']), 'unknown'


def g_borders(rng, test):
    pool = [c for c in (YN_COUNTRIES_TEST if test else YN_COUNTRIES_TRAIN) if c in BORDERS]
    s = rng.choice(pool)
    nb = BORDERS[s]
    k = rng.random()
    if k < 0.45:
        q = rng.choice(['which countries border {s}', 'what are the neighbours of {s}', 'which countries are next to {s}',
                        'what countries does {s} border', 'name the neighbours of {s}', 'neighbours of {s}',
                        'what countries are around {s}', 'what are the neighbors of {s}']).format(s=s)
        if not nb:
            return q, rng.choice(['None. {s} has no land borders.', '{s} has no land neighbours.']).format(s=s), 'geography'
        if len(nb) <= 4:
            return q, list_and(nb) + '.', 'geography'
        return q, ', '.join(nb[:3]) + ' and %s others.' % words(len(nb) - 3), 'geography'
    if k < 0.6:
        q = rng.choice(['how many countries border {s}', 'how many neighbours does {s} have',
                        'how many countries are next to {s}']).format(s=s)
        n = len(nb)
        return q, ('{s} has no land neighbours.' if n == 0 else
                   rng.choice(['{N}.', '{s} borders {n} countries.' if n > 1 else '{s} borders one country.'])
                   ).format(s=s, n=words(n), N=cap(words(n))), 'geography'
    yes = nb and rng.random() < 0.5
    if yes:
        t = rng.choice(nb)
    else:
        same = [c for c in COUNTRIES if D['continent'].get(c) == D['continent'].get(s) and c != s and c not in nb]
        t = rng.choice(same or COUNTRIES)
    q = rng.choice(['does {s} border {t}', 'is {s} next to {t}', 'are {s} and {t} neighbours',
                    'does {s} share a border with {t}', 'is {t} a neighbour of {s}']).format(s=s, t=t)
    return q, ('Yes, {s} borders {t}.' if t in nb else 'No, {s} does not border {t}.').format(s=s, t=t), 'geography'


ROMAN = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
         (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]


def roman(n):
    out = ''
    for v, r in ROMAN:
        while n >= v:
            out += r
            n -= v
    return out


ROMAN_TRAIN, ROMAN_TEST = split(list(range(1, 101)) + list(range(1900, 2031)), 0.08, seed=31)
SQ_TRAIN, SQ_TEST = split(list(range(0, 21)), 0.1, seed=32)


def g_numbers2(rng, test):
    k = rng.choice(['roman', 'roman', 'unroman', 'square'])
    if k == 'roman':
        n = rng.choice(ROMAN_TEST if test else ROMAN_TRAIN)
        q = rng.choice(['what is {n} in roman numerals', 'write {n} in roman numerals', '{n} in roman numerals',
                        'how do you write {n} in roman numerals', 'roman numeral for {n}']).format(n=n)
        return q, rng.choice(['{r}.', '{n} is {r} in Roman numerals.']).format(n=n, r=roman(n)), 'numbers'
    if k == 'unroman':
        n = rng.choice(ROMAN_TEST if test else ROMAN_TRAIN)
        q = rng.choice(['what number is {r}', 'what is the roman numeral {r}', 'what does {r} mean in roman numerals',
                        '{r} in numbers']).format(r=roman(n))
        return q, rng.choice(['{n}.', '{r} is {n}.']).format(n=n, r=roman(n)), 'numbers'
    n = rng.choice(SQ_TEST if test else SQ_TRAIN)
    q = rng.choice(['what is {n} squared', '{n} squared', 'what is {n} times itself', 'what is the square of {n}',
                    'what is the square root of {m}']).format(n=n, m=n * n)
    if 'root' in q:
        return q, rng.choice(['{n}.', 'The square root of {m} is {n}.']).format(n=n, m=n * n), 'numbers'
    return q, rng.choice(['{m}.', '{n} squared is {m}.']).format(n=n, m=n * n), 'numbers'


SYM_REV = {a: s for s, a in REL['symbol']}
ATOM_REV = {a: s for s, a in REL['atomic']}
DOM_REV = {a: ss[0] for a, ss in group('domain').items() if len(ss) == 1}
STATE_REV = {a: s for s, a in REL['state_capital']}
MEANING = collections.defaultdict(list)
for _r in REL:
    if _r.startswith('word_'):
        for _e, _x in REL[_r]:
            if _e not in MEANING[_x.lower()]:
                MEANING[_x.lower()].append(_e)


def g_reverse2(rng, test):
    k = rng.choice(['sym', 'sym', 'atom', 'dom', 'state', 'mean'])
    if k == 'sym':
        a, s = rng.choice(sorted(SYM_REV.items()))
        q = rng.choice(['which element has the symbol {a}', 'what element is {a}', '{a} is the symbol for what',
                        'what does the symbol {a} stand for', 'which element is {a}']).format(a=a)
        return q, rng.choice(['{S}.', '{a} is {s}.', 'It is {s}.']).format(a=a, s=s, S=cap(s)), 'science'
    if k == 'atom':
        a, s = rng.choice(sorted(ATOM_REV.items(), key=lambda x: int(x[0])))
        q = rng.choice(['what is element number {a}', 'which element has atomic number {a}', 'element {a}',
                        'what element has {a} protons']).format(a=a)
        return q, rng.choice(['{S}.', 'Element {a} is {s}.']).format(a=a, s=s, S=cap(s)), 'science'
    if k == 'dom':
        a, s = rng.choice(sorted(DOM_REV.items()))
        q = rng.choice(['which country uses the domain {a}', '{a} is the internet domain of which country',
                        'what country has websites ending in {a}', 'whose domain is {a}']).format(a=a)
        return q, '{s}.'.format(s=s), 'geography'
    if k == 'state':
        a, s = rng.choice(sorted(STATE_REV.items()))
        q = rng.choice(['{a} is the capital of which state', 'which state has the capital {a}',
                        'what state is {a} the capital of']).format(a=a)
        return q, rng.choice(['{s}.', '{a} is the capital of {s}.']).format(a=a, s=s), 'geography'
    x, es = rng.choice(sorted(MEANING.items()))
    q = rng.choice(['what does {x} mean', 'what does {x} mean in english', 'translate {x} into english',
                    'what is {x} in english']).format(x=x)
    return q, rng.choice(['It means {m}.', '{X} means {m}.']).format(x=x, X=cap(x), m=' or '.join(es)), 'words'


# ---------------------------------------------------------------- noise
PREFIX = ['hey tiny, ', 'tiny, ', 'please tell me ', 'can you tell me ', 'i want to know ', 'quick question: ',
          'hello, ', 'hi tiny, ', 'tell me ', 'i wonder ', 'ok, ', 'so ', 'excuse me, ', 'question: ',
          'do you know ', 'hey, ', 'could you tell me ']
SUFFIX = [' please', ' tiny', ', please', ' thanks', '?!', '??', '!', ' :)', ' ?']
KEYS = 'qwertyuiop asdfghjkl zxcvbnm'


def typo(q, rng):
    idx = [i for i, c in enumerate(q) if c.isalpha()]
    if len(idx) < 4:
        return q
    i = rng.choice(idx)
    k = rng.random()
    if k < 0.3 and i + 1 < len(q):
        return q[:i] + q[i + 1] + q[i] + q[i + 2:]          # swap
    if k < 0.55:
        return q[:i] + q[i + 1:]                            # drop
    if k < 0.75:
        return q[:i] + q[i] + q[i:]                         # double
    row = next((r for r in KEYS.split() if q[i].lower() in r), None)
    if row:
        j = row.index(q[i].lower())
        nb = row[max(0, j - 1)] if rng.random() < 0.5 else row[min(len(row) - 1, j + 1)]
        return q[:i] + nb + q[i + 1:]                       # neighbour key
    return q


def style(q, rng, test=False, typos=True):
    if typos and not test and rng.random() < 0.10:
        q = typo(q, rng)
    if rng.random() < 0.18:
        q = rng.choice(PREFIX) + q
    r = rng.random()
    if r < 0.45:
        q = q.lower()
    elif r < 0.75:
        q = cap(q)
    elif r < 0.81:
        q = q.upper()                                       # caps lock
    if rng.random() < 0.5:
        q += '?'
    elif rng.random() < 0.08:
        q += rng.choice(SUFFIX)
    if rng.random() < 0.03:
        q = '  ' + q.replace(' ', '  ', 1)
    return q


# ---------------------------------------------------------------- mixture
GENS = [  # name, function, share of the training data
    ('facts', g_txt, 0.23),
    ('tables', g_rel, 0.31),
    ('landlocked', g_landlocked, 0.015),
    ('borders', g_borders, 0.035),
    ('reverse', g_reverse, 0.05),
    ('reverse2', g_reverse2, 0.03),
    ('yesno', g_yesno, 0.08),
    ('combo', g_combo, 0.04),
    ('numbers', g_numbers, 0.10),
    ('numbers2', g_numbers2, 0.02),
    ('letters', g_letters, 0.06),
    ('unknown', g_unknown, 0.03),
]


def sample(rng, test=False, kind=None):
    while True:
        if kind:
            name, fn = kind, dict((g[0], g[1]) for g in GENS)[kind]
        else:
            name, fn, _ = rng.choices(GENS, weights=[g[2] for g in GENS])[0]
        q, a, theme = fn(rng, test)
        q = style(q, rng, test, typos=name not in ('letters', 'yesno', 'numbers2'))
        if len(q) <= 80 and len(q) + len(a) <= MAXLEN and all(32 <= ord(c) <= 126 for c in q + a):
            return q, a, name


if __name__ == '__main__':
    rng = random.Random(1)
    print('one-off facts:', len(TXT), ' table relations:', {r: len(v) for r, v in REL.items()})
    c = collections.Counter()
    lens = []
    for _ in range(20000):
        q, a, k = sample(rng)
        c[k] += 1
        lens.append(len(q) + len(a) + 2)
    print('mix:', dict(c))
    print('mean length %.1f, max %d' % (sum(lens) / len(lens), max(lens)))
    for name, _, _ in GENS:
        for t in (False, True):
            q, a, _ = sample(rng, t, name)
            print('  %-10s %-5s %-52s -> %s' % (name, 'TEST' if t else '', q, a))
