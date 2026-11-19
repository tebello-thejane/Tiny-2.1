#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Extra geography facts from the public mledoze/countries dataset.
Writes facts/geo2.tsv with: language, currency, people (demonym), landlocked.
Usage: ./geo2.py mledoze/countries.json facts/geo.tsv facts/geo2.tsv"""
import json, sys, unicodedata, collections


def ascii_only(s):
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('’', "'").replace('ı', 'i').replace('ø', 'o').replace('đ', 'd')
    return s.encode('ascii', 'ignore').decode('ascii')


# Main everyday language(s), hand-picked, each checked against the dataset's
# official languages below. Two languages are written "X and Y".
MAIN_LANG = {
    'France': 'French', 'Germany': 'German', 'Spain': 'Spanish', 'Italy': 'Italian',
    'Portugal': 'Portuguese', 'Russia': 'Russian', 'China': 'Chinese', 'Japan': 'Japanese',
    'Brazil': 'Portuguese', 'Mexico': 'Spanish', 'Argentina': 'Spanish', 'Egypt': 'Arabic',
    'Greece': 'Greek', 'Turkey': 'Turkish', 'Poland': 'Polish', 'Sweden': 'Swedish',
    'Norway': 'Norwegian', 'Denmark': 'Danish', 'Finland': 'Finnish', 'Estonia': 'Estonian',
    'Latvia': 'Latvian', 'Lithuania': 'Lithuanian', 'Netherlands': 'Dutch', 'Hungary': 'Hungarian',
    'Romania': 'Romanian', 'Bulgaria': 'Bulgarian', 'Ukraine': 'Ukrainian', 'Iceland': 'Icelandic',
    'United Kingdom': 'English', 'United States': 'English', 'Australia': 'English',
    'Austria': 'German', 'Thailand': 'Thai', 'Vietnam': 'Vietnamese', 'South Korea': 'Korean',
    'North Korea': 'Korean', 'Indonesia': 'Indonesian', 'Iran': 'Persian', 'Saudi Arabia': 'Arabic',
    'Israel': 'Hebrew', 'Chile': 'Spanish', 'Colombia': 'Spanish', 'Peru': 'Spanish', 'Cuba': 'Spanish',
    'Czechia': 'Czech', 'Slovakia': 'Slovak', 'Croatia': 'Croatian', 'Serbia': 'Serbian',
    'Mongolia': 'Mongolian', 'Nepal': 'Nepali', 'Morocco': 'Arabic', 'Albania': 'Albanian',
    'Georgia': 'Georgian', 'Armenia': 'Armenian', 'Ireland': 'English', 'New Zealand': 'English',
    'Canada': 'English and French', 'Belgium': 'Dutch and French', 'India': 'Hindi and English',
    'Pakistan': 'Urdu', 'Bangladesh': 'Bengali', 'Malaysia': 'Malay', 'Kenya': 'Swahili and English',
    'Tanzania': 'Swahili', 'Ethiopia': 'Amharic', 'Nigeria': 'English', 'Ghana': 'English',
    'Algeria': 'Arabic', 'Tunisia': 'Arabic', 'Libya': 'Arabic', 'Iraq': 'Arabic', 'Syria': 'Arabic',
    'Jordan': 'Arabic', 'Lebanon': 'Arabic', 'Yemen': 'Arabic', 'Oman': 'Arabic', 'Qatar': 'Arabic',
    'Kuwait': 'Arabic', 'United Arab Emirates': 'Arabic', 'Sudan': 'Arabic', 'Venezuela': 'Spanish',
    'Ecuador': 'Spanish', 'Bolivia': 'Spanish', 'Uruguay': 'Spanish', 'Paraguay': 'Spanish',
    'Costa Rica': 'Spanish', 'Panama': 'Spanish', 'Guatemala': 'Spanish', 'Honduras': 'Spanish',
    'Nicaragua': 'Spanish', 'El Salvador': 'Spanish', 'Dominican Republic': 'Spanish',
    'Jamaica': 'English', 'Belarus': 'Belarusian and Russian', 'Moldova': 'Romanian',
    'Slovenia': 'Slovene', 'North Macedonia': 'Macedonian', 'Malta': 'Maltese and English',
    'Cyprus': 'Greek', 'Kazakhstan': 'Kazakh', 'Uzbekistan': 'Uzbek', 'Kyrgyzstan': 'Kyrgyz',
    'Tajikistan': 'Tajik', 'Turkmenistan': 'Turkmen', 'Azerbaijan': 'Azerbaijani',
    'Sri Lanka': 'Sinhala and Tamil', 'Myanmar': 'Burmese', 'Cambodia': 'Khmer', 'Laos': 'Lao',
    'Philippines': 'Filipino and English', 'Senegal': 'French', 'Mali': 'French',
    'Somalia': 'Somali', 'Madagascar': 'Malagasy and French', 'Bhutan': 'Dzongkha',
    'Maldives': 'Maldivian', 'Afghanistan': 'Pashto and Dari', 'Luxembourg': 'Luxembourgish',
    'Switzerland': 'German, French and Italian', 'Haiti': 'Haitian Creole and French',
}


def main(src, geo_tsv, dst):
    data = json.load(open(src, encoding='utf-8'))
    known_caps = {}
    for line in open(geo_tsv):
        rel, s, a = line.rstrip('\n').split('\t')
        if rel == 'capital':
            known_caps[s] = a
    rows = []
    for c in data:
        if not c.get('unMember'):
            continue
        name = ascii_only(c['name']['common'])
        name = {'Turkiye': 'Turkey'}.get(name, name)
        if name not in known_caps:
            print('NOT IN geo.tsv', name)
            continue
        if name in MAIN_LANG:
            want = MAIN_LANG[name].replace(', ', ' and ').split(' and ')
            langs = ' '.join(c.get('languages', {}).values())
            ok = all(w.split()[0] in langs or (w == 'Slovene' and 'Slovenian' in langs)
                     or (w == 'Maldivian' and 'Dhivehi' in langs)
                     or (w == 'Haitian Creole' and 'Haitian' in langs)
                     or (w == 'Romanian' and 'Moldavian' in langs) for w in want)
            if not ok:
                print('LANG MISMATCH', name, MAIN_LANG[name], langs)
            else:
                rows.append(('language', name, MAIN_LANG[name]))
        cur = list(c.get('currencies', {}).values())
        if len(cur) == 1:
            cn = ascii_only(cur[0]['name'])
            cn = cn[0].lower() + cn[1:] if cn.split()[0] in ('Euro',) else cn
            rows.append(('currency', name, cn))
        dem = c.get('demonyms', {}).get('eng', {}).get('m')
        if dem:
            rows.append(('people', name, ascii_only(dem)))
        rows.append(('landlocked', name, 'yes' if c.get('landlocked') else 'no'))
    with open(dst, 'w') as f:
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print(collections.Counter(r[0] for r in rows), len(rows))


if __name__ == '__main__':
    main(*sys.argv[1:4])
