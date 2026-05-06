"""ISO 639-1 (two-letter) → English language name table.

The right-hand side is the canonical WALS *Language_name* where it differs
from the standard English name (e.g. ``el`` → ``Greek (Modern)``). Entries
whose name doesn't appear in the loaded WALS data are silently ignored at
init time, so this table is safe to extend.
"""
from __future__ import annotations

# Each line: <iso639-1 code> <space> <WALS-friendly English name>
_RAW = """\
ab Abkhazian
aa Afar
af Afrikaans
ak Akan
sq Albanian
am Amharic
ar Arabic
an Aragonese
hy Armenian
as Assamese
av Avaric
ay Aymara
az Azerbaijani
bm Bambara
ba Bashkir
eu Basque
be Belarusian
bn Bengali
bi Bislama
bs Bosnian
br Breton
bg Bulgarian
my Burmese
ca Catalan
ch Chamorro
ce Chechen
ny Chichewa
zh Mandarin
cv Chuvash
kw Cornish
co Corsican
cr Cree
hr Serbian-Croatian
cs Czech
da Danish
dv Divehi
nl Dutch
dz Dzongkha
en English
eo Esperanto
et Estonian
ee Ewe
fo Faroese
fj Fijian
fi Finnish
fr French
ff Fulah
gl Galician
ka Georgian
de German
el Greek (Modern)
gn Guarani
gu Gujarati
ht Haitian
ha Hausa
he Hebrew (Modern)
hi Hindi
hu Hungarian
ia Interlingua
id Indonesian
ga Irish
ig Igbo
ik Inupiaq
is Icelandic
it Italian
iu Inuktitut
ja Japanese
jv Javanese
kl Kalaallisut
kn Kannada
kr Kanuri
ks Kashmiri
kk Kazakh
km Khmer
ki Kikuyu
rw Kinyarwanda
ky Kyrgyz
kv Komi
kg Kongo
ko Korean
ku Kurdish
la Latin
lb Luxembourgish
lg Ganda
li Limburgish
ln Lingala
lo Lao
lt Lithuanian
lu Luba-Katanga
lv Latvian
gv Manx
mk Macedonian
mg Malagasy
ms Malay
ml Malayalam
mt Maltese
mi Maori
mr Marathi
mh Marshallese
mn Mongolian
na Nauru
nv Navajo
nd North Ndebele
ne Nepali
ng Ndonga
nb Norwegian
nn Norwegian
no Norwegian
nr South Ndebele
oc Occitan
oj Ojibwe
om Oromo
or Oriya
os Ossetian
pa Panjabi
fa Persian
pl Polish
ps Pashto
pt Portuguese
qu Quechua
rm Romansh
rn Rundi
ro Romanian
ru Russian
sa Sanskrit
sc Sardinian
sd Sindhi
se Northern Sami
sm Samoan
sg Sango
sr Serbian-Croatian
gd Scottish Gaelic
sn Shona
si Sinhala
sk Slovak
sl Slovenian
so Somali
st Sotho
es Spanish
su Sundanese
sw Swahili
ss Swati
sv Swedish
ta Tamil
te Telugu
tg Tajik
th Thai
ti Tigrinya
bo Tibetan
tk Turkmen
tl Tagalog
tn Tswana
to Tongan
tr Turkish
ts Tsonga
tt Tatar
tw Twi
ty Tahitian
ug Uyghur
uk Ukrainian
ur Urdu
uz Uzbek
ve Venda
vi Vietnamese
cy Welsh
wa Walloon
wo Wolof
xh Xhosa
yi Yiddish
yo Yoruba
za Zhuang
zu Zulu
"""

ISO_639_1: dict[str, str] = dict(line.split(" ", 1) for line in _RAW.splitlines() if line)
