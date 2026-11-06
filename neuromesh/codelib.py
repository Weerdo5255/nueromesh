"""codelib: the building blocks the toolsmith writes tools from.

Every block is a plain, self-contained function `block(text, params) -> {"output", "say"}` (raising
ValueError when the text is not something it can do). The toolsmith writes a new tool by copying the
block's source, and the helpers it uses, into a tool.py with the parameters it fitted, so the code
of a made tool is exactly code a person can read here. Only the standard library is used, and nothing
here touches files, the network or other processes (the toolsmith checks that in code, too).

Each recipe also carries what the toolsmith needs to build and test it: cues and practice requests
(they teach the coder node which recipe a request needs), and probes - request templates whose right
answer is computed independently of the block, so tests and experiments check the block honestly.
"""
from __future__ import annotations

import base64
import datetime
import re
from dataclasses import dataclass, field

# ============================================================================ helpers
FILLER_WORDS = ("please", "the", "word", "words", "phrase", "text", "string", "sentence", "of", "for", "in", "to",
                "me", "this", "these", "that", "can", "you", "could", "would", "what", "is", "are", "it")


def target(text, params):
    """The part of a request a tool works on: quoted text, else what follows a colon, else what follows
    the last trigger phrase, with filler words trimmed from the front."""
    t = str(text).strip()
    quoted = re.findall(r"[\"“”']([^\"“”']{1,2000})[\"“”']", t)
    if quoted and not re.fullmatch(r"\w+'\w*", quoted[0]):
        return max(quoted, key=len)
    if ":" in t:
        after = t.split(":", 1)[1].strip()
        if after:
            return after
    low = t.lower()
    best = -1
    for trig in sorted(params.get("triggers", []), key=len, reverse=True):
        for m in re.finditer(r"(?<![a-z])" + re.escape(trig.lower()) + r"(?![a-z])", low):
            best = max(best, m.end())
    out = t[best:] if best >= 0 else t
    words = out.strip().split()
    while words and words[0].lower().strip(",.?!") in FILLER_WORDS:
        words.pop(0)
    while words and words[-1].lower().strip(",.?!") in ("please", "backwards", "reversed", "for", "me"):
        words.pop()
    return " ".join(words).strip(" ?!.,")


def numbers_in(text):
    """The numbers in a text, in order."""
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(text))]


def show_number(x):
    x = round(x, 6)
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


# ============================================================================= blocks
def reverse_text(text, params):
    """Write a text backwards."""
    t = target(text, params)
    if not t:
        raise ValueError("there is no text to reverse")
    out = t[::-1]
    return {"output": out, "say": f"'{t}' backwards is '{out}'"}


def change_case(text, params):
    """Upper case, lower case or title case."""
    low = str(text).lower()
    mode = params.get("mode") or ("upper" if re.search(r"upper|capital|caps|shout", low) else
                                  "title" if re.search(r"title|capitali[sz]e each|each word", low) else
                                  "lower" if re.search(r"lower|small", low) else "upper")
    t = target(text, params)
    if not t:
        raise ValueError("there is no text to change")
    out = t.upper() if mode == "upper" else t.lower() if mode == "lower" else t.title()
    return {"output": out, "say": f"In {mode} case: {out}"}


def count_letters(text, params):
    """How many times a letter (or the vowels, or the consonants) appears in a text."""
    low = str(text).lower()
    m = re.search(r"how many ([a-z])'?s?\b(?:\s+(?:are|is))?\s+(?:there\s+)?in\s+(.+)", low)
    if m:
        what, t = m.group(1), m.group(2).strip(" ?!.\"'")
    else:
        m = re.search(r"\b(?:the\s+)?letter\s+([a-z])\b", low)
        what = m.group(1) if m else ("vowels" if "vowel" in low else "consonants" if "consonant" in low else None)
        if what is None:
            raise ValueError("which letter should it count?")
        t = target(text, params)
        t = re.sub(r"^(?:the\s+)?(?:letter\s+[a-z]|vowels|consonants)\s+(?:in|of)\s+", "", t, flags=re.I)
    t = t.strip(" ?!.\"'")
    if not t:
        raise ValueError("there is no text to count in")
    if what == "vowels":
        n = sum(ch in "aeiou" for ch in t.lower())
    elif what == "consonants":
        n = sum(ch.isalpha() and ch not in "aeiou" for ch in t.lower())
    else:
        n = t.lower().count(what)
    label = what if what in ("vowels", "consonants") else f"'{what}'"
    return {"output": n, "say": f"There {'is' if n == 1 else 'are'} {n} {label} in '{t}'"}


def palindrome(text, params):
    """Whether a text reads the same backwards (ignoring spaces, case and punctuation)."""
    t = target(text, params)
    t = re.sub(r"\s+(?:a\s+)?palindrome\s*$", "", t, flags=re.I)
    t = re.sub(r"^is\s+", "", t, flags=re.I)
    letters = re.sub(r"[^a-z0-9]", "", t.lower())
    if not letters:
        raise ValueError("there is no text to check")
    yes = letters == letters[::-1]
    return {"output": "yes" if yes else "no", "say": f"{'Yes' if yes else 'No'}, '{t}' is {'' if yes else 'not '}a palindrome"}


def acronym(text, params):
    """The acronym of a phrase: the first letter of each important word."""
    t = target(text, params)
    small = {"of", "and", "the", "a", "an", "for", "in", "on", "to"}
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", t) if w.lower() not in small]
    if not words:
        raise ValueError("there is no phrase to shorten")
    out = "".join(w[0].upper() for w in words)
    return {"output": out, "say": f"The acronym of '{t}' is {out}"}


def sort_words(text, params):
    """Words in alphabetical order (or numbers in ascending order)."""
    t = target(text, params)
    items = [x for x in re.split(r"[,\s]+", t) if x]
    if len(items) < 2:
        raise ValueError("there is nothing to sort")
    if all(re.fullmatch(r"-?\d+(?:\.\d+)?", x) for x in items):
        items = sorted(items, key=float)
    else:
        items = sorted(items, key=str.lower)
    if re.search(r"\b(?:descending|reverse|largest first|z to a)\b", str(text).lower()):
        items.reverse()
    out = ", ".join(items)
    return {"output": out, "say": f"Sorted: {out}"}


def caesar(text, params):
    """A Caesar cipher (ROT13 unless another shift is given); decoding shifts back."""
    low = str(text).lower()
    m = re.search(r"(?:shift|by|key)\s+(-?\d+)", low)
    shift = int(m.group(1)) if m else int(params.get("shift", 13))
    if re.search(r"\b(?:decode|decrypt|decipher|undo)\b", low) and "rot13" not in low:
        shift = -shift
    t = target(text, params)
    t = re.sub(r"^(?:rot\s?13|caesar(?:\s+cipher)?|(?:with\s+)?shift\s+-?\d+|by\s+-?\d+)\s*", "", t, flags=re.I).strip()
    if not t:
        raise ValueError("there is no text to encode")

    def move(ch):
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            return chr((ord(ch) - base + shift) % 26 + base)
        return ch
    out = "".join(move(ch) for ch in t)
    return {"output": out, "say": f"'{t}' shifted by {shift} is '{out}'"}


def base64_codec(text, params):
    """Base64: encode a text, or decode base64 back to text."""
    low = str(text).lower()
    t = target(text, params)
    t = re.sub(r"^(?:base\s?64\s*)", "", t, flags=re.I).strip()
    if not t:
        raise ValueError("there is no text to encode")
    if re.search(r"\b(?:decode|decrypt|from base\s?64)\b", low):
        try:
            out = base64.b64decode(t.encode(), validate=True).decode("utf-8")
        except Exception:
            raise ValueError(f"'{t}' is not valid base64")
        return {"output": out, "say": f"'{t}' decodes to '{out}'"}
    out = base64.b64encode(t.encode("utf-8")).decode("ascii")
    return {"output": out, "say": f"'{t}' in base64 is {out}"}


MORSE = {"a": ".-", "b": "-...", "c": "-.-.", "d": "-..", "e": ".", "f": "..-.", "g": "--.", "h": "....", "i": "..",
         "j": ".---", "k": "-.-", "l": ".-..", "m": "--", "n": "-.", "o": "---", "p": ".--.", "q": "--.-", "r": ".-.",
         "s": "...", "t": "-", "u": "..-", "v": "...-", "w": ".--", "x": "-..-", "y": "-.--", "z": "--..",
         "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....", "6": "-....",
         "7": "--...", "8": "---..", "9": "----."}


def morse_codec(text, params):
    """Morse code: text to dots and dashes, or back."""
    t = target(text, params)
    t = re.sub(r"^(?:in\s+)?morse(?:\s+code)?\s*", "", t, flags=re.I).strip()
    if not t:
        raise ValueError("there is no text to translate")
    if re.fullmatch(r"[.\-\s/]+", t):
        back = {v: k for k, v in MORSE.items()}
        words = [w for w in re.split(r"\s*/\s*|\s{3,}", t.strip()) if w]
        out = " ".join("".join(back.get(c, "?") for c in w.split()) for w in words)
        return {"output": out, "say": f"'{t}' is '{out}'"}
    out = " / ".join(" ".join(MORSE[c] for c in w if c in MORSE) for w in t.lower().split())
    return {"output": out, "say": f"'{t}' in Morse code is {out}"}


ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
         (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def roman_numerals(text, params):
    """Roman numerals: a number to numerals, or numerals to a number."""
    m = re.search(r"\b(\d{1,4})\b", str(text))
    if m:
        n = int(m.group(1))
        if not 0 < n < 4000:
            raise ValueError("Roman numerals run from 1 to 3999")
        out, rest = "", n
        for value, sym in ROMAN:
            while rest >= value:
                out += sym
                rest -= value
        return {"output": out, "say": f"{n} in Roman numerals is {out}"}
    m = re.search(r"\b([MDCLXVI]{1,15})\b", str(text))
    if not m:
        raise ValueError("there is no number or Roman numeral in that")
    s, total, i = m.group(1), 0, 0
    for value, sym in ROMAN:
        while s.startswith(sym, i):
            total += value
            i += len(sym)
    if i != len(s):
        raise ValueError(f"'{s}' is not a valid Roman numeral")
    return {"output": total, "say": f"{s} is {total}"}


ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def say_number(n):
    """An integer in English words."""
    if n < 0:
        return "minus " + say_number(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" and " + say_number(n % 100) if n % 100 else "")
    for size, name in ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            rest = n % size
            sep = "" if not rest else (" and " if rest < 100 else ", ")
            return say_number(n // size) + " " + name + (sep + say_number(rest) if rest else "")
    return str(n)


def number_words(text, params):
    """A number written out in words."""
    m = re.search(r"-?\d[\d,]*", str(text))
    if not m:
        raise ValueError("there is no number to write out")
    n = int(m.group(0).replace(",", ""))
    if abs(n) >= 10 ** 15:
        raise ValueError("numbers up to the trillions")
    out = say_number(n)
    return {"output": out, "say": f"{m.group(0)} in words is {out}"}


MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def dates_in(text):
    """The dates in a text: 2024-03-01, 1 March 2024, March 1, 2024, 03/01/2024 (month first)."""
    t = str(text).lower()
    found = []
    pat = (r"(\d{4})-(\d{1,2})-(\d{1,2})|(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})|(" + "|".join(MONTHS)
           + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})|(\d{1,2})/(\d{1,2})/(\d{4})")
    for m in re.finditer(pat, t):
        g = m.groups()
        if g[0]:
            y, mo, d = int(g[0]), int(g[1]), int(g[2])
        elif g[3]:
            y, mo, d = int(g[5]), MONTHS[g[4]], int(g[3])
        elif g[6]:
            y, mo, d = int(g[8]), MONTHS[g[6]], int(g[7])
        else:
            y, mo, d = int(g[11]), int(g[9]), int(g[10])
        try:
            found.append(datetime.date(y, mo, d))
        except ValueError:
            raise ValueError(f"{m.group(0)} is not a real date")
    return found


def days_between(text, params):
    """The number of days between two dates (or from today to a date)."""
    ds = dates_in(text)
    if len(ds) == 1 and re.search(r"\b(?:until|till|since|from today|from now|ago)\b", str(text).lower()):
        ds = [datetime.date.today(), ds[0]]
    if len(ds) < 2:
        raise ValueError("it needs two dates")
    n = abs((ds[1] - ds[0]).days)
    return {"output": n, "say": f"There are {n} days between {ds[0].isoformat()} and {ds[1].isoformat()}"}


def weekday(text, params):
    """The day of the week a date falls on."""
    ds = dates_in(text)
    if not ds:
        raise ValueError("there is no date in that")
    name = ds[0].strftime("%A")
    return {"output": name, "say": f"{ds[0].isoformat()} is a {name}"}


PATTERNS = {"email": r"[\w.+-]+@[\w-]+\.[\w.-]+", "url": r"https?://[^\s,]+|www\.[^\s,]+",
            "hashtag": r"#\w+", "mention": r"(?<!\w)@\w+", "phone": r"\+?\d[\d\s().-]{6,}\d",
            "number": r"-?\d+(?:\.\d+)?"}


def extract_items(text, params):
    """Pull every email address, link, hashtag, mention, phone number or number out of a text."""
    low = str(text).lower()
    kind = params.get("kind") or next((k for k, w in (("email", "email"), ("url", "url"), ("url", "link"),
                                                      ("hashtag", "hashtag"), ("mention", "mention"),
                                                      ("phone", "phone"), ("number", "number")) if w in low), None)
    if kind is None:
        raise ValueError("what should it find: emails, links, hashtags, mentions, phone numbers or numbers?")
    t = target(text, params)
    items = [x.rstrip(".,;") for x in re.findall(PATTERNS[kind], t)]
    names = {"email": "email address", "url": "link", "hashtag": "hashtag", "mention": "mention",
             "phone": "phone number", "number": "number"}
    if not items:
        return {"output": "none", "say": f"There are no {names[kind]}s in that"}
    out = ", ".join(items)
    return {"output": out, "say": f"{len(items)} {names[kind]}{'s' if len(items) != 1 else ''}: {out}"}


UNITS = {"km": ("length", 1000.0), "kilometre": ("length", 1000.0), "kilometer": ("length", 1000.0),
         "m": ("length", 1.0), "metre": ("length", 1.0), "meter": ("length", 1.0), "cm": ("length", 0.01),
         "centimetre": ("length", 0.01), "centimeter": ("length", 0.01), "mm": ("length", 0.001),
         "mile": ("length", 1609.344), "yard": ("length", 0.9144), "foot": ("length", 0.3048), "feet": ("length", 0.3048),
         "ft": ("length", 0.3048), "inch": ("length", 0.0254), "inches": ("length", 0.0254), "in": ("length", 0.0254),
         "kg": ("mass", 1.0), "kilogram": ("mass", 1.0), "g": ("mass", 0.001), "gram": ("mass", 0.001),
         "lb": ("mass", 0.45359237), "pound": ("mass", 0.45359237), "oz": ("mass", 0.028349523125),
         "ounce": ("mass", 0.028349523125), "stone": ("mass", 6.35029318),
         "l": ("volume", 1.0), "litre": ("volume", 1.0), "liter": ("volume", 1.0), "ml": ("volume", 0.001),
         "gallon": ("volume", 3.785411784), "pint": ("volume", 0.473176473), "cup": ("volume", 0.2365882365),
         "celsius": ("temperature", None), "c": ("temperature", None), "fahrenheit": ("temperature", None),
         "f": ("temperature", None), "kelvin": ("temperature", None), "k": ("temperature", None)}


def unit_of(word):
    w = word.lower().strip(".,?°")
    if w in UNITS:
        return w
    if w.endswith("es") and w[:-2] in UNITS:
        return w[:-2]
    if w.endswith("s") and w[:-1] in UNITS:
        return w[:-1]
    return None


def convert_units(text, params):
    """Convert a quantity between units of length, mass, volume or temperature."""
    t = str(text).lower().replace("°", " ")
    m = re.search(r"(-?\d[\d,]*(?:\.\d+)?)\s*([a-z]+)\b.*?\b(?:to|in|into)\s+([a-z]+)", t)
    if not m:
        raise ValueError("it needs a number, a unit, and a unit to convert to")
    value = float(m.group(1).replace(",", ""))
    a, b = unit_of(m.group(2)), unit_of(m.group(3))
    learned = params.get("factors", {})
    key = f"{m.group(2)}>{m.group(3)}"
    if key in learned:
        out = value * learned[key]
        return {"output": round(out, 6), "say": f"{show_number(value)} {m.group(2)} is {show_number(out)} {m.group(3)}"}
    if a is None or b is None:
        raise ValueError(f"it does not know the unit '{m.group(2) if a is None else m.group(3)}'")
    ka, fa = UNITS[a]
    kb, fb = UNITS[b]
    if ka != kb:
        raise ValueError(f"{m.group(2)} and {m.group(3)} measure different things")
    if ka == "temperature":
        c = {"c": value, "celsius": value, "f": (value - 32) * 5 / 9, "fahrenheit": (value - 32) * 5 / 9,
             "k": value - 273.15, "kelvin": value - 273.15}[a]
        out = {"c": c, "celsius": c, "f": c * 9 / 5 + 32, "fahrenheit": c * 9 / 5 + 32, "k": c + 273.15,
               "kelvin": c + 273.15}[b]
    else:
        out = value * fa / fb
    return {"output": round(out, 6), "say": f"{show_number(value)} {m.group(2)} is {show_number(out)} {m.group(3)}"}


def word_frequency(text, params):
    """The most common word in a text (ignoring very common words)."""
    t = target(text, params)
    common = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "that", "on", "for", "was", "with"}
    words = [w for w in re.findall(r"[a-z']+", t.lower()) if w not in common]
    if not words:
        raise ValueError("there are no words to count")
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], -words.index(kv[0])))
    return {"output": best[0], "say": f"The most common word is '{best[0]}' ({best[1]} times)"}


def replace_text(text, params):
    """Replace one word or phrase with another in a text."""
    m = re.search(r"replace\s+[\"']?(.+?)[\"']?\s+with\s+[\"']?(.+?)[\"']?\s+in\s*:?\s*(.+)$", str(text), re.I | re.S)
    if not m:
        raise ValueError("say it as: replace X with Y in TEXT")
    old, new, t = m.group(1), m.group(2), m.group(3).strip(" \"'")
    out = re.sub(re.escape(old), new, t, flags=re.I)
    return {"output": out, "say": out}


# ============================================================================ recipes
@dataclass
class Recipe:
    """A building block, and what the toolsmith needs to build, teach and test a tool from it."""
    id: str
    name: str                      # the made tool's name
    block: object                  # the function the tool runs
    helpers: tuple                 # functions and tables it needs, copied in before the block
    title: str
    cues: tuple                    # words and patterns that say a request needs this recipe
    needs: tuple                   # practice requests (they teach the coder node)
    probes: tuple                  # (template, {slot: values}, expect(values) -> answer) checked independently
    keywords: tuple                # routing cues for the made tool's manifest
    triggers: tuple = ()           # phrases after which its text begins
    params: dict = field(default_factory=dict)


def _r(s):
    return s[::-1]


WORDS = ["hello", "neuromesh", "banana", "stressed", "python", "lantern", "racecar", "orbit"]
PHRASES = ["portable network graphics", "as soon as possible", "frequently asked questions", "very important person",
           "national aeronautics and space administration"]
RECIPES: dict[str, Recipe] = {}


def _add(r: Recipe) -> None:
    RECIPES[r.id] = r


_add(Recipe("reverse", "reverser", reverse_text, (target,), "write text backwards",
            ("reverse", "backwards", "backward", "mirror", "flip"),
            ("reverse the word {w}", "write {w} backwards", "spell {w} backwards", "flip the text {w}",
             "what is {w} reversed", "reverse this: {w}"),
            (("reverse the word {w}", {"w": WORDS}, lambda v: _r(v["w"])),
             ("write {w} backwards", {"w": WORDS}, lambda v: _r(v["w"])),
             ("reverse this: {w}", {"w": PHRASES}, lambda v: _r(v["w"]))),
            ("reverse", "backwards", "backward", "re:\\breversed?\\b"),
            ("reverse the word", "reverse", "write", "spell", "flip the text", "flip", "mirror")))
_add(Recipe("case", "case_changer", change_case, (target,), "change text to upper, lower or title case",
            ("uppercase", "upper case", "lowercase", "lower case", "title case", "capitals", "all caps", "capitalize"),
            ("make {w} uppercase", "convert {p} to upper case", "lowercase this: {W}", "put {p} in title case",
             "{w} in all caps"),
            (("make {w} uppercase", {"w": WORDS}, lambda v: v["w"].upper()),
             ("convert {p} to title case", {"p": PHRASES}, lambda v: v["p"].title())),
            ("uppercase", "upper case", "lowercase", "lower case", "title case", "all caps", "re:\\bcapitali[sz]e\\b"),
            ("make", "convert", "put", "lowercase this", "uppercase this")))
_add(Recipe("count_letters", "letter_counter", count_letters, (target,), "count a letter, the vowels or consonants",
            ("how many", "vowels", "consonants", "letter"),
            ("how many r's are in {w}", "count the vowels in {w}", "how many times does the letter e appear in {w}",
             "count the consonants in {p}", "how many a's in {w}"),
            (("how many a's are in {w}", {"w": WORDS}, lambda v: v["w"].count("a")),
             ("count the vowels in {p}", {"p": PHRASES}, lambda v: sum(c in "aeiou" for c in v["p"]))),
            ("vowels", "consonants", "re:how many [a-z]'?s\\b", "re:\\bletter [a-z]\\b"),
            ("count the vowels in", "count the consonants in", "count")))
_add(Recipe("palindrome", "palindrome_checker", palindrome, (target,), "check whether text is a palindrome",
            ("palindrome",),
            ("is {w} a palindrome", "check if {w} is a palindrome", "palindrome test: {p}"),
            (("is {w} a palindrome", {"w": WORDS}, lambda v: "yes" if v["w"] == _r(v["w"]) else "no"),),
            ("palindrome",), ("check if", "is")))
_add(Recipe("acronym", "acronym_maker", acronym, (target,), "make an acronym from a phrase",
            ("acronym", "initials", "abbreviate", "abbreviation"),
            ("what is the acronym for {p}", "make an acronym of {p}", "abbreviate {p}", "initials of {p}"),
            (("what is the acronym for {p}", {"p": PHRASES},
              lambda v: "".join(w[0].upper() for w in v["p"].split() if w not in ("of", "and", "the", "a", "an", "for", "in", "on", "to"))),),
            ("acronym", "abbreviate", "initials"), ("acronym for", "acronym of", "abbreviate", "initials of")))
_add(Recipe("sort", "word_sorter", sort_words, (target,), "sort words alphabetically or numbers in order",
            ("sort", "alphabetical", "alphabetically", "alphabetize", "in order"),
            ("sort these words: {l}", "put {l} in alphabetical order", "alphabetize: {l}", "sort the numbers {n}"),
            (("sort these words: {l}", {"l": ["pear, apple, fig", "zebra lion ant", "delta alpha charlie bravo"]},
              lambda v: ", ".join(sorted(v["l"].replace(",", " ").split(), key=str.lower))),
             ("sort the numbers: {n}", {"n": ["5, 3, 9, 1", "10 2 33 4"]},
              lambda v: ", ".join(sorted(v["n"].replace(",", " ").split(), key=float)))),
            ("re:\\bsort\\b", "alphabetical", "alphabetically", "alphabetize"),
            ("sort these words", "sort the numbers", "sort", "alphabetize", "put")))
_add(Recipe("caesar", "caesar_cipher", caesar, (target,), "encode or decode with a Caesar cipher (ROT13)",
            ("rot13", "caesar", "cipher", "encrypt", "decrypt"),
            ("rot13 {w}", "encode {w} with rot13", "caesar cipher shift 3: {w}", "decrypt rot13: {w}"),
            (("rot13 {w}", {"w": WORDS}, lambda v: "".join(chr((ord(c) - 97 + 13) % 26 + 97) for c in v["w"])),),
            ("rot13", "caesar", "cipher", "encrypt", "decrypt"), ("rot13", "encode", "encrypt", "decrypt", "cipher")))
_add(Recipe("base64", "base64_codec", base64_codec, (target,), "encode text as base64 or decode it",
            ("base64", "base 64"),
            ("base64 encode {w}", "encode {w} in base64", "decode base64: {b}"),
            (("base64 encode {w}", {"w": WORDS}, lambda v: base64.b64encode(v["w"].encode()).decode()),
             ("decode base64: {b}", {"b": ["aGVsbG8=", "bmV1cm9tZXNo"]}, lambda v: base64.b64decode(v["b"]).decode())),
            ("base64", "base 64"), ("base64 encode", "encode", "decode")))
_add(Recipe("morse", "morse_codec", morse_codec, (target, ), "translate text to and from Morse code",
            ("morse",),
            ("{w} in morse code", "translate {w} to morse", "what is {w} in morse"),
            (("translate {w} to morse", {"w": ["sos", "hello", "orbit"]},
              lambda v: " ".join(MORSE[c] for c in v["w"])),),
            ("morse",), ("translate", "what is")))
_add(Recipe("roman", "roman_numerals", roman_numerals, (), "convert numbers to and from Roman numerals",
            ("roman numeral", "roman numerals", "roman"),
            ("{n} in roman numerals", "write {n} as a roman numeral", "what number is {r} in roman numerals"),
            (("{n} in roman numerals", {"n": ["1994", "2024", "49", "3888"]},
              lambda v: {"1994": "MCMXCIV", "2024": "MMXXIV", "49": "XLIX", "3888": "MMMDCCCLXXXVIII"}[v["n"]]),
             ("what number is the roman numeral {r}", {"r": ["XIV", "MMXXVI", "CDXLIV"]},
              lambda v: {"XIV": 14, "MMXXVI": 2026, "CDXLIV": 444}[v["r"]])),
            ("roman numeral", "roman numerals", "re:\\broman\\b")))
_add(Recipe("number_words", "number_speller", number_words, (say_number,), "write numbers out in words",
            ("in words", "spell out", "write out", "written out"),
            ("write {n} in words", "spell out {n}", "how do you write {n} in words"),
            (("write {n} in words", {"n": ["42", "115", "2026", "1000000"]},
              lambda v: {"42": "forty-two", "115": "one hundred and fifteen", "2026": "two thousand and twenty-six",
                         "1000000": "one million"}[v["n"]]),),
            ("in words", "spell out", "write out")))
_add(Recipe("days_between", "day_counter", days_between, (dates_in,), "count the days between two dates",
            ("days between", "days from", "how many days", "days until", "days since"),
            ("how many days between {d1} and {d2}", "days from {d1} to {d2}", "count the days between {d1} and {d2}"),
            (("how many days between {d1} and {d2}", {"d1": ["2024-01-01", "2023-06-15"], "d2": ["2024-03-01", "2025-01-01"]},
              lambda v: abs((datetime.date.fromisoformat(v["d2"]) - datetime.date.fromisoformat(v["d1"])).days)),),
            ("days between", "how many days", "days until", "days since", "days from")))
_add(Recipe("weekday", "weekday_finder", weekday, (dates_in,), "find the day of the week of a date",
            ("day of the week", "what day", "weekday"),
            ("what day of the week was {d}", "which weekday is {d}", "what day was {d}"),
            (("what day of the week was {d}", {"d": ["1969-07-20", "2000-01-01", "2026-11-06"]},
              lambda v: datetime.date.fromisoformat(v["d"]).strftime("%A")),),
            ("day of the week", "weekday", "re:\\bwhat day (?:was|is|will)\\b")))
_add(Recipe("extract", "extractor", extract_items, (target,), "pull emails, links, hashtags or phone numbers out of text",
            ("extract", "find all", "emails", "email addresses", "links", "urls", "hashtags", "phone numbers"),
            ("extract the emails from: {t}", "find all the links in: {t}", "list the hashtags in: {h}"),
            (("extract the emails from: {t}", {"t": ["write to ann@example.com or bob@test.org", "mail me at x.y@mesh.io"]},
              lambda v: ", ".join(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", v["t"]))),
             ("list the hashtags in: {h}", {"h": ["loving #python and #ai today", "#one #two"]},
              lambda v: ", ".join(re.findall(r"#\w+", v["h"])))),
            ("re:\\bextract\\b", "email addresses", "hashtags", "re:\\b(?:find|list) (?:all )?(?:the )?(?:emails|links|urls|hashtags|phone numbers)\\b"),
            ("extract the emails from", "find all the links in", "list the hashtags in", "extract", "find all")))
_add(Recipe("convert", "converter", convert_units, (unit_of, show_number), "convert between units",
            ("convert", "how many", "in miles", "in kilometres", "to fahrenheit", "to celsius", "in pounds", "in kg"),
            ("convert {v} km to miles", "how many pounds is {v} kg", "{v} fahrenheit in celsius", "convert {v} inches to cm"),
            (("convert {v} km to miles", {"v": ["5", "42.195", "100"]}, lambda v: round(float(v["v"]) * 1000 / 1609.344, 6)),
             ("convert {v} celsius to fahrenheit", {"v": ["0", "100", "37"]}, lambda v: round(float(v["v"]) * 9 / 5 + 32, 6))),
            ("re:\\bconvert\\b.*\\bto\\b", "re:\\d\\s*(?:km|miles?|kg|lbs?|pounds?|celsius|fahrenheit|inch(?:es)?|cm|feet|gallons?|litres?)\\b")))
_add(Recipe("word_frequency", "word_frequency", word_frequency, (target,), "find the most common word in a text",
            ("most common word", "most frequent word", "word frequency"),
            ("what is the most common word in: {t}", "most frequent word in: {t}"),
            (("what is the most common word in: {t}", {"t": ["the cat saw the other cat", "red blue red green red"]},
              lambda v: {"the cat saw the other cat": "cat", "red blue red green red": "red"}[v["t"]]),),
            ("most common word", "most frequent word", "word frequency")))
_add(Recipe("replace", "replacer", replace_text, (), "replace a word or phrase in a text",
            ("replace", "substitute", "swap"),
            ("replace {a} with {x} in: {p}", "swap {a} for {x} in: {p}", "substitute {x} for {a} in: {p}"),
            (("replace cat with dog in: {t}", {"t": ["the cat sat", "a cat and another cat"]},
              lambda v: v["t"].replace("cat", "dog")),),
            ("re:\\breplace\\b.+\\bwith\\b",)))

TABLES = {"MORSE": MORSE, "ROMAN": ROMAN, "ONES": ONES, "TENS": TENS, "MONTHS": MONTHS, "PATTERNS": PATTERNS,
          "UNITS": UNITS, "FILLER_WORDS": FILLER_WORDS}
BLOCK_TABLES = {"morse_codec": ("MORSE",), "roman_numerals": ("ROMAN",), "number_words": ("ONES", "TENS"),
                "say_number": ("ONES", "TENS"), "dates_in": ("MONTHS",), "extract_items": ("PATTERNS",),
                "convert_units": ("UNITS",), "unit_of": ("UNITS",), "target": ("FILLER_WORDS",)}
