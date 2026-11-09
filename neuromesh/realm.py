"""Fact and fiction: whether something is about the real world or from a story, and saying so (P24).

Everything its memory holds, and everything it says, is marked with a realm:

    fact       about the real world: what is, was, or is claimed to be so
    fiction    from a novel, a film, a game, a series, a myth or legend, a dream, a person's own story, or
               something it or you imagined - true *within* that work, never about the world

How it tells them apart, strongest first:
    1. what the text itself says: "in the novel ...", "in my story ...", "once upon a time", "I dreamed",
       "imagine ...", "in Star Wars, ..." - and "in real life", "really", "historically" for fact
    2. the names it knows belong to a work (Gandalf, Hogwarts, Darth Vader ... and any in [realm.works]) -
       and what its memory already marked: a fact learned as fiction stays fiction
    3. its realm node: an ordinary internal capability (capabilities.toml [realm]), taught by the policy
       above and by people, grown, trained and promoted like any node; once validated it judges what
       nothing in the text settles
    4. otherwise fact - but only as sure as the evidence ("nothing marks it as fiction")

Where it is used:
    memory      every fact a memory node learns carries its realm and the work it belongs to; its own
                statements too (a dream or something imagined is fiction; its opinions are its own)
    answers     an answer from fiction says so ("In The Lord of the Rings, Gandalf is a wizard.") and the
                answer carries its realm, which the web page shows
    the ego     on the way in it notes the realm of what its memory holds; on the way out it marks the
                answer, and in conversation it never passes fiction off as fact: "is Gandalf real?" gets
                "No - Gandalf is a character in The Lord of the Rings."
    short-term  what you tell it in conversation is marked too ("in my story, the dragon Vex ..." is your
                fiction), and kept that way if its ego commits it to long-term memory (conversation.py)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .teachers import Lesson, Teacher

REALM = "realm"
FACT, FICTION = "fact", "fiction"
LABELS = (FACT, FICTION)

# names that belong to a work (lower case). People can add more: [realm.works] in the directives.
WORKS: dict[str, str] = {
    **{n: "The Lord of the Rings" for n in ("gandalf", "frodo", "frodo baggins", "bilbo", "bilbo baggins", "aragorn",
                                             "legolas", "gimli", "samwise", "samwise gamgee", "gollum", "sauron", "saruman",
                                             "mordor", "middle-earth", "middle earth", "the shire", "rivendell", "gondor",
                                             "mount doom", "the one ring", "hobbit", "hobbits", "minas tirith")},
    **{n: "Harry Potter" for n in ("harry potter", "hermione", "hermione granger", "ron weasley", "dumbledore",
                                    "albus dumbledore", "voldemort", "lord voldemort", "hogwarts", "severus snape", "snape",
                                    "hagrid", "draco malfoy", "quidditch", "diagon alley", "gryffindor", "slytherin",
                                    "hufflepuff", "ravenclaw", "muggle", "muggles", "horcrux", "horcruxes")},
    **{n: "Star Wars" for n in ("darth vader", "luke skywalker", "anakin skywalker", "obi-wan kenobi", "obi-wan", "yoda",
                                 "princess leia", "han solo", "chewbacca", "tatooine", "the death star", "death star",
                                 "jedi", "the jedi", "sith", "lightsaber", "lightsabers", "the millennium falcon", "r2-d2",
                                 "c-3po", "palpatine", "boba fett")},
    **{n: "Star Trek" for n in ("spock", "captain kirk", "james t. kirk", "jean-luc picard", "captain picard", "starfleet",
                                 "the uss enterprise", "uss enterprise", "klingon", "klingons", "vulcans",
                                 "warp drive", "the borg")},
    **{n: "A Song of Ice and Fire" for n in ("westeros", "winterfell", "jon snow", "daenerys", "daenerys targaryen",
                                              "tyrion lannister", "cersei lannister", "king's landing", "the iron throne",
                                              "white walkers", "arya stark")},
    **{n: "the Sherlock Holmes stories" for n in ("sherlock holmes", "dr watson", "doctor watson", "professor moriarty", "moriarty",
                                       "221b baker street")},
    **{n: "The Chronicles of Narnia" for n in ("narnia", "aslan", "the white witch")},
    **{n: "Dune" for n in ("arrakis", "paul atreides", "muad'dib", "the spice melange", "sandworm", "sandworms", "fremen")},
    **{n: "The Hunger Games" for n in ("katniss", "katniss everdeen", "panem", "district 12", "peeta mellark")},
    **{n: "Peter Pan" for n in ("peter pan", "neverland", "captain hook", "tinker bell")},
    **{n: "Alice's Adventures in Wonderland" for n in ("the mad hatter", "the cheshire cat", "cheshire cat",
                                                        "the queen of hearts", "the white rabbit")},
    **{n: "The Wizard of Oz" for n in ("the wizard of oz", "dorothy gale", "the tin man", "the yellow brick road",
                                        "emerald city", "the emerald city")},
    **{n: "DC Comics" for n in ("batman", "bruce wayne", "superman", "clark kent", "gotham", "gotham city", "the joker",
                                 "wonder woman", "lex luthor")},
    **{n: "Marvel Comics" for n in ("spider-man", "spiderman", "peter parker", "iron man", "tony stark", "wakanda",
                                     "black panther", "captain america", "the hulk", "thanos", "vibranium", "the avengers")},
    **{n: "Pokémon" for n in ("pikachu", "charizard", "pokémon", "pokemon", "ash ketchum")},
    **{n: "The Legend of Zelda" for n in ("hyrule", "princess zelda", "ganon", "ganondorf")},
    **{n: "Moby-Dick" for n in ("captain ahab", "moby dick", "moby-dick")},
    **{n: "Pride and Prejudice" for n in ("elizabeth bennet", "mr darcy", "mr. darcy", "fitzwilliam darcy")},
    **{n: "Dracula" for n in ("count dracula",)},
    **{n: "Frankenstein" for n in ("frankenstein's monster", "victor frankenstein")},
    **{n: "Winnie-the-Pooh" for n in ("winnie the pooh", "winnie-the-pooh", "eeyore", "the hundred acre wood")},
    **{n: "myth and legend" for n in ("the lost city of atlantis", "unicorn", "unicorns", "the loch ness monster",
                                       "bigfoot", "the yeti", "camelot", "excalibur", "king arthur", "the holy grail",
                                       "medusa", "the minotaur", "the kraken", "leprechauns", "mermaids", "a mermaid",
                                       "a dragon", "valhalla")},
}
WORK_TITLES = sorted({w for w in WORKS.values() if w != "myth and legend"} | {
    "The Hobbit", "The Silmarillion", "Game of Thrones", "The Matrix", "Doctor Who", "Lord of the Rings", "Hamlet", "Macbeth",
    "Romeo and Juliet", "War and Peace", "1984", "Nineteen Eighty-Four", "Brave New World", "The Great Gatsby",
    "To Kill a Mockingbird", "Jurassic Park", "The Simpsons", "Breaking Bad", "Minecraft", "Super Mario",
    "Discworld", "The Expanse", "Toy Story", "Finding Nemo", "Shrek", "The Witcher", "Middlemarch",
    "Jane Eyre", "Wuthering Heights", "Les Misérables", "Don Quixote", "The Odyssey", "The Iliad", "Beowulf",
    "Percy Jackson", "Stranger Things", "The Mandalorian", "Mass Effect", "Final Fantasy"},
    key=len, reverse=True)
KINDS_OF_WORK = r"(?:novel|book|story|stories|short story|film|movie|game|video game|series|tv series|show|tv show|episode|" \
                r"play|comic|comics|manga|anime|cartoon|fairy tale|fable|legend|myth|poem|saga|universe|franchise|fiction)"
_FICTION_CUES = [
    (re.compile(r"\b(?i:in|from)\s+(" + "|".join(re.escape(t) for t in WORK_TITLES) + r")(?![\w])"), lambda m: m.group(1)),
    (re.compile(r"\b(?:in|from)\s+(?:my|our|your|his|her|their)\s+(?:own\s+)?(story|novel|book|game|campaign|world|"
                r"fiction|screenplay|script|fanfic|fan fiction|d&d campaign|dungeons and dragons campaign|tale|poem|dream)\b", re.I),
     lambda m: f"your {m.group(1).lower()}"),
    (re.compile(r"\b(?i:in|from)\s+(?i:the)\s+(?i:" + KINDS_OF_WORK + r")\s+['\"“‘]?([A-Z0-9][\w'’:&!.-]*(?:\s+(?:[A-Z0-9][\w'’:&!.-]*|of|the|and|in|a|an|to|on)(?=\s|$|[,.;:!?]))*)"),
     lambda m: m.group(1).strip(" ,.'\"’”").strip()),
    (re.compile(r"\b(?:in|from)\s+(?:the|a|an|this|that)\s+(?:same\s+)?" + KINDS_OF_WORK + r"\b", re.I), lambda m: "a story"),
    (re.compile(r"\bonce upon a time\b|\bhappily ever after\b|\bin a galaxy far,? far away\b", re.I), lambda m: "a story"),
    (re.compile(r"\b(?:i|we)\s+(?:dreamed|dreamt|had a dream)\b|\bin (?:my|a|the) dream\b", re.I), lambda m: "a dream"),
    (re.compile(r"\b(?:imagine|let'?s pretend|pretend that|make[- ]believe|hypothetically,? (?:if|suppose)|"
                r"in an alternate (?:universe|history)|role-?play)\b", re.I), lambda m: "imagination"),
    (re.compile(r"\b(?:fictional|imaginary|made[- ]up|invented)\s+(?:character|person|place|city|country|world|planet|"
                r"creature|story|town|kingdom|animal)\b", re.I), lambda m: "a story"),
    (re.compile(r"\b(?:a|the)\s+character\s+(?:in|from)\b|\b(?:is|was)\s+a\s+(?:fictional|imaginary)\b", re.I), lambda m: "a story"),
]
# about a work rather than in it: who wrote it, when it came out, who played whom - facts about the real world
_META = re.compile(r"\b(?:wrote|written|writer|author|authored|published|publisher|released|release date|directed|director|"
                   r"produced|producer|starred|starring|actor|actress|played by|who played|cast|box office|sold|copies|"
                   r"adapted|adaptation|premiered|aired|sequel|prequel|created|creator|invented|illustrated|translated|"
                   r"award|oscar|budget|filmed|studio|voiced|voice actor|how many (?:books|films|movies|seasons|episodes))\b", re.I)
_FACT_CUES = re.compile(r"\b(?:in real life|in reality|really (?:exist|existed|is|was|are|were|happen|happened)|actually "
                        r"(?:exist|existed|happened)|historically|a true story|in the real world|non-?fiction|"
                        r"documentary)\b", re.I)
REAL_Q = re.compile(r"^\s*(?:but\s+|so\s+|and\s+)?(?:is|are|was|were)\s+(?P<a>.+?)\s+(?:a\s+)?(?:real(?:\s+(?:person|place|thing|"
                    r"people|places|animal|creature|city|country))?|fictional|fiction|made[- ]up|imaginary|a myth|"
                    r"mythical|a legend|legendary)\s*\??\s*$|"
                    r"^\s*(?:did|does|do)\s+(?P<b>.+?)\s+(?:really\s+|actually\s+)?exist\s*\??\s*$", re.I)


@dataclass
class Judgement:
    realm: str
    work: str = ""
    confidence: float = 0.5
    how: str = ""

    def to_dict(self) -> dict:
        return {"realm": self.realm, "work": self.work, "confidence": round(self.confidence, 2), "how": self.how}


def known_work(text: str, extra: dict | None = None) -> tuple[str, str] | None:
    """The first name in text that belongs to a work: (name, work)."""
    low = " " + re.sub(r"[^\w'’.&-]+", " ", str(text).lower()) + " "
    names = {**WORKS, **{k.lower(): v for k, v in (extra or {}).items()}}
    for name in sorted(names, key=len, reverse=True):
        if f" {name} " in low:
            return name, names[name]
    for title in WORK_TITLES:  # a work named outright ("in The Hobbit, ...", "the plot of Dune")
        lead = "" if " " in title else r"(?:\b(?:in|from|of|read|reading|watched|watching|played|playing|about)\s+)"
        if re.search(rf"{lead}(?<![\w]){re.escape(title)}(?![\w])", str(text)):
            return title, title
    return None


def judge(text: str, extra: dict | None = None) -> Judgement | None:
    """What the text itself says (cues), then names it knows. None when nothing settles it."""
    t = " ".join(str(text).split())
    if not t:
        return None
    for rx, work in _FICTION_CUES:
        m = rx.search(t)
        if m:
            w = work(m)
            named = known_work(t, extra)
            if w in ("a story", "") and named:
                w = named[1]
            return Judgement(FICTION, w, 0.9, f"it says so ({m.group(0).strip()[:40]!r})")
    if _FACT_CUES.search(t) and not REAL_Q.match(t):
        return Judgement(FACT, "", 0.8, f"it says so ({_FACT_CUES.search(t).group(0)!r})")
    named = known_work(t, extra)
    if named:
        meta = _META.search(t)
        if meta:  # who wrote it, who played him: a fact about the work, not something within it
            return Judgement(FACT, named[1], 0.75, f"it is about {named[1]} itself ({meta.group(0)!r}), not what happens in it")
        return Judgement(FICTION, named[1], 0.85, f"{named[0]} is from {named[1]}")
    return None


_SENTENCE = re.compile(r"\b(?:is|was|are|were|has|had|have|can|could|will|would|does|did|lives|lived|becomes|became|"
                       r"dies|died|rules|ruled|fights|fought|finds|found|leads|led|carries|carried|wields|wielded|"
                       r"teaches|taught|travels|travelled|traveled|uses|used|owns|owned|loves|loved)\b", re.I)


def frame(answer: str, work: str) -> str:
    """An answer from fiction, said as such ("In The Lord of the Rings, Gandalf is a wizard.")."""
    a = str(answer or "").strip()
    if not a:
        return a
    low = a.lower()
    if (work and work.lower() in low) or re.search(r"\b(?:fiction|fictional|character|novel|story|film|movie|series|"
                                                   r"myth|legend|legendary|mythical|imagin\w*|dream\w*|book|game)\b", low[:200]):
        return a
    where = {"a story": "the story", "a dream": "a dream", "imagination": "what was imagined",
             "myth and legend": "myth and legend"}.get(work, work or "the story")
    if not _SENTENCE.search(a[:300]):  # a fragment ("a wizard who guides the Fellowship"): said, then placed
        return f"{a.rstrip(' .')} (in {where}: fiction, not fact)."
    first, rest = a.split(" ", 1) if " " in a else (a, "")
    if first.lower() in ("the", "a", "an", "it", "he", "she", "they", "there", "this", "that", "its", "his", "her", "their"):
        first = first.lower()
    return f"In {where}, {first}{(' ' + rest) if rest else ''}"


# ------------------------------------------------------------------ the realm node's teacher
_PLACES = [("Paris", "France"), ("Rome", "Italy"), ("Tokyo", "Japan"), ("Canberra", "Australia"), ("Ottawa", "Canada"),
           ("Nairobi", "Kenya"), ("Lima", "Peru"), ("Oslo", "Norway"), ("Cairo", "Egypt"), ("Madrid", "Spain")]
_FACTS = ["{c} is the capital of {k}", "water boils at 100 degrees Celsius at sea level", "the Moon orbits the Earth",
          "Mount Everest is the highest mountain on Earth", "{c} is a city in {k}", "the Pacific is the largest ocean",
          "Ada Lovelace wrote about the Analytical Engine", "the Eiffel Tower is in Paris", "light travels faster than sound",
          "the human heart has four chambers", "honey bees live in colonies", "{k} has a population of millions",
          "the Great Wall is in China", "penguins live in the southern hemisphere", "copper conducts electricity",
          "the Amazon river flows through Brazil", "how tall is the Eiffel Tower", "what is the capital of {k}",
          "when did the Berlin Wall fall", "who painted the Mona Lisa", "in real life, whales are mammals",
          "historically, Rome was founded in the eighth century BC", "photosynthesis turns light into sugar in plants",
          "the train to {c} leaves at nine", "my sister lives in {c}", "I work as a nurse in {c}"]
_FICTIONS = ["{n} lives in {w}", "in {w}, {n} is very brave", "who is {n}", "what is {n}'s sword called",
             "{n} is a character in {w}", "once upon a time a fox ruled the forest", "in my story the dragon Vex sleeps under the moon",
             "I dreamed that the city was made of glass", "imagine a castle floating over the sea",
             "in the film the hero saves the city", "the wizard cast a spell on the castle", "in my novel the captain is called Ines",
             "where does {n} live", "in the game you can fly to the moon", "a fictional country called Ruritania",
             "let's pretend the floor is lava", "in the series the ship travels faster than light",
             "in my D&D campaign the elves worship the sea"]


class RealmTeacher(Teacher):
    """The fact/fiction policy as a teacher, for the realm node only."""
    name = REALM
    specialist = True

    def __init__(self, seed: int | None = None, extra: dict | None = None):
        self.rng = random.Random(seed)
        self.extra = extra or {}

    def status(self) -> dict:
        return {"detail": "specialist: teaches the realm node fact from fiction"}

    async def label(self, spec, text):
        if spec.name != REALM:
            return None
        j = judge(text, self.extra)
        if j is None:
            return None  # nothing settles it: it doesn't guess for the node
        return Lesson(j.realm, j.confidence, self.name, j.how)

    async def generate(self, spec, label, n):
        if spec.name != REALM or label not in LABELS:
            return []
        out: list[str] = []
        names = list(WORKS.items())
        for _ in range(n * 20):
            if label == FACT:
                c, k = self.rng.choice(_PLACES)
                t = self.rng.choice(_FACTS).format(c=c, k=k)
            else:
                n_, w = self.rng.choice(names)
                t = self.rng.choice(_FICTIONS).format(n=n_.title() if not n_.startswith("the ") else n_, w=w)
            if t not in out:
                out.append(t)
            if len(out) >= n:
                break
        return out


class Realm:
    """Its sense of fact and fiction (see the module docstring)."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.realm
        self.counts = {FACT: 0, FICTION: 0}

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def node(self):
        s = self.sys
        if REALM not in s.capabilities:
            return None
        n = s.registry.champion(REALM)
        if n is None or not set(LABELS) <= set(n.labels):
            return None
        if (n.metrics.get("holdout_acc") or 0.0) < s.directives.nodes.serve_min_accuracy:
            return None
        return n

    def judge(self, text: str, teach: bool = True) -> Judgement:
        """Fact or fiction, and how it knows (see the module docstring)."""
        if not self.enabled:
            return Judgement(FACT, "", 0.0, "fact and fiction are off ([realm] enabled = false)")
        j = judge(text, self.pol.works)
        if j is not None:
            if teach and REALM in self.sys.capabilities and len(str(text)) <= 400:
                self.sys.learn(REALM, " ".join(str(text).split()), j.realm, REALM)  # the policy keeps teaching the node
            return j
        node = self.node()
        if node is not None:
            got, conf = node.predict(self.sys.encode(" ".join(str(text).split())))
            if got in LABELS and conf >= 0.75:
                return Judgement(got, "a story" if got == FICTION else "", round(float(conf) * 0.9, 2),
                                 f"its realm node {node.id} ({conf:.0%} sure)")
        return Judgement(FACT, "", 0.5, "nothing marks it as fiction")

    def of_memory(self, text: str):
        """What its memory already holds for this, and as what: (recall, realm, work) or None."""
        mem = self.sys.memory
        if not mem.enabled:
            return None
        try:
            r = mem.recall(text, quiet=True)
        except Exception:
            return None
        if r is None:
            return None
        return r, getattr(r, "realm", "") or FACT, getattr(r, "work", "")

    # ------------------------------------------------------------- in conversation
    def answer(self, text: str) -> tuple[str, str, str, str] | None:
        """'Is Gandalf real?', 'Did Atlantis exist?': (answer, how, realm, work) when it knows, else None (other
        routes answer)."""
        if not self.enabled:
            return None
        m = REAL_Q.match(" ".join(str(text).split()))
        if not m:
            return None
        about = (m.group("a") or m.group("b") or "").strip(" ?.!")
        if not about or len(about) > 80:
            return None
        named = known_work(about, self.pol.works)
        if named:
            w = named[1]
            same = named[0].lower() == w.lower()
            what = "from myth and legend" if w == "myth and legend" else "" if same else f"from {w}"
            within = "those stories" if w == "myth and legend" or same else w
            return (f"No - {_cap(about)} is {what + ': ' if what else ''}fiction, not fact. What I know of it is true only "
                    f"within {within}.", f"{named[0]} is a name from {w}", FICTION, w)
        mem = self.of_memory(f"who is {about}") or self.of_memory(f"what is {about}") or self.of_memory(about)
        if mem is not None:
            r, realm, work = mem
            if realm == FICTION:
                return (f"No - {_cap(about)} is from {work or 'a story'}, as far as I know: fiction, not fact. {frame(r.answer, work)}",
                        f"its memory node {r.node.id} holds it as fiction", FICTION, work)
            return (f"Yes, as far as I know {about} is real: {r.answer}", f"its memory node {r.node.id} holds it as fact",
                    FACT, "")
        j = self.judge(about, teach=False)
        if j.realm == FICTION and j.confidence >= 0.7:
            return (f"I think {about} is fiction{(' - from ' + j.work) if j.work and j.work != 'a story' else ''}, "
                    f"not something in the real world ({j.how}).", j.how, FICTION, j.work)
        return None

    def mark(self, order, res) -> dict | None:
        """On the way out: the answer's realm; an answer from fiction says so (P24)."""
        if not self.enabled or not res.ok:
            return None
        route = res.route or ""
        if route.startswith(("skill:", "tool:")) and res.knowledge not in ("memory", "recalled", "researched"):
            return None
        if route in ("tasks", "objectives", "toolsmith", "scratch", "vision", "ego", "mind", "thinking"):
            return None
        if route.startswith("dream:"):  # its imagination: fiction by its nature (P13)
            res.realm = {"realm": FICTION, "work": "imagination", "confidence": 1.0, "how": "it imagined it"}
            return res.realm
        if res.realm:  # already marked (its memory held it as fiction, or as fact)
            mark = res.realm
        else:
            j = self.judge(f"{order.text} {res.answer or res.output or ''}"[:600])
            mark = j.to_dict()
        res.realm = mark
        self.counts[mark["realm"]] = self.counts.get(mark["realm"], 0) + 1
        if mark["realm"] == FICTION:
            if self.pol.frame_answers and res.answer:
                lead, sep, tail = res.answer.partition("\nSource")  # the answer, not where it came from
                framed = frame(lead, mark.get("work", ""))
                if framed != lead:
                    res.answer = framed + sep + tail
            res.trace.append(f"realm: fiction{(' (' + mark['work'] + ')') if mark.get('work') else ''} - {mark.get('how', '')}; "
                             "it says so, and never passes it off as fact (P24)")
        return mark

    def status(self) -> dict:
        n = self.node()
        return {"enabled": self.enabled, "frame_answers": self.pol.frame_answers, "counts": self.counts,
                "works": len(WORKS) + len(self.pol.works),
                "node": {"id": n.id, "accuracy": n.metrics.get("holdout_acc")} if n is not None else None}


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
