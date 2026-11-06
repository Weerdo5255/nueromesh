"""mathsolver: the calculator's mathematics, in pure Python (no third-party packages).

  arithmetic     exact fractions and big integers, powers, roots, logarithms, trigonometry (radians,
                 or degrees with ° / "degrees"), factorials, percentages, constants (pi, e, tau, phi)
  algebra        solve equations (linear, quadratic with surds and complex roots, polynomials of any
                 degree by rational roots then numerically, any other equation numerically), systems
                 of equations (exactly when linear, numerically otherwise), inequalities, solve for one
                 letter in terms of others, expand, factor
  calculus       derivatives (symbolic, any order, at a point), integrals (symbolic where its rules
                 reach, definite ones numerically too, also to infinity), limits, sums and products
  number theory  primes, prime factors, divisors, gcd/lcm, binomials, Fibonacci, bases
  statistics     mean, median, mode, variance, standard deviation, range
  matrices       determinant, inverse, transpose, rank, products

When an order names a quantity it does not know ("the speed of light times 3"), it raises NeedToKnow:
the tool then asks a director, and is run again with the answer as context.
"""
from __future__ import annotations

import cmath
import math
import random
import re
from fractions import Fraction
from functools import reduce

MAX_DIGITS = 3000
PHI = (1 + math.sqrt(5)) / 2


class MathError(ValueError):
    pass


class NeedToKnow(Exception):
    """The order names a quantity the calculator does not know; a director may."""

    def __init__(self, phrase: str, ask: str = ""):
        super().__init__(phrase)
        self.phrase = phrase
        self.ask = ask or f"What is {phrase}?"


# ============================================================================ numbers
def _check(v):
    if isinstance(v, Fraction) and (len(str(abs(v.numerator))) > MAX_DIGITS or len(str(v.denominator)) > MAX_DIGITS):
        raise MathError("the number is too large to work with")
    if isinstance(v, float) and math.isnan(v):
        raise MathError("the result is not a number")
    if isinstance(v, complex) and (math.isnan(v.real) or math.isnan(v.imag)):
        raise MathError("the result is not a number")
    return v


def exact(v) -> bool:
    return isinstance(v, Fraction)


def as_int(v):
    """The integer value of v, or None."""
    if isinstance(v, Fraction) and v.denominator == 1:
        return v.numerator
    if isinstance(v, float) and v.is_integer() and abs(v) < 2 ** 63:
        return int(v)
    return None


def real(v):
    if isinstance(v, complex):
        if abs(v.imag) > 1e-12 * max(1.0, abs(v.real)):
            raise MathError("the result is not a real number")
        return v.real
    return v


def snap(x: float) -> float:
    r = round(x)
    return float(r) if abs(x - r) < 1e-10 * max(1.0, abs(x)) else x


def fmt(v, decimals: bool = False) -> str:
    """A number as a person reads it: exact where it can be (1/3), else up to 10 significant digits."""
    if isinstance(v, Fraction):
        if v.denominator == 1:
            s = str(v.numerator)
            if len(s) > 60:
                return f"{s[:12]}...{s[-6:]} ({len(s.lstrip('-'))} digits)"
            return s
        if not decimals and v.denominator <= 10000:
            return f"{v.numerator}/{v.denominator}"
        return fmt(float(v))
    if isinstance(v, complex):
        re_, im = snap(v.real), snap(v.imag)
        if abs(im) < 1e-12:
            return fmt(re_)
        sign = "-" if im < 0 else "+"
        imag = "" if abs(abs(im) - 1) < 1e-12 else fmt(abs(im))
        return (f"{fmt(re_)} {sign} {imag}i" if abs(re_) > 1e-12 else f"{'-' if im < 0 else ''}{imag}i")
    if isinstance(v, float):
        if math.isinf(v):
            return "infinity" if v > 0 else "-infinity"
        x = snap(v)
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return f"{x:.10g}"
    return str(v)


def out(v):
    """A number for the tool's JSON output."""
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return v.numerator if len(str(abs(v.numerator))) <= 60 else str(v.numerator)
        return float(f"{float(v):.12g}")
    if isinstance(v, complex):
        if abs(v.imag) < 1e-12 * max(1.0, abs(v.real)):
            return out(v.real)
        return fmt(v)
    if isinstance(v, float):
        x = snap(v)
        return int(x) if x.is_integer() and abs(x) < 1e15 else float(f"{x:.12g}")
    return v


def snap_known(x: float, tol: float = 1e-7) -> float:
    """x moved onto a well-known value when it is within tol of one (for numerical limits)."""
    cands = [float(round(x)), float(Fraction(x).limit_denominator(64)), math.e, 1 / math.e, math.pi, math.pi / 2,
             math.pi / 4, PHI, math.log(2), math.log(10), math.sqrt(math.pi)] + [math.sqrt(n) for n in range(2, 101)]
    cands += [-c for c in cands]
    best = min(cands, key=lambda c: abs(c - x))
    return best if abs(best - x) <= tol * max(1.0, abs(x)) else x


def nice(x: float) -> str | None:
    """A recognisable exact form of a float: an integer, a small fraction, a multiple of pi, a square root."""
    if isinstance(x, Fraction):
        return fmt(x)
    if not isinstance(x, (int, float)) or math.isinf(x) or math.isnan(x):
        return None
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    for value, name in ((math.e, "e"), (-math.e, "-e"), (PHI, "φ"), (math.sqrt(math.pi), "√π")):
        if abs(x - value) < 1e-10:
            return name
    f = Fraction(x).limit_denominator(64)
    if abs(float(f) - x) < 1e-10:
        return f"{f.numerator}/{f.denominator}"
    k = Fraction(x / math.pi).limit_denominator(12)
    if k and abs(float(k) * math.pi - x) < 1e-10:
        num = "" if abs(k.numerator) == 1 else str(abs(k.numerator))
        s = f"{'-' if k < 0 else ''}{num}π" + (f"/{k.denominator}" if k.denominator != 1 else "")
        return s
    for d in (1, 2, 3, 4, 6):
        n = round((x * d) ** 2)
        if 1 < n < 100000 and abs(math.sqrt(n) / d - abs(x)) < 1e-10 and math.isqrt(n) ** 2 != n:
            k, m = _square_part(n)
            root = f"{'' if k == 1 else k}√{m}"
            return f"{'-' if x < 0 else ''}{root}" + (f"/{d}" if d != 1 else "")
    return None


def show_num(x) -> str:
    """A number with its exact form when it has one: '2', '1/3', '√2 ≈ 1.414213562'."""
    if isinstance(x, Fraction):
        if x.denominator == 1 or x.denominator > 10000:
            return fmt(x)
        return f"{fmt(x)} ≈ {fmt(float(x))}"
    if isinstance(x, complex):
        return fmt(x)
    n = nice(x)
    if n is None or n == fmt(x):
        return fmt(x)
    if re.fullmatch(r"-?\d+", n):
        return n
    return f"{n} ≈ {fmt(x)}"


def _square_part(n: int) -> tuple[int, int]:
    """n = k^2 * m with m square-free."""
    k, m, d = 1, n, 2
    while d * d <= m:
        while m % (d * d) == 0:
            m //= d * d
            k *= d
        d += 1
    return k, m


def number(text: str):
    t = text.replace(",", "")
    if re.fullmatch(r"\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?", t):
        return Fraction(t)
    raise MathError(f"'{text}' is not a number")


# ========================================================================== tokenizer
FUNCS = {"sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh",
         "atanh", "sqrt", "cbrt", "exp", "ln", "log", "log10", "log2", "abs", "floor", "ceil", "round", "sign",
         "factorial", "gcd", "lcm", "min", "max", "choose", "perm", "fib", "root", "atan2"}
ALIASES = {"arcsin": "asin", "arccos": "acos", "arctan": "atan", "sine": "sin", "cosine": "cos", "tangent": "tan",
           "exponential": "exp", "fact": "factorial", "ncr": "choose", "comb": "choose", "binomial": "choose",
           "npr": "perm", "lg": "log10", "ceiling": "ceil", "absolute": "abs", "fibonacci": "fib", "sgn": "sign"}
CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "phi": PHI, "inf": math.inf, "infinity": math.inf}
GREEK = {"alpha", "beta", "gamma", "delta", "theta", "lambda", "mu", "sigma", "omega", "rho", "psi", "chi", "kappa",
         "epsilon", "zeta", "eta", "nu", "xi"}
FILLER = {"the", "a", "an", "of", "is", "are", "was", "what", "whats", "equals", "equal", "to", "value", "please",
          "calculate", "compute", "evaluate", "find", "work", "out", "tell", "me", "can", "you", "could", "how", "much",
          "answer", "result", "give", "determine", "get", "number", "do", "does", "it", "that", "this"}
_TOKEN = re.compile(r"\s*(?:(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)"
                    r"|(?P<name>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>\*\*|<=|>=|==|!=|[-+*/^!(),=|\[\];%<>]))")


def tokenize(text: str) -> list[tuple[str, object]]:
    out_, i = [], 0
    text = text.strip()
    while i < len(text):
        m = _TOKEN.match(text, i)
        if not m or m.end() == i:
            if text[i].isspace():
                i += 1
                continue
            raise MathError(f"it does not understand '{text[i]}'")
        i = m.end()
        if m.group("num"):
            raw = m.group("num")
            if re.fullmatch(r"\d+[eE]", raw):  # "2e" is 2 times e
                out_.append(("num", Fraction(raw[:-1])))
                out_.append(("name", "e"))
            else:
                out_.append(("num", Fraction(raw.replace(",", ""))))
        elif m.group("name"):
            out_.append(("name", m.group("name")))
        else:
            op = m.group("op")
            out_.append(("op", "^" if op == "**" else "==" if op == "==" else op))
    return out_


# ============================================================================= parser
class Parser:
    """Expressions as trees of tuples: ('num', v) ('var', name) ('+'|'-'|'*'|'/'|'^', a, b) ('neg', a)
    ('fn', name, [args]) ('list', [items])."""

    def __init__(self, tokens, variables: bool = True):
        self.t, self.i, self.variables = tokens, 0, variables

    def peek(self, k: int = 0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else ("end", None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, op: str):
        tok = self.take()
        if tok != ("op", op):
            raise MathError(f"expected '{op}'")

    def parse(self):
        node = self.expr()
        if self.peek()[0] != "end":
            raise MathError(f"it could not read past '{self.peek()[1]}'")
        return node

    def expr(self):
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            node = (op, node, self.term())
        return node

    def _starts_primary(self) -> bool:
        kind, val = self.peek()
        return kind == "name" or (kind == "op" and val == "(") or (kind == "op" and val == "[")

    def term(self):
        node = self.unary()
        while True:
            kind, val = self.peek()
            if kind == "op" and val in ("*", "/"):
                self.take()
                node = (val, node, self.unary())
            elif kind == "op" and val == "%" and self._after_percent_is_operand():
                self.take()
                node = ("fn", "mod", [node, self.unary()])
            elif self._starts_primary():  # implicit multiplication: 2x, 3(x+1), x sin(x)
                node = ("*", node, self.power())
            else:
                return node

    def _after_percent_is_operand(self) -> bool:
        kind, val = self.peek(1)
        return kind in ("num", "name") or (kind == "op" and val in ("(", "-"))

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        base = self.postfix()
        if self.peek() == ("op", "^"):
            self.take()
            return ("^", base, self.unary())
        return base

    def postfix(self):
        node = self.primary()
        while True:
            if self.peek() == ("op", "!"):
                self.take()
                node = ("fn", "factorial", [node])
            elif self.peek() == ("op", "%") and not self._after_percent_is_operand():
                self.take()
                node = ("/", node, ("num", Fraction(100)))
            else:
                return node

    def primary(self):
        kind, val = self.take()
        if kind == "num":
            if self.peek()[0] == "name" and self.peek()[1] not in FUNCS and ALIASES.get(self.peek()[1]) is None:
                pass  # "2x": the implicit product is made by term()
            return ("num", val)
        if (kind, val) == ("op", "("):
            items = [self.expr()]
            while self.peek() == ("op", ","):
                self.take()
                items.append(self.expr())
            self.expect(")")
            return items[0] if len(items) == 1 else ("list", items)
        if (kind, val) == ("op", "|"):
            inner = self.expr()
            self.expect("|")
            return ("fn", "abs", [inner])
        if (kind, val) == ("op", "["):
            items = []
            if self.peek() != ("op", "]"):
                items.append(self.expr())
                while self.peek() in (("op", ","), ("op", ";")):
                    self.take()
                    items.append(self.expr())
            self.expect("]")
            return ("list", items)
        if kind == "name":
            name = ALIASES.get(val.lower(), val.lower())
            if name in FUNCS or re.fullmatch(r"log_?\d+", name):
                if self.peek() == ("op", "("):
                    self.take()
                    args = [self.expr()]
                    while self.peek() == ("op", ","):
                        self.take()
                        args.append(self.expr())
                    self.expect(")")
                else:  # sin x, sqrt 16, ln 2x
                    arg = self.power()
                    if arg[0] == "num" and self.peek()[0] == "name" and self.peek()[1].lower() not in FUNCS:
                        arg = ("*", arg, self.power())
                    args = [arg]
                if re.fullmatch(r"log_?\d+", name):
                    return ("fn", "log", args + [("num", Fraction(int(re.sub(r"\D", "", name))))])
                return ("fn", name, args)
            if name in CONSTS:
                return ("num", CONSTS[name])
            if not self.variables:
                raise MathError(f"it does not know '{val}'")
            return ("var", val if len(val) == 1 else name)
        raise MathError(f"it could not read '{val}'")


def parse(text: str, variables: bool = True):
    return Parser(tokenize(text), variables).parse()


def variables_in(node) -> set[str]:
    kind = node[0]
    if kind == "var":
        return {node[1]}
    if kind == "num":
        return set()
    if kind in ("fn", "list"):
        args = node[2] if kind == "fn" else node[1]
        return set().union(*(variables_in(a) for a in args)) if args else set()
    return set().union(*(variables_in(a) for a in node[1:]))


# ========================================================================= evaluation
def _exact_root(x: Fraction, n: int):
    """The exact n-th root of a fraction, or None."""
    if x < 0 and n % 2 == 0:
        return None
    sign = -1 if x < 0 else 1
    a, b = abs(x.numerator), x.denominator
    ra, rb = round(a ** (1 / n)), round(b ** (1 / n))
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            p, q = ra + da, rb + db
            if p >= 0 and q > 0 and p ** n == a and q ** n == b:
                return sign * Fraction(p, q)
    if a < 10 ** 30 and b < 10 ** 30:
        return None
    p, q = _iroot(a, n), _iroot(b, n)
    return sign * Fraction(p, q) if p ** n == a and q ** n == b else None


def _iroot(a: int, n: int) -> int:
    lo, hi = 0, 1 << (a.bit_length() // n + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** n <= a:
            lo = mid
        else:
            hi = mid - 1
    return lo


def power(a, b):
    if exact(b) and b.denominator == 1:
        n = b.numerator
        if exact(a):
            if a == 0 and n < 0:
                raise MathError("division by zero")
            if abs(n) * max(1, len(str(abs(a.numerator))), len(str(a.denominator))) > MAX_DIGITS:
                return _check(float(a) ** n) if abs(float(a)) != 1 else Fraction(int(math.copysign(1, float(a))) ** (n % 2 or 2))
            return _check(a ** n)
        return _check(complex(a) ** n if isinstance(a, complex) else float(a) ** n)
    if exact(a) and exact(b) and a >= 0 and b.denominator <= 12:
        r = _exact_root(a, b.denominator)
        if r is not None:
            return power(r, Fraction(b.numerator))
    fa, fb = (complex(a) if isinstance(a, complex) else float(a)), (complex(b) if isinstance(b, complex) else float(b))
    if isinstance(fa, float) and fa < 0 and isinstance(fb, float):
        if exact(b) and b.denominator % 2 == 1:  # an odd root of a negative number is real
            return -(abs(fa) ** fb) if b.numerator % 2 else abs(fa) ** fb
        return _check(complex(fa) ** fb)
    try:
        return _check(fa ** fb)
    except OverflowError:
        raise MathError("the number is too large to work with")


def factorial(n):
    k = as_int(n)
    if k is None or k < 0:
        g = math.gamma(float(n) + 1)  # the gamma function for non-integers
        return g
    if k > 1500:
        raise MathError("factorials above 1500! are too large to show")
    return Fraction(math.factorial(k))


def _f(x):
    if isinstance(x, complex):
        return x
    return float(x)


def _trig_exact(name: str, x):
    """sin(0) = 0 and friends, so exact results stay exact."""
    if exact(x) and x == 0:
        return {"sin": Fraction(0), "tan": Fraction(0), "asin": Fraction(0), "atan": Fraction(0), "sinh": Fraction(0),
                "tanh": Fraction(0), "cos": Fraction(1), "cosh": Fraction(1), "exp": Fraction(1)}.get(name)
    if exact(x) and x == 1 and name == "ln":
        return Fraction(0)
    return None


def apply(name: str, args: list):
    a = args[0] if args else None
    if name in ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "ln"):
        e = _trig_exact(name, a)
        if e is not None:
            return e
    if name in ("gcd", "lcm", "min", "max"):
        vals = _flatten(args)
        if name in ("min", "max"):
            return (min if name == "min" else max)(vals, key=lambda v: float(real(v)))
        ints = [as_int(v) for v in vals]
        if any(i is None for i in ints):
            raise MathError(f"{name} needs whole numbers")
        return Fraction(reduce(math.gcd, ints) if name == "gcd" else reduce(lambda x, y: abs(x * y) // math.gcd(x, y) if x and y else 0, ints))
    if name == "sqrt":
        if exact(a) and a >= 0:
            r = _exact_root(a, 2)
            if r is not None:
                return r
        return _check(cmath.sqrt(_f(a)) if (isinstance(a, complex) or float(a) < 0) else math.sqrt(float(a)))
    if name == "cbrt":
        return power(a, Fraction(1, 3))
    if name == "root":
        return power(a, 1 / args[1] if not exact(args[1]) else Fraction(1) / args[1])
    if name == "abs":
        return abs(a)
    if name in ("floor", "ceil"):
        r = real(a)
        return Fraction(math.floor(r) if name == "floor" else math.ceil(r))
    if name == "round":
        r = real(a)
        if len(args) > 1:
            return Fraction(round(Fraction(r) if exact(r) else r, as_int(args[1]) or 0))
        return Fraction(round(r))
    if name == "sign":
        r = real(a)
        return Fraction((r > 0) - (r < 0))
    if name == "mod":
        b = args[1]
        if b == 0:
            raise MathError("modulo by zero")
        return a % b if exact(a) and exact(b) else Fraction(float(a) % float(b)) if as_int(a) is not None else float(a) % float(b)
    if name == "factorial":
        return factorial(a)
    if name in ("choose", "perm"):
        n, k = as_int(args[0]), as_int(args[1])
        if n is None or k is None or n < 0 or k < 0:
            raise MathError(f"{name} needs whole numbers")
        return Fraction(math.comb(n, k) if name == "choose" else math.perm(n, k))
    if name == "fib":
        n = as_int(a)
        if n is None or n < 0 or n > 10000:
            raise MathError("Fibonacci numbers from 0 to 10000")
        return Fraction(fibonacci(n))
    if name == "log":
        if len(args) > 1:
            base = args[1]
            if exact(a) and exact(base) and base > 0 and base != 1:
                k = _exact_log(a, base)
                if k is not None:
                    return k
            return _check(_ln(a) / _ln(base))
        return apply("log10", [a])
    if name == "log10":
        if exact(a):
            k = _exact_log(a, Fraction(10))
            if k is not None:
                return k
        return _check(_ln(a) / math.log(10))
    if name == "log2":
        if exact(a):
            k = _exact_log(a, Fraction(2))
            if k is not None:
                return k
        return _check(_ln(a) / math.log(2))
    if name == "ln":
        return _check(_ln(a))
    if name == "atan2":
        return math.atan2(float(real(a)), float(real(args[1])))
    if name in ("sec", "csc", "cot"):
        base = {"sec": math.cos, "csc": math.sin, "cot": math.tan}[name](float(real(a)))
        if abs(base) < 1e-15:
            raise MathError(f"{name} is undefined there")
        return 1 / base
    fn = getattr(cmath if isinstance(a, complex) else math, name, None)
    if fn is None:
        raise MathError(f"it does not know the function {name}")
    try:
        return _check(fn(_f(a)))
    except ValueError:
        if name in ("asin", "acos", "acosh", "atanh"):
            return _check(getattr(cmath, name)(complex(_f(a))))
        raise MathError(f"{name} is undefined there")
    except OverflowError:
        raise MathError("the number is too large to work with")


def _ln(a):
    if isinstance(a, complex) or float(a) < 0:
        return cmath.log(complex(_f(a)))
    if float(a) == 0:
        raise MathError("the logarithm of zero is undefined")
    if exact(a) and (a.numerator > 10 ** 300 or a.denominator > 10 ** 300):
        return math.log(a.numerator) - math.log(a.denominator)
    return math.log(float(a))


def _exact_log(a: Fraction, base: Fraction):
    """log_base(a) when it is a whole number or simple fraction (log2(8) = 3, log10(0.001) = -3)."""
    if a <= 0 or base <= 0 or base == 1:
        return None
    k = math.log(float(a)) / math.log(float(base)) if float(a) > 0 else None
    if k is None:
        return None
    f = Fraction(k).limit_denominator(12)
    if power(base, f) == a:
        return f
    return None


def _flatten(args: list) -> list:
    out_ = []
    for a in args:
        out_.extend(_flatten(a) if isinstance(a, list) else [a])
    return out_


def fibonacci(n: int) -> int:
    def fd(k):
        if k == 0:
            return 0, 1
        a, b = fd(k >> 1)
        c, d = a * (2 * b - a), a * a + b * b
        return (d, c + d) if k & 1 else (c, d)
    return fd(n)[0]


def ev(node, env: dict | None = None):
    """Evaluate exactly where possible (fractions), else in floats or complex numbers."""
    env = env or {}
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "var":
        if node[1] in env:
            return env[node[1]]
        raise MathError(f"it needs a value for {node[1]}")
    if kind == "neg":
        return -ev(node[1], env)
    if kind == "list":
        return [ev(a, env) for a in node[1]]
    if kind == "fn":
        return apply(node[1], [ev(a, env) for a in node[2]])
    a, b = ev(node[1], env), ev(node[2], env)
    if isinstance(a, list) or isinstance(b, list):
        raise MathError("lists can only go into functions such as gcd, max or mean")
    if kind == "+":
        return _check(a + b)
    if kind == "-":
        return _check(a - b)
    if kind == "*":
        return _check(a * b)
    if kind == "/":
        if b == 0:
            raise MathError("division by zero")
        return _check(a / b)
    if kind == "^":
        return power(a, b)
    raise MathError(f"it cannot evaluate {kind}")


def fev(node, env: dict) -> float:
    """Evaluate in floats (for numerical methods); nan where undefined."""
    try:
        v = ev(node, env)
        if isinstance(v, complex):
            return v.real if abs(v.imag) < 1e-12 else float("nan")
        return float(v)
    except (MathError, ZeroDivisionError, OverflowError, ValueError, TypeError):
        return float("nan")


# ========================================================================== printing
PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3, "^": 4}


def show(node) -> str:
    """An expression as a person writes it: polynomials canonically (3x^2 - 2x + 1), the rest plainly."""
    vs = sorted(variables_in(node))
    if vs and len(vs) <= 3:
        p = to_poly(node, vs)
        if p is not None and len(p) <= 12:
            return pshow(p, vs)
    return _show(node)


def _show(node, parent: int = 0) -> str:
    kind = node[0]
    if kind == "num":
        s = nice(node[1]) if isinstance(node[1], float) else None
        s = s or fmt(node[1])
        return f"({s})" if parent >= 2 and (s.startswith("-") or "/" in s) else s
    if kind == "var":
        return node[1]
    if kind == "fn":
        if node[1] == "abs":
            return f"|{_show(node[2][0])}|"
        if node[1] in ("ln", "log10") and node[2][0][0] == "fn" and node[2][0][1] == "abs":
            return f"{node[1]}|{_show(node[2][0][2][0])}|"
        return f"{node[1]}({', '.join(_show(a) for a in node[2])})"
    if kind == "list":
        return "[" + ", ".join(_show(a) for a in node[1]) + "]"
    p = PREC[kind]
    if kind == "neg":
        s = "-" + _show(node[1], 2)  # -x*cos(x), but -(x + 1)
    elif kind == "+":
        right = node[2]
        if right[0] == "neg":
            s = f"{_show(node[1], p)} - {_show(right[1], p + 1)}"
        elif right[0] == "num" and isinstance(right[1], (Fraction, float)) and right[1] < 0:
            s = f"{_show(node[1], p)} - {fmt(-right[1])}"
        else:
            s = f"{_show(node[1], p)} + {_show(right, p)}"
    elif kind == "-":
        s = f"{_show(node[1], p)} - {_show(node[2], p + 1)}"
    elif kind == "*":
        a, b = node[1], node[2]
        left = _show(a, p)
        right = _show(b, p + (b[0] == "/"))
        lead = b
        while lead[0] == "*":
            lead = lead[1]
        if a[0] == "num" and lead[0] in ("var", "fn", "^") and not left.startswith("("):
            s = f"{left}{right}"
        else:
            s = f"{left}*{right}"
    elif kind == "/":
        top = node[1]
        left = ("-" + _show(("num", -top[1]), p)) if top[0] == "num" and not isinstance(top[1], complex) and top[1] < 0 \
            else _show(top, p)
        s = f"{left}/{_show(node[2], p + 1)}"
    else:  # ^
        s = f"{_show(node[1], p + 1)}^{_show(node[2], p)}"
    return f"({s})" if p < parent else s


# ======================================================================== polynomials
def _padd(p, q):
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0) + v
        if r[k] == 0:
            del r[k]
    return r


def _pmul(p, q):
    r: dict = {}
    for k1, v1 in p.items():
        for k2, v2 in q.items():
            k = tuple(a + b for a, b in zip(k1, k2))
            r[k] = r.get(k, 0) + v1 * v2
    return {k: v for k, v in r.items() if v != 0}


def to_poly(node, vs: list):
    """The expression as a polynomial in vs ({exponents: coefficient}), or None if it is not one."""
    n = len(vs)
    zero = (0,) * n
    kind = node[0]
    if kind == "num":
        v = node[1]
        if isinstance(v, complex):
            return None
        return {zero: v} if v != 0 else {}
    if kind == "var":
        if node[1] not in vs:
            return None
        k = [0] * n
        k[vs.index(node[1])] = 1
        return {tuple(k): Fraction(1)}
    if kind in ("+", "-"):
        p, q = to_poly(node[1], vs), to_poly(node[2], vs)
        if p is None or q is None:
            return None
        return _padd(p, q if kind == "+" else {k: -v for k, v in q.items()})
    if kind == "neg":
        p = to_poly(node[1], vs)
        return None if p is None else {k: -v for k, v in p.items()}
    if kind == "*":
        p, q = to_poly(node[1], vs), to_poly(node[2], vs)
        return None if p is None or q is None else _pmul(p, q)
    if kind == "/":
        p, q = to_poly(node[1], vs), to_poly(node[2], vs)
        if p is None or q is None or len(q) != 1 or zero not in q:
            return None
        return {k: v / q[zero] for k, v in p.items()}
    if kind == "^":
        if variables_in(node[2]):
            return None
        e = as_int(ev(node[2]))
        if e is None or e < 0 or e > 40:
            return None
        p = to_poly(node[1], vs)
        if p is None:
            return None
        r = {zero: Fraction(1)}
        for _ in range(e):
            r = _pmul(r, p)
        return r
    if kind == "fn" and not variables_in(node):
        v = ev(node)
        if isinstance(v, complex) or isinstance(v, list):
            return None
        return {zero: v} if v != 0 else {}
    return None


def _coef(c, first: bool, mono: str) -> str:
    neg = c < 0
    a = -c if neg else c
    if mono and a == 1:
        body = ""
    elif exact(a) and a.denominator != 1:
        body = f"({fmt(a)})" if mono else fmt(a)
    else:
        body = nice(float(a)) if isinstance(a, float) and nice(float(a)) and len(nice(float(a))) < 8 else fmt(a)
    sign = ("-" if neg else "") if first else (" - " if neg else " + ")
    return sign + body


def pshow(p: dict, vs: list) -> str:
    """A polynomial as a person writes it, highest degree first."""
    if not p:
        return "0"
    terms = sorted(p.items(), key=lambda kv: (-sum(kv[0]), [-e for e in kv[0]]))
    out_ = ""
    for i, (k, c) in enumerate(terms):
        mono = "".join(v + (f"^{e}" if e > 1 else "") for v, e in zip(vs, k) if e)
        out_ += _coef(c, i == 0, mono) + mono
    return out_


def coeffs(p: dict, i: int = 0) -> list:
    """Single-variable coefficients, constant first."""
    deg = max((k[i] for k in p), default=0)
    c = [Fraction(0)] * (deg + 1)
    for k, v in p.items():
        c[k[i]] += v
    return c


def _horner(c: list, x):
    r = 0
    for a in reversed(c):
        r = r * x + a
    return r


def _deflate(c: list, r) -> list:
    """Divide by (x - r): c is constant first."""
    n = len(c) - 1
    out_ = [0] * n
    acc = c[n]
    for i in range(n - 1, -1, -1):
        out_[i] = acc
        acc = c[i] + acc * r
    return out_


def _divisors(n: int) -> list[int]:
    n = abs(n)
    if n == 0:
        return [0]
    if n > 10 ** 12:
        return [1, n]
    small = [d for d in range(1, math.isqrt(n) + 1) if n % d == 0]
    return sorted(set(small + [n // d for d in small]))


def _integerise(c: list) -> list[int] | None:
    if not all(exact(x) for x in c):
        return None
    lcm = reduce(lambda a, b: a * b // math.gcd(a, b), (x.denominator for x in c), 1)
    ints = [int(x * lcm) for x in c]
    g = reduce(math.gcd, ints)
    return [x // g for x in ints] if g else ints


def rational_roots(c: list) -> tuple[list, list]:
    """Rational roots (with multiplicity) by the rational root theorem, and what is left."""
    ints = _integerise(c)
    if ints is None:
        return [], c
    work = [Fraction(x) for x in ints]
    roots = []
    while len(work) > 1 and work[0] == 0:
        roots.append(Fraction(0))
        work = work[1:]
    changed = True
    while changed and len(work) > 2:
        changed = False
        a0, an = int(work[0]), int(work[-1])
        if abs(a0) > 10 ** 12 or abs(an) > 10 ** 12:
            break
        for q in _divisors(an):
            for p in _divisors(a0):
                for cand in (Fraction(p, q), Fraction(-p, q)):
                    if _horner(work, cand) == 0:
                        roots.append(cand)
                        work = _deflate(work, cand)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    if len(work) == 2:  # linear left
        roots.append(-work[0] / work[1])
        work = [work[1]]
    return roots, work


def durand_kerner(c: list) -> list[complex]:
    """All roots of a polynomial numerically (constant-first coefficients)."""
    c = [complex(float(x.real) if isinstance(x, complex) else float(x)) for x in c]
    while len(c) > 1 and abs(c[-1]) < 1e-300:
        c.pop()
    n = len(c) - 1
    if n < 1:
        return []
    lead = c[-1]
    mon = [x / lead for x in c]
    roots = [(0.4 + 0.9j) ** k for k in range(n)]
    for _ in range(800):
        moved = 0.0
        new = []
        for i, r in enumerate(roots):
            den = 1
            for j, s in enumerate(roots):
                if i != j:
                    den *= (r - s)
            val = _horner(mon, r)
            step = val / den if den != 0 else 0
            new.append(r - step)
            moved = max(moved, abs(step))
        roots = new
        if moved < 1e-15:
            break
    polished = []
    d = [i * mon[i] for i in range(1, len(mon))]
    for r in roots:
        for _ in range(30):
            dv = _horner(d, r)
            if dv == 0:
                break
            step = _horner(mon, r) / dv
            r -= step
            if abs(step) < 1e-16:
                break
        if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)):
            r = complex(r.real, 0.0)
        polished.append(r)
    return polished


def quadratic_exact(a: int, b: int, c: int) -> list[str]:
    """The roots of ax^2 + bx + c with surds or i, as text, when they are not rational."""
    d = b * b - 4 * a * c
    k, m = _square_part(abs(d))
    den = 2 * a
    g = math.gcd(math.gcd(abs(b), k), abs(den))
    b2, k2, den2 = -b // g, k // g, den // g
    if den2 < 0:
        b2, den2 = -b2, -den2
    root = f"{'' if k2 == 1 else k2}√{m}" if m != 1 else str(k2)
    if d > 0:
        head = f"{b2} ± {root}" if b2 else f"±{root}"
        return [f"({head})/{den2}" if den2 != 1 else head]
    re_ = Fraction(-b, den)
    im = Fraction(k, abs(den))
    im_s = ("" if im == 1 else fmt(im)) if m == 1 else (f"{'' if im.numerator == 1 else im.numerator}√{m}"
                                                         + (f"/{im.denominator}" if im.denominator != 1 else ""))
    return [f"{fmt(re_)} ± {im_s}i" if re_ else f"±{im_s}i"]


def solve_poly(c: list) -> dict:
    """Every root of a single-variable polynomial: exact where possible."""
    while len(c) > 1 and c[-1] == 0:
        c = c[:-1]
    deg = len(c) - 1
    if deg == 0:
        return {"identity": c[0] == 0}
    roots, rest = rational_roots(c)
    real_roots: list = [r for r in roots]
    exact_text: list[str] = []
    complex_roots: list = []
    if len(rest) == 3:
        ints = _integerise(rest)
        if ints is not None:
            cc, bb, aa = ints
            exact_text += quadratic_exact(aa, bb, cc)
            disc = bb * bb - 4 * aa * cc
            for r in (cmath.sqrt(disc),):
                for sgn in (1, -1):
                    v = (-bb + sgn * r) / (2 * aa)
                    (real_roots if disc > 0 else complex_roots).append(v.real if disc > 0 else v)
            rest = [rest[-1]]
    if len(rest) > 1:
        for r in durand_kerner(rest):
            if abs(r.imag) < 1e-9:
                real_roots.append(r.real)
            else:
                complex_roots.append(r)
    return {"degree": deg, "real": real_roots, "complex": complex_roots, "exact": exact_text}


def _sort_key(v):
    return float(v.real) if isinstance(v, complex) else float(v)


def factor_poly(c: list, var: str) -> str:
    """A polynomial factored over the rationals: 2(x - 3)(2x + 1)(x^2 + 1)."""
    ints = _integerise(c)
    if ints is None:
        raise MathError("it can only factor polynomials with exact coefficients")
    lead_frac = c[-1] / ints[-1] if ints[-1] else Fraction(1)
    roots, rest = rational_roots([Fraction(x) for x in ints])
    content = lead_frac
    work = [Fraction(x) for x in ints]
    parts: dict[str, int] = {}
    order: list[str] = []
    for r in roots:
        p, q = r.numerator, r.denominator
        lin = [Fraction(-p), Fraction(q)]  # q x - p
        work = _pdiv_exact(work, lin)
        term = (f"{'' if q == 1 else q}{var}" + (f" - {p}" if p > 0 else f" + {-p}" if p < 0 else ""))
        key = f"({term})" if p != 0 else (var if q == 1 else f"{q}{var}")
        if key not in parts:
            order.append(key)
        parts[key] = parts.get(key, 0) + 1
    rest_ints = _integerise(work) or work
    g = Fraction(work[-1]) / rest_ints[-1] if rest_ints[-1] else 1
    content *= g
    rest_poly = {(i,): Fraction(v) for i, v in enumerate(rest_ints) if v != 0}
    out_ = "" if content == 1 else ("-" if content == -1 else (fmt(content) if content.denominator == 1 else f"({fmt(content)})"))
    for key in order:
        out_ += key + (f"^{parts[key]}" if parts[key] > 1 else "")
    if len(rest_ints) > 1:
        out_ += f"({pshow(rest_poly, [var])})"
    elif not order:
        out_ = pshow({(i,): v for i, v in enumerate(c) if v != 0}, [var])
    return out_ or "1"


def _pdiv_exact(num: list, den: list) -> list:
    num = list(num)
    out_ = [Fraction(0)] * (len(num) - len(den) + 1)
    for i in range(len(out_) - 1, -1, -1):
        q = num[i + len(den) - 1] / den[-1]
        out_[i] = q
        for j, d in enumerate(den):
            num[i + j] -= q * d
    return out_


# ======================================================================== calculus
def d(node, v: str):
    """The derivative of an expression with respect to v (symbolic)."""
    kind = node[0]
    if v not in variables_in(node):
        return ("num", Fraction(0))
    if kind == "var":
        return ("num", Fraction(1))
    if kind == "neg":
        return ("neg", d(node[1], v))
    if kind in ("+", "-"):
        return (kind, d(node[1], v), d(node[2], v))
    if kind == "*":
        a, b = node[1], node[2]
        return ("+", ("*", d(a, v), b), ("*", a, d(b, v)))
    if kind == "/":
        a, b = node[1], node[2]
        return ("/", ("-", ("*", d(a, v), b), ("*", a, d(b, v))), ("^", b, ("num", Fraction(2))))
    if kind == "^":
        a, b = node[1], node[2]
        if v not in variables_in(b):
            return ("*", ("*", b, ("^", a, ("-", b, ("num", Fraction(1))))), d(a, v))
        if v not in variables_in(a):
            return ("*", ("*", node, ("fn", "ln", [a])), d(b, v))
        return ("*", node, ("+", ("*", d(b, v), ("fn", "ln", [a])), ("/", ("*", b, d(a, v)), a)))
    if kind == "fn":
        name, args = node[1], node[2]
        u = args[0]
        du = d(u, v)
        one, two = ("num", Fraction(1)), ("num", Fraction(2))
        rules = {
            "sin": ("fn", "cos", [u]), "cos": ("neg", ("fn", "sin", [u])),
            "tan": ("/", one, ("^", ("fn", "cos", [u]), two)), "exp": node,
            "ln": ("/", one, u), "log10": ("/", one, ("*", u, ("fn", "ln", [("num", Fraction(10))]))),
            "log2": ("/", one, ("*", u, ("fn", "ln", [two]))),
            "sqrt": ("/", one, ("*", two, ("fn", "sqrt", [u]))),
            "cbrt": ("/", one, ("*", ("num", Fraction(3)), ("^", ("fn", "cbrt", [u]), two))),
            "asin": ("/", one, ("fn", "sqrt", [("-", one, ("^", u, two))])),
            "acos": ("neg", ("/", one, ("fn", "sqrt", [("-", one, ("^", u, two))]))),
            "atan": ("/", one, ("+", one, ("^", u, two))),
            "sinh": ("fn", "cosh", [u]), "cosh": ("fn", "sinh", [u]),
            "tanh": ("-", one, ("^", ("fn", "tanh", [u]), two)),
            "sec": ("*", ("fn", "sec", [u]), ("fn", "tan", [u])),
            "csc": ("neg", ("*", ("fn", "csc", [u]), ("fn", "cot", [u]))),
            "cot": ("neg", ("^", ("fn", "csc", [u]), two)),
            "abs": ("/", u, ("fn", "abs", [u])),
        }
        if name == "log" and len(args) == 2:
            return d(("/", ("fn", "ln", [u]), ("fn", "ln", [args[1]])), v)
        if name == "log":
            return d(("fn", "log10", [u]), v)
        if name not in rules:
            raise MathError(f"it cannot differentiate {name}")
        return simplify(("*", rules[name], du))
    raise MathError("it cannot differentiate that")


def _num(node):
    return node[1] if node[0] == "num" and not isinstance(node[1], complex) else None


def simplify(node):
    """Fold numbers and drop zeros and ones: enough to show derivatives and integrals plainly."""
    kind = node[0]
    if kind in ("num", "var"):
        return node
    if kind == "list":
        return ("list", [simplify(a) for a in node[1]])
    if kind == "fn":
        args = [simplify(a) for a in node[2]]
        if all(a[0] == "num" for a in args):
            try:
                v = apply(node[1], [a[1] for a in args])
                if exact(v):
                    return ("num", v)
                if isinstance(v, float) and v.is_integer() and abs(v) < 1e12 or \
                        isinstance(v, float) and abs(v - round(v)) < 1e-12:  # ln(e) = 1
                    return ("num", Fraction(round(v)))
            except (MathError, ValueError, ZeroDivisionError):
                pass
        return ("fn", node[1], args)
    if kind == "neg":
        a = simplify(node[1])
        if a[0] == "neg":
            return a[1]
        if _num(a) is not None:
            return ("num", -a[1])
        return ("neg", a)
    a, b = simplify(node[1]), simplify(node[2])
    x, y = _num(a), _num(b)
    if x is not None and y is not None and exact(x) and exact(y):
        try:
            v = ev((kind, a, b))
            if exact(v):
                return ("num", v)
        except MathError:
            pass
    if kind == "+":
        if x == 0:
            return b
        if y == 0:
            return a
        if b[0] == "neg":
            return simplify(("-", a, b[1]))
    if kind == "-":
        if y == 0:
            return a
        if x == 0:
            return simplify(("neg", b))
        if a == b:
            return ("num", Fraction(0))
    if kind == "*":
        if x == 0 or y == 0:
            return ("num", Fraction(0))
        if x == 1:
            return b
        if y == 1:
            return a
        if x == -1:
            return simplify(("neg", b))
        if y is not None and x is None:
            return simplify(("*", b, a))
        if exact(x) and x.numerator == 1 and x.denominator > 1:  # (1/2)*ln|x| -> ln|x|/2
            return simplify(("/", b, ("num", Fraction(x.denominator))))
        if b[0] == "/" and _num(b[1]) == 1:  # a * (1/b) -> a/b
            return simplify(("/", a, b[2]))
        if a[0] == "/" and _num(a[1]) == 1:
            return simplify(("/", b, a[2]))
        if x is not None and b[0] == "/":  # 2 * (x/b) -> 2x/b
            return simplify(("/", ("*", a, b[1]), b[2]))
        if x is None and b[0] == "*" and _num(b[1]) is not None:  # sin(x)*(2x) -> 2(x*sin(x))
            return simplify(("*", b[1], ("*", b[2], a) if a[0] == "fn" else ("*", a, b[2])))
        if a[0] == "fn" and b[0] in ("var", "^"):  # polynomial factors first: x^2*cos(x)
            return ("*", b, a)
        if x is not None and b[0] == "*" and _num(b[1]) is not None:
            return simplify(("*", ("num", x * b[1][1]), b[2]))
        if a[0] == "neg":
            return simplify(("neg", ("*", a[1], b)))
        if b[0] == "neg":
            return simplify(("neg", ("*", a, b[1])))
    if kind == "/":
        if x == 0:
            return ("num", Fraction(0))
        if y == 1:
            return a
        if a == b:
            return ("num", Fraction(1))
    if kind == "^":
        if y == 0:
            return ("num", Fraction(1))
        if y == 1:
            return a
        if x == 1:
            return ("num", Fraction(1))
    return (kind, a, b)


def _linear(node, v: str):
    """(a, b) when node is a*v + b with constant a != 0."""
    p = to_poly(node, [v]) if variables_in(node) <= {v} else None
    if p is None:
        return None
    c = coeffs(p)
    if len(c) != 2 or c[1] == 0:
        return None
    return c[1], c[0]


def integrate(node, v: str):
    """An antiderivative by its rules (polynomials, sums, constant multiples, standard functions of a
    linear argument), or None where its rules do not reach."""
    node = simplify(node)
    vs = variables_in(node)
    if v not in vs:
        return ("*", node, ("var", v))
    others = sorted(vs - {v})
    p = to_poly(node, [v] + others)
    if p is not None:
        r = {}
        for k, c in p.items():
            r[(k[0] + 1,) + k[1:]] = c / (k[0] + 1)
        return poly_node(r, [v] + others)
    kind = node[0]
    if kind in ("+", "-"):
        a, b = integrate(node[1], v), integrate(node[2], v)
        return None if a is None or b is None else (kind, a, b)
    if kind == "neg":
        a = integrate(node[1], v)
        return None if a is None else ("neg", a)
    if kind == "*":
        for c, f in ((node[1], node[2]), (node[2], node[1])):
            if v not in variables_in(c):
                r = integrate(f, v)
                return None if r is None else ("*", c, r)
        for p, f in ((node[1], node[2]), (node[2], node[1])):  # by parts: polynomial times sin, cos or exp
            if to_poly(p, [v]) is not None and variables_in(p) <= {v} and _parts_friendly(f, v):
                return _by_parts(p, f, v)
        return None
    if kind == "/":
        a, b = node[1], node[2]
        if v not in variables_in(b):
            r = integrate(a, v)
            return None if r is None else ("/", r, b)
        lin = _linear(b, v)
        if lin and v not in variables_in(a):
            return ("*", ("/", a, ("num", lin[0])), ("fn", "ln", [("fn", "abs", [b])]))
        return None
    if kind == "^":
        base, e = node[1], node[2]
        if v not in variables_in(e):
            lin = _linear(base, v)
            if lin:
                ev_ = ev(e) if not variables_in(e) else None
                if ev_ is not None and ev_ == -1:
                    return ("/", ("fn", "ln", [("fn", "abs", [base])]), ("num", lin[0]))
                return ("/", ("^", base, ("+", e, ("num", Fraction(1)))), ("*", ("num", lin[0]), ("+", e, ("num", Fraction(1)))))
            return None
        if v not in variables_in(base):
            lin = _linear(e, v)
            if lin:
                if base == ("num", math.e):
                    return ("/", node, ("num", lin[0]))
                return ("/", node, ("*", ("num", lin[0]), ("fn", "ln", [base])))
        return None
    if kind == "fn":
        name, u = node[1], node[2][0]
        lin = _linear(u, v)
        if not lin:
            return None
        a = ("num", lin[0])
        table = {"sin": ("neg", ("fn", "cos", [u])), "cos": ("fn", "sin", [u]), "exp": ("fn", "exp", [u]),
                 "sinh": ("fn", "cosh", [u]), "cosh": ("fn", "sinh", [u]),
                 "tan": ("neg", ("fn", "ln", [("fn", "abs", [("fn", "cos", [u])])])),
                 "sqrt": ("*", ("num", Fraction(2, 3)), ("^", u, ("num", Fraction(3, 2)))),
                 "ln": ("-", ("*", u, ("fn", "ln", [u])), u)}
        if name in table:
            return ("/", table[name], a) if lin[0] != 1 else table[name]
    return None


def _parts_friendly(f, v: str) -> bool:
    if f[0] == "fn" and f[1] in ("sin", "cos", "exp", "sinh", "cosh") and _linear(f[2][0], v):
        return True
    return f[0] == "^" and v not in variables_in(f[1]) and _linear(f[2], v) is not None


def _by_parts(p, f, v: str):
    """The tabular method: p F1 - p' F2 + p'' F3 - ..., while the polynomial's derivatives last."""
    total, sign, F = None, 1, f
    for _ in range(12):
        F = integrate(F, v)
        if F is None:
            return None
        term = ("*", p, F) if sign > 0 else ("neg", ("*", p, F))
        total = term if total is None else ("+", total, term)
        p = simplify(d(p, v))
        sign = -sign
        if p == ("num", Fraction(0)):
            return simplify(total)
    return None


def poly_node(p: dict, vs: list):
    node = ("num", Fraction(0))
    for k, c in sorted(p.items(), key=lambda kv: -sum(kv[0])):
        term = ("num", c)
        for var, e in zip(vs, k):
            if e:
                term = ("*", term, ("var", var) if e == 1 else ("^", ("var", var), ("num", Fraction(e))))
        node = ("+", node, term)
    return simplify(node)


def simpson(f, a: float, b: float, eps: float = 1e-10, depth: int = 48) -> float:
    """Adaptive Simpson integration."""
    def s(a, b, fa, fm, fb):
        return (b - a) / 6 * (fa + 4 * fm + fb)

    def rec(a, b, fa, fm, fb, whole, eps, depth):
        m = (a + b) / 2
        lm, rm = (a + m) / 2, (m + b) / 2
        flm, frm = f(lm), f(rm)
        left, right = s(a, m, fa, flm, fm), s(m, b, fm, frm, fb)
        if depth <= 0 or abs(left + right - whole) <= 15 * eps:
            return left + right + (left + right - whole) / 15
        return rec(a, m, fa, flm, fm, left, eps / 2, depth - 1) + rec(m, b, fm, frm, fb, right, eps / 2, depth - 1)
    fa, fb, fm = f(a), f(b), f((a + b) / 2)
    return rec(a, b, fa, fm, fb, s(a, b, fa, fm, fb), eps, depth)


def definite(node, v: str, a: float, b: float) -> float:
    """A definite integral numerically, also over infinite ranges."""
    def f(x):
        y = fev(node, {v: x})
        if math.isnan(y) or math.isinf(y):
            for dx in (1e-9, -1e-9):  # a removable point (sin(x)/x at 0)
                y = fev(node, {v: x + dx})
                if not (math.isnan(y) or math.isinf(y)):
                    return y
            raise MathError("the integral has a singularity in that range")
        return y
    if a > b:
        return -definite(node, v, b, a)
    if math.isinf(a) and math.isinf(b):
        return definite(node, v, -math.inf, 0) + definite(node, v, 0, math.inf)
    if math.isinf(b):  # x = a + t/(1-t)
        g = lambda t: f(a + t / (1 - t)) / (1 - t) ** 2  # noqa: E731
        return simpson(g, 0.0, 1 - 1e-9)
    if math.isinf(a):
        g = lambda t: f(b - (1 - t) / t) / t ** 2  # noqa: E731
        return simpson(g, 1e-9, 1.0)
    return simpson(f, a, b)


def limit(node, v: str, a: float, side: str = "") -> float:
    """A limit numerically, from both sides (or one), or towards infinity."""
    def seq(direction):
        vals = []
        for k in range(1, 8):
            if math.isinf(a):
                x = math.copysign(10.0 ** (k + 1), a)
            else:
                x = a + direction * 10.0 ** -k
            vals.append(fev(node, {v: x}))
        return vals

    def settle(vals):
        good = [x for x in vals if not math.isnan(x)]
        if len(good) < 4:
            raise MathError("the expression is undefined near that point")
        last = good[-3:]
        if abs(last[-1]) > 1e4 and all(abs(b) > 3 * abs(c) for c, b in zip(last, last[1:])):
            return math.copysign(math.inf, last[-1])  # it grows without bound
        if max(last) - min(last) > 1e-3 * max(1.0, abs(last[-1])):
            raise MathError("the limit does not settle (it may oscillate)")
        # Richardson extrapolation (steps shrink tenfold): removes the leading error term
        rich = [(10 * b - c) / 9 for c, b in zip(good[-4:-1], good[-3:])]
        best = rich[-2] if len(rich) > 1 and abs(rich[-1] - rich[-2]) > 1e-7 * max(1.0, abs(rich[-2])) else rich[-1]
        return snap_known(best, 1e-6) if abs(best) < 1e12 else best
    if math.isinf(a):
        return settle(seq(1))
    left = settle(seq(-1)) if side != "+" else None
    right = settle(seq(1)) if side != "-" else None
    if left is not None and right is not None:
        if (math.isinf(left) or math.isinf(right)) and left == right:
            return left
        if math.isinf(left) or math.isinf(right) or abs(left - right) > 1e-4 * max(1.0, abs(left)):
            raise MathError(f"the limit does not exist: from the left it tends to {fmt(left)}, from the right to {fmt(right)}")
        return (left + right) / 2
    return left if left is not None else right


# ===================================================================== numeric roots
def numeric_roots(node, v: str, lo: float = -100.0, hi: float = 100.0, steps: int = 4000) -> list[float]:
    """Real roots of f(v) = 0 in [lo, hi] (and far out), by sign changes and touching minima."""
    f = lambda x: fev(node, {v: x})  # noqa: E731
    xs = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    xs += [s * 10 ** k for k in (2.5, 3, 4, 5, 6) for s in (-1, 1)]
    xs = sorted(set(xs))
    ys = [f(x) for x in xs]
    roots = []

    def refine(a, b, fa):
        for _ in range(200):
            m = (a + b) / 2
            fm = f(m)
            if math.isnan(fm):
                return None
            if fm == 0 or (b - a) < 1e-15 * max(1.0, abs(m)):
                return m
            if (fm > 0) == (fa > 0):
                a, fa = m, fm
            else:
                b = m
        return (a + b) / 2
    for i in range(len(xs) - 1):
        a, b, fa, fb = xs[i], xs[i + 1], ys[i], ys[i + 1]
        if math.isnan(fa) or math.isnan(fb):
            continue
        if fa == 0:
            roots.append(a)
        elif (fa > 0) != (fb > 0):
            r = refine(a, b, fa)
            if r is not None and abs(f(r)) < 1e-6 * max(1.0, abs(fa), abs(fb)) ** 0.5:
                roots.append(r)
    for i in range(1, len(xs) - 1):  # touching roots: x^2 = 0, sin(x) = 1
        y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
        if any(math.isnan(y) for y in (y0, y1, y2)):
            continue
        if abs(y1) < abs(y0) and abs(y1) <= abs(y2) and (y0 > 0) == (y1 > 0) == (y2 > 0):
            a, b = xs[i - 1], xs[i + 1]
            for _ in range(120):
                m1, m2 = a + (b - a) / 3, b - (b - a) / 3
                if abs(f(m1)) < abs(f(m2)):
                    b = m2
                else:
                    a = m1
            m = (a + b) / 2
            if abs(f(m)) < 1e-10:
                roots.append(m)
    out_: list[float] = []
    for r in sorted(roots):
        r = snap(r) if abs(r - round(r)) < 1e-9 else r
        if not out_ or abs(r - out_[-1]) > 1e-7 * max(1.0, abs(r)):
            out_.append(r)
    return out_


def newton_system(fs: list, vs: list[str]) -> list[dict]:
    """Solutions of a nonlinear system numerically, from a grid of starting points."""
    n = len(vs)
    grid = [-6, -2, -0.5, 0.7, 2.5, 6]
    starts = [[]]
    for _ in range(n):
        starts = [s + [g] for s in starts for g in grid]
    rng = random.Random(1)
    rng.shuffle(starts)
    found: list[list[float]] = []
    for x in starts[:300]:
        x = [float(t) for t in x]
        for _ in range(60):
            env = dict(zip(vs, x))
            F = [fev(f, env) for f in fs]
            if any(math.isnan(t) for t in F):
                break
            if max(abs(t) for t in F) < 1e-12:
                break
            J = []
            for f in fs:
                row = []
                for j in range(n):
                    h = 1e-7 * max(1.0, abs(x[j]))
                    e2 = dict(env)
                    e2[vs[j]] = x[j] + h
                    row.append((fev(f, e2) - fev(f, env)) / h)
                J.append(row)
            step = _solve_float(J, [-t for t in F])
            if step is None:
                break
            x = [a + b for a, b in zip(x, step)]
        env = dict(zip(vs, x))
        F = [fev(f, env) for f in fs]
        if all(not math.isnan(t) and abs(t) < 1e-9 for t in F):
            x = [snap(t) for t in x]
            if not any(max(abs(a - b) for a, b in zip(x, y)) < 1e-6 for y in found):
                found.append(x)
    return [dict(zip(vs, x)) for x in sorted(found)]


def _solve_float(A, b):
    n = len(A)
    M = [list(map(float, row)) + [float(bb)] for row, bb in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-14:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                k = M[r][c] / M[c][c]
                M[r] = [a - k * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def gauss(A: list, b: list):
    """Exact Gaussian elimination: ('unique', solution) | ('infinite', rank) | ('none', None)."""
    rows, cols = len(A), len(A[0])
    M = [list(r) + [bb] for r, bb in zip(A, b)]
    piv_cols, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                k = M[i][c]
                M[i] = [x - k * y for x, y in zip(M[i], M[r])]
        piv_cols.append(c)
        r += 1
        if r == rows:
            break
    for i in range(r, rows):
        if M[i][cols] != 0:
            return "none", None
    if r < cols:
        return "infinite", r
    sol = [Fraction(0)] * cols
    for i, c in enumerate(piv_cols):
        sol[c] = M[i][cols]
    return "unique", sol


# ===================================================================== number theory
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    rng = random.Random(n)
    while True:
        y, c, m = rng.randrange(1, n), rng.randrange(1, n), 128
        g, r, q = 1, 1, 1
        x = ys = y
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g


def factorise(n: int) -> dict[int, int]:
    if n == 0:
        raise MathError("zero has no prime factorisation")
    n = abs(n)
    if n.bit_length() > 100:
        raise MathError("that number is too large to factor here")
    out_: dict[int, int] = {}
    for p in (2, 3, 5):
        while n % p == 0:
            out_[p] = out_.get(p, 0) + 1
            n //= p
    p, inc = 7, [4, 2, 4, 2, 4, 6, 2, 6]
    i = 0
    while p * p <= n and p < 100000:
        while n % p == 0:
            out_[p] = out_.get(p, 0) + 1
            n //= p
        p += inc[i % 8]
        i += 1
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            out_[m] = out_.get(m, 0) + 1
            continue
        f = _rho(m)
        stack += [f, m // f]
    return dict(sorted(out_.items()))


def nth_prime(n: int) -> int:
    if n < 1 or n > 300000:
        raise MathError("it finds the 1st to the 300,000th prime")
    bound = 15 if n < 6 else int(n * (math.log(n) + math.log(math.log(n)))) + 10
    sieve = bytearray([1]) * (bound + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, math.isqrt(bound) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, bound + 1, i)))
    count = 0
    for i, flag in enumerate(sieve):
        if flag:
            count += 1
            if count == n:
                return i
    raise MathError("prime search failed")


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n:,}{suffix}"


# ============================================================================ wording
LEAD = re.compile(r"^\s*(?:(?:please|ok|hey|so|now)\s*,?\s+)*(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
                  r"(?:(?:tell|show)\s+me\s+|help\s+me\s+|i\s+(?:want|need)\s+(?:to\s+)?(?:know\s+)?)?"
                  r"(?:what\s*(?:is|'s|s|are|was)|whats|calculate|compute|evaluate|work\s+out|figure\s+out|find|determine|"
                  r"give\s+me|get)?\s*(?:the\s+(?:value|result|answer)\s+(?:of|for)\s+)?", re.I)
WORDS = [
    (r"=\s*\?+\s*$", ""), (r"\s*[?]+\s*$", ""), (r"\s*=\s*$", ""), (r"(?<=\D)\.\s*$", ""), (r"\bplease\b", ""),
    (r"×|·", "*"), (r"÷", "/"), (r"−|–", "-"), (r"²", "^2"), (r"³", "^3"), (r"√", "sqrt "), (r"π", "pi"),
    (r"∞", "inf"), (r"≤", "<="), (r"≥", ">="), (r"≠", "!="), (r"θ", "theta"),
    (r"(\d(?:[\d,]*\d)?(?:\.\d+)?)\s*(?:%|per\s*cent|percent)\s+of\s+", r"(\1/100)*"),
    (r"\blog(?:arithm)?\s+(?:to\s+)?base\s+(\d+(?:\.\d+)?)\s+of\s+(\(.+?\)|\S+)", r"log(\2, \1)"),
    (r"\bnatural\s+log(?:arithm)?\s+of\b", "ln "), (r"\blog(?:arithm)?\s+of\b", "log "),
    (r"\b(?:the\s+)?square\s+root\s+of\b", "sqrt "), (r"\b(?:the\s+)?cube\s+root\s+of\b", "cbrt "),
    (r"\b(?:the\s+)?factorial\s+of\s+(\(.+?\)|\S+)", r"(\1)!"),
    (r"\b(?:the\s+)?absolute\s+value\s+of\b", "abs "),
    (r"\b(?:the\s+)?(sine|cosine|tangent|sin|cos|tan)\s+of\b", r"\1 "),
    (r"\bto\s+the\s+power\s+of\b|\braised\s+to(?:\s+the\s+power\s+of)?\b|\bto\s+the\s+(?=\d)", "^"),
    (r"\bsquared\b", "^2"), (r"\bcubed\b", "^3"),
    (r"\b(\d+)(?:st|nd|rd|th)\s+power\b", r"\1"),
    (r"\bmultiplied\s+by\b|\btimes\b", "*"), (r"\bdivided\s+by\b|\bover\b", "/"),
    (r"\bplus\b|\badded\s+to\b", "+"), (r"\bminus\b|\bsubtract(?:ed)?\b(?!\s+from)", "-"),
    (r"\bmodulo\b|\bmod\b", "%"), (r"(?<=\d)\s*x\s*(?=\d)", "*"),
    (r"\bthe\s+sum\s+of\s+(\S+)\s+and\s+(\S+)", r"(\1 + \2)"),
    (r"\bthe\s+product\s+of\s+(\S+)\s+and\s+(\S+)", r"(\1 * \2)"),
    (r"\bthe\s+difference\s+between\s+(\S+)\s+and\s+(\S+)", r"(\1 - \2)"),
    (r"\bthe\s+quotient\s+of\s+(\S+)\s+and\s+(\S+)", r"(\1 / \2)"),
    (r"\b(?:the\s+)?remainder\s+(?:of|when)\s+(\S+)\s+(?:is\s+)?divided\s+by\s+(\S+)", r"(\1 % \2)"),
    (r"\b(?:the\s+)?remainder\s+(?:of|when)\s+(\S+)\s+(?:is\s+)?/\s*(\S+)", r"(\1 % \2)"),
    (r"\b(\d+\s*/\s*\d+)\s+of\s+", r"(\1)*"), (r"\bhalf\s+of\b", "0.5*"), (r"\b(?:double|twice)\b", "2*"), (r"\btriple\b", "3*"),
]
DEGREES = (r"(\d)\s*(?:°|degrees?\b|deg\b)", r"\1*pi/180")  # only beside trigonometry


def words_to_maths(text: str) -> str:
    t = text.strip()
    for pat, rep in WORDS:
        t = re.sub(pat, rep, t, flags=re.I)
    if re.search(r"\b(?:sin|cos|tan|sec|csc|cot)\b", t):
        t = re.sub(DEGREES[0], DEGREES[1], t, flags=re.I)
    t = LEAD.sub("", t, count=1)
    return t.strip().rstrip(".").strip()


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def first_number(text: str):
    m = re.search(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(text))
    if not m:
        return None
    return Fraction(m.group(0).replace(",", ""))


class Context:
    """What directors told the calculator: answers to the questions it asked."""

    def __init__(self, entries=()):
        self.entries = [e for e in entries or [] if isinstance(e, dict)]
        self.used: list[str] = []

    def lookup(self, phrase: str):
        key = _norm(re.sub(r"^(?:the|a|an)\s+", "", phrase.strip(), flags=re.I))
        for e in self.entries:
            if key and (key in _norm(e.get("ask", "")) or _norm(e.get("about", "")) == key):
                return e
        return None


def resolve(text: str, ctx: Context, variables: bool = True):
    """Parse an expression; a phrase it does not know is looked up in the context, or asked about."""
    try:
        node = parse(text, variables)
        if not any(_unknown(v) for v in variables_in(node)):
            return node, {}
    except MathError:
        pass
    try:
        toks = tokenize(text)
    except MathError:
        raise MathError("it could not find a calculation in that")
    groups, cur = [], []
    for tok in toks + [("end", None)]:
        if tok[0] == "name" and _unknown(tok[1]):
            cur.append(tok[1])
            continue
        if cur:
            groups.append(cur)
            cur = []
    phrases = []
    for g in groups:
        words = list(g)
        while words and words[0].lower() in FILLER:
            words.pop(0)
        while words and words[-1].lower() in FILLER:
            words.pop()
        if words and any(w.lower() not in FILLER for w in words):
            said = list(g)  # the question keeps its own words: "the number of bones in the human body"
            while said and said[0].lower() in ASK_DROP:
                said.pop(0)
            while said and said[-1].lower() in ASK_DROP | {"the", "a", "an", "of"}:
                said.pop()
            phrases.append((g, " ".join(words), " ".join(said)))
    if not phrases or len(phrases) > 3:
        raise MathError("it could not find a calculation in that")
    rebuilt, i, subs, groups_of = [], 0, {}, {}
    asks = {" ".join(g): said for g, _, said in phrases}
    phrases = [(g, p) for g, p, _ in phrases]
    names = {id(g): f"q{n}" for n, (g, _) in enumerate(phrases)}
    group_iter = iter(phrases)
    current = next(group_iter, None)
    while i < len(toks):
        if current and toks[i][0] == "name" and toks[i][1] == current[0][0] and \
                [t[1] for t in toks[i:i + len(current[0])]] == current[0]:
            rebuilt.append(("name", names[id(current[0])]))
            subs[names[id(current[0])]] = current[1]
            groups_of[names[id(current[0])]] = " ".join(current[0])
            i += len(current[0])
            current = next(group_iter, None)
            continue
        rebuilt.append(toks[i])
        i += 1
    # a phrase counts only when an operator joins it to the calculation ("the speed of light * 3");
    # words merely beside it ("12 * 7 for my homework") are dropped
    joins_left = {"+", "-", "*", "/", "^", "(", ",", "%"}
    joins_right = {"+", "-", "*", "/", "^", ")", ",", "%", "!"}
    kept = []
    for i, tok in enumerate(rebuilt):
        if tok[0] == "name" and tok[1] in subs:
            prev = rebuilt[i - 1] if i else ("start", None)
            nxt = rebuilt[i + 1] if i + 1 < len(rebuilt) else ("end", None)
            left_ok = prev[0] == "start" or (prev[0] == "op" and prev[1] in joins_left)
            right_ok = nxt[0] == "end" or (nxt[0] == "op" and nxt[1] in joins_right)
            if not (left_ok and right_ok) or (prev[0] == "start" and nxt[0] == "end"):
                subs.pop(tok[1])
                continue
        kept.append(tok)
    try:
        node = Parser(kept, True).parse()
    except MathError:
        raise MathError("it could not find a calculation in that")
    if not subs and not any(t[0] == "op" and t[1] in "+-*/^%!" for t in kept) and len(kept) == 1:
        raise MathError("it could not find a calculation in that")
    env = {}
    for name, phrase in subs.items():
        entry = ctx.lookup(phrase)
        if entry is None:
            raise NeedToKnow(phrase, f"What is {asks.get(groups_of.get(name, ''), phrase)}?")
        value = first_number(entry.get("answer") or "")
        if value is None:
            what = entry.get("answer")
            raise MathError(f"it needed a number for {phrase}, and the mesh could not give one"
                            + (f" (it was told '{what[:80]}')" if what else ""))
        env[name] = value
        ctx.used.append(f"{phrase} = {fmt(value)}, from {entry.get('source') or 'a director'}")
    return _substitute(node, env), subs


ASK_DROP = {"what", "whats", "is", "are", "was", "calculate", "compute", "evaluate", "find", "please", "tell", "me",
            "can", "you", "could", "how", "much", "give", "determine", "get", "do", "does", "it", "that", "this", "equals",
            "equal", "to", "value", "answer", "result"}


def _unknown(word: str) -> bool:
    w = word.lower()
    w = ALIASES.get(w, w)
    if w in FUNCS or w in CONSTS or w in GREEK or re.fullmatch(r"log_?\d+", w):
        return False
    if len(w) == 1 or re.fullmatch(r"[a-z]_?\d+", w):
        return False
    return True


def _substitute(node, env: dict):
    kind = node[0]
    if kind == "var" and node[1] in env:
        return ("num", env[node[1]])
    if kind in ("num", "var"):
        return node
    if kind == "fn":
        return ("fn", node[1], [_substitute(a, env) for a in node[2]])
    if kind == "list":
        return ("list", [_substitute(a, env) for a in node[1]])
    return (kind,) + tuple(_substitute(a, env) for a in node[1:])


def _split_vars(text: str) -> str:
    """'2xy' -> '2x*y' when short letter runs are not functions or names it knows."""
    def fix(m):
        w = m.group(0)
        lw = w.lower()
        if lw in FUNCS or lw in CONSTS or lw in GREEK or lw in ALIASES or len(w) > 3 or not w.isalpha():
            return w
        return "*".join(w)
    return re.sub(r"(?<![A-Za-z_])[A-Za-z]{2,3}(?![A-Za-z_(])", fix, text)


# ============================================================================ intents
def _result(output, say: str, kind: str, expression: str = "", **extra) -> dict:
    r = {"output": output, "say": say, "kind": kind}
    if expression:
        r["expression"] = expression
    r.update(extra)
    return r


_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def _numbers(text: str) -> list:
    return [Fraction(x.replace(",", "")) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text.replace(",", " ")) if x]


def _numbers_list(text: str) -> list:
    """Numbers in a list like '3, 5 and 7' or '1 2 3' (commas separate, so 1,000 needs no comma here)."""
    return [Fraction(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]


def number_theory(low: str):
    m = re.search(r"\bis\s+(" + _NUM + r")\s+(?:a\s+)?prime(?:\s+number)?\b", low)
    if m:
        n = as_int(number(m.group(1)))
        if n is None:
            raise MathError("only whole numbers can be prime")
        p = is_prime(n)
        if p:
            return _result("yes", f"Yes, {n:,} is prime.", "prime")
        f = factorise(n) if n > 1 else {}
        why = f": {n:,} = {_factor_text(f)}" if f and n > 1 else ""
        return _result("no", f"No, {n:,} is not prime{why}.", "prime")
    m = re.search(r"\b(?:prime\s+factor(?:s|i[sz]ation)?\s+of|factori[sz]e|factor)\s+(" + _NUM + r")\s*$", low)
    if m:
        n = as_int(number(m.group(1)))
        if n is None:
            raise MathError("only whole numbers have prime factors")
        f = factorise(n)
        text = _factor_text(f)
        return _result(text, f"{n:,} = {text}.", "factor")
    m = re.search(r"\b(?:the\s+)?(\d[\d,]*)(?:st|nd|rd|th)\s+prime(?:\s+number)?\b", low)
    if m:
        n = int(m.group(1).replace(",", ""))
        p = nth_prime(n)
        return _result(p, f"The {ordinal(n)} prime is {p:,}.", "prime")
    m = re.search(r"\b(?:all\s+)?(?:the\s+)?(?:divisors|factors)\s+of\s+(" + _NUM + r")\b", low)
    if m:
        n = as_int(number(m.group(1)))
        if n is None or n == 0:
            raise MathError("divisors of a non-zero whole number")
        ds = _divisors(n) if abs(n) <= 10 ** 12 else None
        if ds is None:
            raise MathError("that number is too large to list its divisors")
        return _result(", ".join(map(str, ds)), f"The divisors of {n:,} are {', '.join(map(str, ds))} ({len(ds)} of them).",
                       "divisors")
    m = re.search(r"\b(gcd|gcf|hcf|lcm|greatest\s+common\s+(?:divisor|factor)|highest\s+common\s+factor|least\s+common\s+multiple|lowest\s+common\s+multiple)\s+(?:of\s+)?(.+)$", low)
    if m:
        which = "lcm" if "l" == m.group(1)[0] else "gcd"
        nums = [as_int(x) for x in _numbers_list(m.group(2))]
        if len(nums) < 2 or any(x is None for x in nums):
            raise MathError(f"{which} needs two or more whole numbers")
        v = apply(which, [Fraction(x) for x in nums])
        name = "greatest common divisor" if which == "gcd" else "least common multiple"
        return _result(int(v), f"The {name} of {', '.join(map(str, nums))} is {int(v):,}.", which)
    m = re.search(r"(" + _NUM + r")\s*(?:choose|c)\s*(" + _NUM + r")\b", low)
    if m and ("choose" in low or re.search(r"\d\s*c\s*\d", low)):
        n, k = as_int(number(m.group(1))), as_int(number(m.group(2)))
        v = math.comb(n, k)
        return _result(v, f"{n} choose {k} = {v:,}.", "binomial")
    m = re.search(r"\b(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+fibonacci(?:\s+number)?\b|\bfibonacci\s+(?:number\s+)?(\d+)\b", low)
    if m:
        n = int(m.group(1) or m.group(2))
        v = apply("fib", [Fraction(n)])
        return _result(out(v), f"Fibonacci number {n} is {fmt(v)}.", "fibonacci")
    m = re.search(r"\b(" + _NUM + r"|0x[0-9a-f]+|0b[01]+|0o[0-7]+)\s+(?:in|to|into)\s+(binary|hex(?:adecimal)?|octal|decimal|base\s*\d+)\b", low)
    if m:
        raw = m.group(1).replace(",", "")
        n = int(raw, 0) if raw.startswith(("0x", "0b", "0o")) else int(raw)
        to = m.group(2)
        base = {"binary": 2, "octal": 8, "decimal": 10}.get(to, 16 if to.startswith("hex") else int(re.sub(r"\D", "", to) or 10))
        s = _to_base(n, base)
        return _result(s, f"{raw} in {to} is {s}.", "base")
    m = re.search(r"\b(?:binary|hex(?:adecimal)?|octal)\s+([0-9a-f]+)\s+(?:in|to|into)\s+decimal\b", low)
    if m:
        base = 2 if "binary" in low else 8 if "octal" in low else 16
        n = int(m.group(1), base)
        return _result(n, f"{m.group(1)} in base {base} is {n:,} in decimal.", "base")
    return None


def _to_base(n: int, base: int) -> str:
    if not 2 <= base <= 36:
        raise MathError("bases from 2 to 36")
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    s, x = "", abs(n)
    while x:
        s = digits[x % base] + s
        x //= base
    return ("-" if n < 0 else "") + s


def _factor_text(f: dict) -> str:
    return " × ".join(f"{p}" + (f"^{e}" if e > 1 else "") for p, e in f.items())


def statistics(low: str):
    m = re.search(r"\b(arithmetic\s+mean|geometric\s+mean|harmonic\s+mean|mean|average|median|mode|"
                  r"(?:population\s+|sample\s+)?(?:standard\s+deviation|variance|std(?:\s*dev)?)|range|sum|product|"
                  r"maximum|minimum|max|min)\s+of\s+(?:the\s+)?(?:numbers\s+)?([-\d.,\s]+(?:and\s+[-\d.]+)?)\s*$", low)
    if not m:
        return None
    what, xs = m.group(1), _numbers_list(m.group(2).replace(" and ", ", "))
    if len(xs) < 1:
        return None
    if what in ("sum", "product", "maximum", "minimum", "max", "min") and len(xs) < 2:
        return None
    n = len(xs)
    mean = sum(xs) / n
    if what in ("mean", "average", "arithmetic mean"):
        return _result(out(mean), f"The mean of {n} numbers is {show_num(mean)}.", "mean")
    if what == "geometric mean":
        if any(x <= 0 for x in xs):
            raise MathError("a geometric mean needs positive numbers")
        v = power(reduce(lambda a, b: a * b, xs), Fraction(1, n))
        return _result(out(v), f"The geometric mean is {show_num(v)}.", "mean")
    if what == "harmonic mean":
        if any(x == 0 for x in xs):
            raise MathError("a harmonic mean needs non-zero numbers")
        v = n / sum(1 / x for x in xs)
        return _result(out(v), f"The harmonic mean is {show_num(v)}.", "mean")
    if what == "median":
        s = sorted(xs)
        v = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        return _result(out(v), f"The median is {fmt(v, True)}.", "median")
    if what == "mode":
        counts: dict = {}
        for x in xs:
            counts[x] = counts.get(x, 0) + 1
        top = max(counts.values())
        modes = [x for x, c in counts.items() if c == top]
        if top == 1:
            return _result("none", "Every number appears once, so there is no mode.", "mode")
        text = ", ".join(fmt(x, True) for x in modes)
        return _result(out(modes[0]) if len(modes) == 1 else text,
                       f"The mode is {text} ({top} times)." if len(modes) == 1 else f"The modes are {text} ({top} times each).", "mode")
    if what == "range":
        v = max(xs) - min(xs)
        return _result(out(v), f"The range is {fmt(v, True)} (from {fmt(min(xs), True)} to {fmt(max(xs), True)}).", "range")
    if what == "sum":
        v = sum(xs)
        return _result(out(v), f"The sum is {fmt(v, True)}.", "sum")
    if what == "product":
        v = reduce(lambda a, b: a * b, xs)
        return _result(out(v), f"The product is {fmt(v, True)}.", "product")
    if what in ("maximum", "max", "minimum", "min"):
        v = max(xs) if what.startswith("max") else min(xs)
        return _result(out(v), f"The {'largest' if what.startswith('max') else 'smallest'} is {fmt(v, True)}.", what[:3])
    ss = sum((x - mean) ** 2 for x in xs)
    pop_var = ss / n
    samp_var = ss / (n - 1) if n > 1 else None
    sample = "sample" in what
    if "variance" in what:
        v = samp_var if sample else pop_var
        if v is None:
            raise MathError("a sample variance needs two or more numbers")
        other = "" if sample or samp_var is None else f" (the sample variance is {fmt(samp_var, True)})"
        return _result(out(v), f"The {'sample' if sample else 'population'} variance is {fmt(v, True)}{other}.", "variance")
    v = apply("sqrt", [samp_var if sample else pop_var]) if (samp_var if sample else pop_var) is not None else None
    if v is None:
        raise MathError("a sample standard deviation needs two or more numbers")
    other = "" if sample or samp_var is None else f" (the sample standard deviation is {fmt(apply('sqrt', [samp_var]), True)})"
    return _result(out(v), f"The {'sample' if sample else 'population'} standard deviation is {fmt(v, True)}{other}.", "stdev")


def percentages(low: str):
    m = re.search(r"what\s+(?:percent|percentage|%)\s+of\s+(" + _NUM + r")\s+is\s+(" + _NUM + r")", low)
    if m:
        a, b = number(m.group(1)), number(m.group(2))
        v = b / a * 100
        return _result(out(v), f"{fmt(b, True)} is {fmt(v, True)}% of {fmt(a, True)}.", "percent")
    m = re.search(r"(" + _NUM + r")\s+is\s+what\s+(?:percent|percentage|%)\s+of\s+(" + _NUM + r")", low)
    if m:
        a, b = number(m.group(1)), number(m.group(2))
        v = a / b * 100
        return _result(out(v), f"{fmt(a, True)} is {fmt(v, True)}% of {fmt(b, True)}.", "percent")
    m = re.search(r"\b(increase|raise|decrease|reduce|lower|cut)\s+(" + _NUM + r")\s+by\s+(" + _NUM + r")\s*(?:%|percent)", low)
    if m:
        v0, p = number(m.group(2)), number(m.group(3))
        up = m.group(1) in ("increase", "raise")
        v = v0 * (1 + p / 100) if up else v0 * (1 - p / 100)
        return _result(out(v), f"{fmt(v0, True)} {'increased' if up else 'decreased'} by {fmt(p, True)}% is {fmt(v, True)}.", "percent")
    m = re.search(r"percent(?:age)?\s+(?:change|increase|decrease)\s+from\s+(" + _NUM + r")\s+to\s+(" + _NUM + r")", low)
    if m:
        a, b = number(m.group(1)), number(m.group(2))
        if a == 0:
            raise MathError("a percentage change from zero is undefined")
        v = (b - a) / a * 100
        return _result(out(v), f"From {fmt(a, True)} to {fmt(b, True)} is a {fmt(abs(v), True)}% "
                               f"{'increase' if v >= 0 else 'decrease'}.", "percent")
    return None


def _matrix(text: str):
    m = re.search(r"\[\s*\[.*?\]\s*\]", text)
    if m:
        rows = re.findall(r"\[([^\[\]]*)\]", m.group(0))
        mat = [[number(x.strip()) if re.fullmatch(r"\d+(?:\.\d+)?", x.strip().lstrip("-")) is None
                else (Fraction(x.strip())) for x in r.split(",") if x.strip()] for r in rows]
        return mat, m.end()
    m = re.search(r"\[([^\[\]]*;[^\[\]]*)\]", text)
    if m:
        mat = [[Fraction(x) for x in row.replace(",", " ").split()] for row in m.group(1).split(";")]
        return mat, m.end()
    return None, 0


def _det(M) -> Fraction:
    n = len(M)
    A = [list(r) for r in M]
    det = Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            det = -det
        det *= A[c][c]
        for r in range(c + 1, n):
            k = A[r][c] / A[c][c]
            A[r] = [a - k * b for a, b in zip(A[r], A[c])]
    return det


def _mshow(M) -> str:
    return "[" + ", ".join("[" + ", ".join(fmt(x) for x in r) + "]" for r in M) + "]"


def matrices(low: str):
    if "[" not in low:
        return None
    A, end = _matrix(low)
    if A is None:
        return None
    if any(len(r) != len(A[0]) for r in A):
        raise MathError("every row of a matrix needs the same length")
    B, _ = _matrix(low[end:])
    square = len(A) == len(A[0])
    if B is not None and re.search(r"\]\s*(?:\*|times|x|multiplied by)\s*\[", low):
        if len(A[0]) != len(B):
            raise MathError("those matrices cannot be multiplied (their sizes do not match)")
        C = [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
        return _result(_mshow(C), f"The product is {_mshow(C)}.", "matrix")
    if re.search(r"\bdet(?:erminant)?\b", low):
        if not square:
            raise MathError("only square matrices have a determinant")
        v = _det(A)
        return _result(out(v), f"The determinant is {fmt(v)}.", "matrix")
    if re.search(r"\binverse\b|\binvert\b", low):
        if not square:
            raise MathError("only square matrices have an inverse")
        n = len(A)
        status, _ = gauss(A, [Fraction(0)] * n)
        if _det(A) == 0:
            raise MathError("that matrix has no inverse (its determinant is 0)")
        inv = []
        for j in range(n):
            e = [Fraction(int(i == j)) for i in range(n)]
            _, col = gauss(A, e)
            inv.append(col)
        inv = [list(r) for r in zip(*inv)]
        return _result(_mshow(inv), f"The inverse is {_mshow(inv)}.", "matrix")
    if re.search(r"\btranspose\b", low):
        T = [list(r) for r in zip(*A)]
        return _result(_mshow(T), f"The transpose is {_mshow(T)}.", "matrix")
    if re.search(r"\brank\b", low):
        status, r = gauss(A, [Fraction(0)] * len(A))
        rank = len(A[0]) if status == "unique" else r
        return _result(rank, f"The rank is {rank}.", "matrix")
    return None


def _bounds(text: str) -> float:
    t = text.strip().lower()
    if t in ("inf", "infinity", "+inf", "+infinity", "∞"):
        return math.inf
    if t in ("-inf", "-infinity", "-∞"):
        return -math.inf
    return float(real(ev(parse(words_to_maths(t), False))))


def calculus(low: str, ctx: Context):
    m = re.search(r"\b(?:the\s+)?(?:(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+)?(?:derivative|differential)\s+of\s+(.+?)"
                  r"(?:\s+(?:with\s+respect\s+to|wrt|w\.r\.t\.)\s+([a-z]))?(?:\s+(?:at|when|where|for)\s+([a-z])\s*=\s*(.+))?$", low)
    m2 = re.search(r"\b(?:differentiate|d/d([a-z]))\s+(.+?)(?:\s+(?:with\s+respect\s+to|wrt)\s+([a-z]))?(?:\s+(?:at|when|where)\s+([a-z])\s*=\s*(.+))?$", low)
    if m or m2:
        if m:
            order_word, body, wrt, at_var, at_val = m.groups()
        else:
            wrt0, body, wrt, at_var, at_val = m2.groups()
            wrt, order_word = wrt or wrt0, None
        order_n = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}.get(order_word or "first")
        if order_n is None:
            order_n = int(re.sub(r"\D", "", order_word))
        if order_n > 8:
            raise MathError("derivatives up to the 8th")
        node = parse(_split_vars(words_to_maths(body)))
        vs = variables_in(node)
        v = wrt or ("x" if "x" in vs else (sorted(vs)[0] if vs else "x"))
        der = node
        for _ in range(order_n):
            der = simplify(d(der, v))
        name = {1: "derivative", 2: "second derivative", 3: "third derivative"}.get(order_n, f"{ordinal(order_n)} derivative")
        text = show(der)
        if at_val:
            val = ev(parse(words_to_maths(at_val), False))
            r = ev(der, {at_var or v: val})
            return _result(out(r), f"The {name} of {show(node)} is {text}; at {at_var or v} = {fmt(val)} it is {show_num(r)}.",
                           "derivative", text)
        return _result(text, f"The {name} of {show(node)} with respect to {v} is {text}.", "derivative", text)
    m = re.search(r"\b(?:(?:the\s+)?(?:definite\s+|indefinite\s+)?integral\s+of|integrate|(?:the\s+)?area\s+under(?:\s+the\s+curve)?)\s+(.+?)"
                  r"(?:\s*d([a-z]))?(?:\s+(?:with\s+respect\s+to|wrt)\s+([a-z]))?"
                  r"(?:\s+(?:from|between)\s+(\S+)\s+(?:to|and)\s+(\S+))?$", low)
    if m:
        body, dv, wrt, a, b = m.groups()
        node = parse(_split_vars(words_to_maths(body)))
        vs = variables_in(node)
        v = dv or wrt or ("x" if "x" in vs or not vs else sorted(vs)[0])
        anti = integrate(node, v)
        if a is None:
            if anti is None:
                raise MathError(f"its rules find no antiderivative of {show(node)}; give limits ('from 0 to 1') "
                                "and it will integrate numerically")
            text = show(simplify(anti)) + " + C"
            return _result(text, f"The integral of {show(node)} with respect to {v} is {text}.", "integral", text)
        lo, hi = _bounds(a), _bounds(b)
        numeric = definite(node, v, lo, hi)
        exact_v = None
        if anti is not None and not math.isinf(lo) and not math.isinf(hi):
            try:
                fa, fb = ev(anti, {v: Fraction(a) if re.fullmatch(_NUM, a) else lo}), ev(anti, {v: Fraction(b) if re.fullmatch(_NUM, b) else hi})
                e = fb - fa
                if not isinstance(e, complex) and abs(float(e) - numeric) <= 1e-6 * max(1.0, abs(numeric)):
                    exact_v = e
            except (MathError, ValueError, ZeroDivisionError, TypeError):
                pass
        value = exact_v if exact_v is not None else numeric
        how = f" (an antiderivative is {show(simplify(anti))})" if anti is not None else " (numerically)"
        return _result(out(value), f"The integral of {show(node)} from {fmt(lo)} to {fmt(hi)} is {show_num(value)}{how}.",
                       "integral", show(node))
    m = re.search(r"\b(?:the\s+)?lim(?:it)?\s+(?:of\s+)?(.+?)\s+as\s+([a-z])\s*(?:->|→|approaches|tends\s+to|goes\s+to)\s*(\S+?)(\+|-)?$", low) \
        or re.search(r"\blim(?:it)?\s+([a-z])\s*(?:->|→)\s*(\S+?)(\+|-)?\s+(?:of\s+)?(.+)$", low)
    if m:
        if m.re.pattern.startswith(r"\blim(?:it)?\s+([a-z])"):
            v, to, side, body = m.groups()
        else:
            body, v, to, side = m.groups()
        node = parse(_split_vars(words_to_maths(body)))
        a = _bounds(to)
        r = limit(node, v, a, side or "")
        where = f"{v} → {fmt(a)}" + (side or "")
        return _result(out(r) if not math.isinf(r) else fmt(r),
                       f"The limit of {show(node)} as {where} is {show_num(r)}.", "limit", show(node))
    m = re.search(r"\b(sum|product)\s+of\s+(.+?)\s+(?:for|as|with)\s+([a-z])\s+(?:from|=|going\s+from)\s*(\S+)\s+to\s+(\S+)$", low) \
        or re.search(r"\b(sum|product)\s+(?:from\s+)?([a-z])\s*=\s*(\S+)\s+to\s+(\S+)\s+of\s+(.+)$", low)
    if m:
        g = m.groups()
        if len(g) == 5 and re.fullmatch(r"[a-z]", g[1] or "") and m.re.pattern.startswith(r"\b(sum|product)\s+(?:from"):
            what, v, a, b, body = g
        else:
            what, body, v, a, b = g
        lo, hi = as_int(ev(parse(words_to_maths(a), False))), as_int(ev(parse(words_to_maths(b), False)))
        if lo is None or hi is None:
            raise MathError("sums and products run over whole numbers")
        if hi - lo > 1_000_000:
            raise MathError("at most a million terms")
        node = parse(_split_vars(words_to_maths(body)))
        acc = Fraction(0) if what == "sum" else Fraction(1)
        for k in range(lo, hi + 1):
            t = ev(node, {v: Fraction(k)})
            acc = acc + t if what == "sum" else acc * t
        return _result(out(acc), f"The {what} of {show(node)} for {v} from {lo} to {hi} is {show_num(acc)}.", what, show(node))
    m = re.search(r"\bsum\s+of\s+(?:all\s+)?(?:the\s+)?first\s+(\d[\d,]*)\s+(?:positive\s+|natural\s+)?(integers|numbers|whole\s+numbers|odd\s+numbers|even\s+numbers|squares|cubes)", low)
    if m:
        n, kind = int(m.group(1).replace(",", "")), m.group(2)
        v = {"odd numbers": n * n, "even numbers": n * (n + 1), "squares": n * (n + 1) * (2 * n + 1) // 6,
             "cubes": (n * (n + 1) // 2) ** 2}.get(kind, n * (n + 1) // 2)
        return _result(v, f"The sum of the first {n:,} {kind} is {v:,}.", "sum")
    return None


def _equations(text: str):
    """The equations in an order (split on ',', ';' and 'and'), each as (lhs - rhs, relation)."""
    parts = [p.strip() for p in re.split(r"\s*(?:;|,(?!\d{3})|\band\b)\s*", text) if p.strip()]
    out_ = []
    for p in parts:
        m = re.match(r"^(.*?)(<=|>=|==|!=|=|<|>)(.*)$", p)
        if not m:
            return None
        lhs, rel, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
        if not lhs or not rhs:
            return None
        node = simplify(("-", parse(_split_vars(lhs)), parse(_split_vars(rhs))))
        out_.append((node, "=" if rel in ("=", "==") else rel, f"{lhs} {rel} {rhs}"))
    return out_


def algebra(low: str, ctx: Context):
    m = re.search(r"\b(expand|simplify|multiply\s+out)\s+(.+)$", low)
    if m:
        node = parse(_split_vars(words_to_maths(m.group(2))))
        vs = sorted(variables_in(node))
        p = to_poly(node, vs) if vs else None
        if p is None:
            text = show(simplify(node))
        else:
            text = pshow(p, vs)
        return _result(text, f"{show(node) if p is None else words_to_maths(m.group(2))} = {text}.", "expand", text)
    m = re.search(r"\b(?:factor|factori[sz]e)\s+(.+)$", low)
    if m and re.search(r"[a-z]", re.sub(r"\b(?:the|polynomial|expression)\b", "", m.group(1))):
        node = parse(_split_vars(words_to_maths(m.group(1))))
        vs = sorted(variables_in(node))
        if len(vs) != 1:
            raise MathError("it factors polynomials in one letter")
        p = to_poly(node, vs)
        if p is None:
            raise MathError(f"{show(node)} is not a polynomial")
        text = factor_poly(coeffs(p), vs[0])
        return _result(text, f"{pshow(p, vs)} = {text}.", "factor", text)
    m = re.search(r"\b(?:solve|find\s+(?:the\s+value\s+of\s+)?[a-z](?:\s+and\s+[a-z])*\s*(?:if|when|where|given|such\s+that|:)|"
                  r"what\s+is\s+[a-z]\s+(?:if|when|given))\s*:?\s*(.+?)(?:\s+for\s+([a-z](?:\s*(?:,|and)\s*[a-z])*))?$", low)
    body, target = (m.group(1), m.group(2)) if m else (None, None)
    m_for = re.search(r"\bsolve\s+for\s+([a-z](?:\s*(?:,|and)\s*[a-z])*)\s*(?::|in|,|\s)\s*(.+)$", low)
    if m_for:
        body, target = m_for.group(2), m_for.group(1)
    m_roots = re.search(r"\b(?:roots|zeros|zeroes|solutions)\s+of\s+(.+)$", low)
    if body is None and m_roots:
        body = m_roots.group(1)
        if "=" not in body:
            body += " = 0"
    bare = False
    if body is None and re.search(r"[a-z]", re.sub(r"\b(?:pi|e|sqrt|sin|cos|tan|ln|log|exp|abs)\b", "", low)) and \
            re.search(r"(?<![<>!=])=(?!=)|<|>", low) and not re.search(r"\b(?:when|where|at|if|with|for)\s+[a-z]\s*=", low):
        body, bare = low, True
    if body is None:
        return None
    body = words_to_maths(body)
    try:
        eqs = _equations(body)
    except MathError:
        if bare:
            return None
        raise
    if not eqs:
        return None
    allvars = sorted(set().union(*(variables_in(n) for n, _, _ in eqs)))
    targets = [t for t in re.findall(r"\b[a-z]\b", target or "") if t in allvars]  # "solve for x" with no x: what is there
    if not allvars:
        raise MathError("there is nothing to solve for: no unknown letter")
    if len(eqs) > 1 or (len(allvars) > 1 and len(targets) > 1):
        return _system(eqs, allvars if len(eqs) > 1 else targets)
    node, rel, original = eqs[0]
    v = targets[0] if targets else ("x" if "x" in allvars else allvars[0])
    others = [x for x in allvars if x != v]
    if others:
        return _solve_for(node, v, others, original)
    if rel != "=":
        return _inequality(node, rel, v, original)
    return _solve_one(node, v, original)


def _solve_one(node, v: str, original: str) -> dict:
    p = to_poly(node, [v])
    if p is not None:
        c = coeffs(p)
        while len(c) > 1 and c[-1] == 0:
            c = c[:-1]
        if len(c) == 1:
            if c[0] == 0:
                return _result("all numbers", f"{original} is true for every {v}.", "solve")
            return _result("no solution", f"{original} has no solution.", "solve")
        r = solve_poly(c)
        reals = sorted({round(float(x), 12): x for x in r["real"]}.values(), key=_sort_key)
        parts = []
        exact_reals = [x for x in reals if exact(x)]
        approx = [x for x in reals if not exact(x)]
        parts += [f"{v} = {fmt(x)}" for x in exact_reals]
        if r["exact"]:
            for e in r["exact"]:
                if "i" in e:
                    continue
                parts.append(f"{v} = {e}")
            approx_txt = " or ".join(f"{v} ≈ {fmt(x)}" for x in approx)
        else:
            parts += [f"{v} ≈ {fmt(x)}" if nice(x) is None else f"{v} = {nice(x)}" for x in approx]
            approx_txt = ""
        comp = [e for e in r["exact"] if "i" in e]
        if r["complex"] and not comp:
            seen = []
            for z in sorted(r["complex"], key=lambda z: (z.real, -abs(z.imag))):
                if z.imag > 0:
                    seen.append(f"{fmt(z.real)} ± {fmt(z.imag)}i")
            comp = seen
        say = f"{original}: " + (" or ".join(parts) if parts else "no real solution")
        if approx_txt:
            say += f" ({approx_txt})"
        if comp:
            say += ("; complex: " if parts else ", complex: ") + "; ".join(f"{v} = {z}" for z in comp)
        output = (out(reals[0]) if len(reals) == 1 and not comp and exact(reals[0]) else
                  " or ".join(parts) if parts else ("; ".join(f"{v} = {z}" for z in comp) or "no solution"))
        return _result(output, say + ".", "solve", show(node) + " = 0")
    roots = numeric_roots(node, v)
    if not roots:
        raise MathError(f"it found no real solution of {original} between -1,000,000 and 1,000,000")
    shown = roots
    more = ""
    if len(roots) > 8:
        near = sorted(roots, key=abs)[:6]
        shown = sorted(near)
        more = f" (and {len(roots) - 6} more: it repeats)"
    texts = [f"{v} = {nice(x)}" if nice(x) else f"{v} ≈ {fmt(x)}" for x in shown]
    output = out(shown[0]) if len(shown) == 1 else " or ".join(texts)
    return _result(output, f"{original}: " + " or ".join(texts) + more + " (found numerically).", "solve", show(node) + " = 0")


def _inequality(node, rel: str, v: str, original: str) -> dict:
    p = to_poly(node, [v])
    if p is None:
        raise MathError("it solves inequalities of polynomials")
    c = coeffs(p)
    while len(c) > 1 and c[-1] == 0:
        c = c[:-1]
    r = solve_poly(c) if len(c) > 1 else {"real": []}
    pts = sorted({round(float(x), 12): x for x in r.get("real", [])}.values(), key=_sort_key)
    test = {"<": lambda y: y < 0, ">": lambda y: y > 0, "<=": lambda y: y <= 0, ">=": lambda y: y >= 0, "!=": lambda y: y != 0}[rel]
    f = lambda x: float(_horner(c, Fraction(x) if not isinstance(x, float) else x))  # noqa: E731
    edges = [float(x) for x in pts]
    probes = ([edges[0] - 1] if edges else [0.0]) + [(a + b) / 2 for a, b in zip(edges, edges[1:])] + ([edges[-1] + 1] if edges else [])
    pieces = []
    closed = rel in ("<=", ">=")
    for i, x in enumerate(probes):
        if not test(f(x)):
            continue
        lo = pts[i - 1] if i > 0 else None
        hi = pts[i] if i < len(pts) else None
        if lo is None and hi is None:
            pieces.append(f"every {v}")
        elif lo is None:
            pieces.append(f"{v} {'≤' if closed else '<'} {fmt(hi) if exact(hi) else fmt(float(hi))}")
        elif hi is None:
            pieces.append(f"{v} {'≥' if closed else '>'} {fmt(lo) if exact(lo) else fmt(float(lo))}")
        else:
            op = "≤" if closed else "<"
            pieces.append(f"{fmt(lo) if exact(lo) else fmt(float(lo))} {op} {v} {op} {fmt(hi) if exact(hi) else fmt(float(hi))}")
    if closed and not pieces and pts:
        pieces = [f"{v} = {fmt(x)}" for x in pts if test(0)]
    text = " or ".join(pieces) if pieces else "no solution"
    return _result(text, f"{original}: {text}.", "inequality")


def _solve_for(node, v: str, others: list, original: str) -> dict:
    vs = [v] + others
    p = to_poly(node, vs)
    if p is None:
        raise MathError(f"it can only rearrange polynomial equations for {v}")
    deg = max((k[0] for k in p), default=0)
    if deg != 1:
        raise MathError(f"{original} is not linear in {v}; give values for {', '.join(others)} or more equations")
    a = {k[1:]: c for k, c in p.items() if k[0] == 1}
    b = {k[1:]: c for k, c in p.items() if k[0] == 0}
    if len(a) == 1 and (0,) * len(others) in a:
        k = a[(0,) * len(others)]
        rhs = {kk: -c / k for kk, c in b.items()}
        text = pshow(rhs, others) if rhs else "0"
    else:
        text = f"-({pshow(b, others) or '0'})/({pshow(a, others)})"
    return _result(f"{v} = {text}", f"Solving {original} for {v}: {v} = {text}.", "solve")


def _system(eqs: list, vs: list) -> dict:
    if any(rel != "=" for _, rel, _ in eqs):
        raise MathError("systems of inequalities are not supported")
    nodes = [n for n, _, _ in eqs]
    polys = [to_poly(n, vs) for n in nodes]
    if all(p is not None and all(sum(k) <= 1 for k in p) for p in polys):
        A = [[p.get(tuple(int(i == j) for i in range(len(vs))), Fraction(0)) for j in range(len(vs))] for p in polys]
        b = [-p.get((0,) * len(vs), Fraction(0)) for p in polys]
        status, sol = gauss(A, b)
        if status == "none":
            return _result("no solution", "Those equations contradict each other: there is no solution.", "system")
        if status == "infinite":
            return _result("infinitely many solutions",
                           f"Those equations have infinitely many solutions ({len(vs) - sol} free unknown(s)).", "system")
        text = ", ".join(f"{x} = {fmt(s)}" for x, s in zip(vs, sol))
        return _result(text, f"{text}.", "system")
    if len(nodes) != len(vs):
        raise MathError(f"{len(nodes)} equations for {len(vs)} unknowns: it needs as many equations as unknowns")
    found = newton_system(nodes, vs)
    if not found:
        raise MathError("it found no solution numerically")
    texts = ["(" + ", ".join(f"{x} = {nice(s) or fmt(s)}" for x, s in sol.items()) + ")" for sol in found[:8]]
    return _result(" or ".join(texts), f"Found {len(found)} solution{'s' if len(found) != 1 else ''} numerically: "
                   + " or ".join(texts) + ("; there may be others." if found else ""), "system")


def evaluate(text: str, ctx: Context, raw: str) -> dict:
    m = re.search(r"^(.*?)\s+(?:at|when|where|if|for|with)\s+((?:[a-z]\s*=\s*[^,;]+(?:\s*(?:,|and)\s*)?)+)$", text)
    env = {}
    body = text
    if m and "=" in m.group(2):
        body = m.group(1)
        for name, val in re.findall(r"([a-z])\s*=\s*([^,;]+?)(?=\s*(?:,|\band\b|$))", m.group(2)):
            env[name] = ev(parse(words_to_maths(val), False))
    expr = words_to_maths(body)
    if not expr:
        raise MathError("it could not find a calculation in that")
    node, subs = resolve(_split_vars(expr) if env else expr, ctx, variables=True)
    vs = variables_in(node) - set(env)
    if "i" in vs:  # the imaginary unit, where nothing else is called i
        env["i"] = 1j
        vs.discard("i")
    if vs:
        raise MathError(f"it needs a value for {', '.join(sorted(vs))} (or an equation to solve)")
    value = ev(node, env)
    if isinstance(value, list):
        raise MathError("that is a list, not a calculation")
    shown = re.sub(r"\s*\^\s*", "^", " ".join(expr.split()))
    shown = re.sub(r"(\d+(?:\.\d+)?)\*pi/180", r"\1°", shown)
    shown = re.sub(r"\bsqrt\s*", "√", shown)
    shown = re.sub(r"\((\d+(?:\.\d+)?)/100\)\*\s*", r"\1% of ", shown)
    for name, phrase in subs.items():
        shown = re.sub(rf"\bq\d+\b", phrase, shown, count=1) if name not in shown else re.sub(rf"\b{name}\b", phrase, shown)
    given = {k: x for k, x in env.items() if not (k == "i" and x == 1j)}
    where = (" when " + ", ".join(f"{k} = {fmt(x)}" for k, x in given.items())) if given else ""
    exact_note = ""
    if exact(value) and value.denominator != 1 and value.denominator <= 10000:
        exact_note = f" = {fmt(float(value))}" if len(fmt(float(value))) <= 14 else f" ≈ {fmt(float(value))}"
    elif isinstance(value, float):
        n = nice(value)
        if n and re.search(r"[π√eφ]", n) and n != fmt(value):
            exact_note = f" (= {n})"
    told = ""
    if ctx.used:
        told = " (" + "; ".join(ctx.used) + ")"
    if fmt(value) == shown.replace(" ", "") and exact_note:  # 22/7 = 3.142857143, not 22/7 = 22/7
        say = f"{shown}{exact_note.replace(' = ', ' = ', 1)}{where}{told}"
    else:
        say = f"{shown} = {fmt(value)}{exact_note}{where}{told}"
    return _result(out(value), say, "arithmetic", expr, **({"used": list(ctx.used)} if ctx.used else {}))


def answer(text: str, context=None) -> dict:
    """The calculator's answer to an order: a result dict, or raises MathError / NeedToKnow."""
    ctx = Context(context)
    raw = " ".join(str(text).split())
    low = raw.lower()
    low = re.sub(r"×|·", "*", low)
    for handler in (matrices, percentages, number_theory, statistics):
        r = handler(low)
        if r is not None:
            return r
    for handler in (calculus, algebra):
        r = handler(low, ctx)
        if r is not None:
            return r
    return evaluate(low, ctx, raw)
