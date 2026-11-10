"""The calculator's mathematics (plugins/calculator/mathsolver.py) and its tool protocol: equations,
calculus, number theory, statistics, matrices, and asking a director for what it does not know."""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "plugins" / "calculator" / "tool.py"
spec = importlib.util.spec_from_file_location("mathsolver", ROOT / "plugins" / "calculator" / "mathsolver.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def ask(text, context=None):
    return ms.answer(text, context)


def run_tool(text, context=None, folder=TOOL):
    request = {"input": text}
    if context:
        request["context"] = context
    p = subprocess.run([sys.executable, "-I", str(folder)], input=json.dumps(request), capture_output=True, text=True,
                       timeout=30, cwd=folder.parent)
    return json.loads(p.stdout.strip().splitlines()[-1])


class ArithmeticTests(unittest.TestCase):
    def test_what_the_first_calculator_did_still_works(self):
        for text, want in [("what is 12 * 7", 84), ("calculate 100 divided by 8", 12.5), ("19 plus 23 minus 2 = ?", 40),
                           ("What is 7 plus 8 times 2?", 23), ("What is 12 squared?", 144), ("What is 25% of 80?", 20),
                           ("What is 2 to the power of 10?", 1024), ("What is (20 - 5) * 3?", 45), ("what is 3 x 4", 12)]:
            self.assertEqual(ask(text)["output"], want, text)
        self.assertEqual(ask("what is 12 * 7")["say"], "12 * 7 = 84")

    def test_exact_numbers_and_functions(self):
        self.assertEqual(ask("what is 1/3 + 1/6")["say"], "1/3 + 1/6 = 1/2 = 0.5")
        self.assertEqual(ask("2^100")["output"], 1267650600228229401496703205376)
        self.assertEqual(ask("square root of 144")["output"], 12)
        self.assertIn("√2", ask("sqrt(2)")["say"])
        self.assertEqual(ask("what is sin(30 degrees)")["output"], 0.5)
        self.assertEqual(ask("log base 2 of 8")["output"], 3)
        self.assertEqual(ask("10!")["output"], 3628800)
        self.assertEqual(ask("what is 17 mod 5")["output"], 2)
        self.assertEqual(ask("what is e^(i*pi)")["output"], -1)
        self.assertEqual(ask("x^2 + 3x when x = 2")["say"], "x^2 + 3x = 10 when x = 2")
        self.assertEqual(ask("12 * 7 for my homework")["output"], 84)  # words beside a calculation are not a quantity
        with self.assertRaises(ms.MathError):
            ask("what is 1 / 0")


class AlgebraTests(unittest.TestCase):
    def test_equations(self):
        self.assertEqual(ask("solve 2x + 3 = 7")["output"], 2)
        self.assertEqual(ask("solve x^2 - 5x + 6 = 0")["output"], "x = 2 or x = 3")
        self.assertIn("x = ±√2", ask("solve x^2 - 2 = 0")["say"])
        self.assertIn("x = -1 ± 2i", ask("solve x^2 + 2x + 5 = 0")["say"])
        self.assertEqual(ask("solve x^3 - 6x^2 + 11x - 6 = 0")["output"], "x = 1 or x = 2 or x = 3")
        self.assertIn("x ≈ 1.167303978", ask("solve x^5 - x - 1 = 0")["say"])  # no closed form: numerically
        self.assertIn("π/6", ask("solve sin(x) = 0.5")["say"])
        self.assertEqual(ask("solve 2^x = 32")["output"], 5)
        self.assertEqual(ask("solve for x: 2x + 3 = 11")["output"], 4)
        self.assertEqual(ask("solve 3(x - 2) = 2(x + 1)")["output"], 8)

    def test_systems_inequalities_and_rearranging(self):
        self.assertEqual(ask("solve 2x + y = 5, x - y = 1")["output"], "x = 2, y = 1")
        self.assertIn("(x = 4, y = 3)", ask("solve x^2 + y^2 = 25 and x - y = 1")["say"])
        self.assertEqual(ask("solve x + y = 1, 2x + 2y = 3")["output"], "no solution")
        self.assertEqual(ask("solve 3x - 2 > 7")["output"], "x > 3")
        self.assertEqual(ask("solve x^2 - 4 < 0")["output"], "-2 < x < 2")
        self.assertEqual(ask("solve y = 2x + 3 for x")["output"], "x = (1/2)y - 3/2")
        self.assertEqual(ask("expand (x+1)^3")["output"], "x^3 + 3x^2 + 3x + 1")
        self.assertEqual(ask("factor 2x^2 + 5x - 3")["output"], "(x + 3)(2x - 1)")


class CalculusTests(unittest.TestCase):
    def test_derivatives(self):
        self.assertEqual(ask("derivative of x^3 + 2x")["output"], "3x^2 + 2")
        self.assertEqual(ask("second derivative of x^4")["output"], "12x^2")
        self.assertEqual(ask("derivative of x^2 at x = 3")["output"], 6)
        self.assertEqual(ask("differentiate e^(2x)")["output"], "2e^(2x)")
        self.assertEqual(ask("derivative of ln(x^2+1)")["output"], "2x/(x^2 + 1)")
        self.assertEqual(ask("derivative of sin(x)*x^2")["output"], "x^2*cos(x) + 2x*sin(x)")

    def test_integrals_limits_and_sums(self):
        self.assertEqual(ask("integral of x^2")["output"], "(1/3)x^3 + C")
        self.assertEqual(ask("integrate x^2 from 0 to 3")["output"], 9)
        self.assertEqual(ask("integral of sin(x) from 0 to pi")["output"], 2)
        self.assertEqual(ask("integrate x*sin(x)")["output"], "-x*cos(x) + sin(x) + C")  # by parts
        self.assertEqual(ask("integrate 1/x")["output"], "ln|x| + C")
        self.assertIn("√π", ask("integral of e^(-x^2) from -inf to inf")["say"])  # over the whole line, numerically
        self.assertEqual(ask("limit of sin(x)/x as x -> 0")["output"], 1)
        self.assertIn("is e ≈", ask("limit of (1 + 1/x)^x as x -> infinity")["say"])
        with self.assertRaises(ms.MathError):
            ask("limit of 1/x as x -> 0")  # -infinity from the left, infinity from the right
        self.assertEqual(ask("sum of n^2 for n from 1 to 10")["output"], 385)
        self.assertEqual(ask("sum from k=1 to 100 of k")["output"], 5050)


class NumberTheoryAndMoreTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(ask("is 97 prime")["output"], "yes")
        self.assertEqual(ask("is 91 prime")["say"], "No, 91 is not prime: 91 = 7 × 13.")
        self.assertEqual(ask("factor 360")["output"], "2^3 × 3^2 × 5")
        self.assertEqual(ask("prime factors of 600851475143")["output"], "71 × 839 × 1471 × 6857")
        self.assertEqual(ask("the 1000th prime")["output"], 7919)
        self.assertEqual(ask("lcm of 4, 6 and 10")["output"], 60)
        self.assertEqual(ask("10 choose 3")["output"], 120)
        self.assertEqual(ask("the 50th fibonacci number")["output"], 12586269025)
        self.assertEqual(ask("255 in binary")["output"], "11111111")

    def test_statistics_matrices_percentages(self):
        self.assertEqual(ask("mean of 3, 5, 7, 10")["output"], 6.25)
        self.assertEqual(ask("median of 4, 1, 3, 2")["output"], 2.5)
        self.assertEqual(ask("standard deviation of 2, 4, 4, 4, 5, 5, 7, 9")["output"], 2)
        self.assertEqual(ask("determinant of [[1,2],[3,4]]")["output"], -2)
        self.assertEqual(ask("inverse of [[1,2],[3,4]]")["output"], "[[-2, 1], [3/2, -1/2]]")
        self.assertEqual(ask("what percent of 120 is 30")["output"], 25)
        self.assertEqual(ask("increase 80 by 15%")["output"], 92)


class InputCheckingTests(unittest.TestCase):
    """mathsolver.tidy: what people paste - braces, LaTeX, labels and ratios with ':', other symbols, odd brackets."""

    def test_braces_brackets_and_labels(self):
        for text, want in [("calculate: {2 + 3} * [4 - 1]", 15), ("{{1+1}}*3", 6), ("q1: 3 + 4", 7), ("math: 6 × 7", 42),
                           ("Problem 2: 7*6", 42), ("a) 5*5", 25), ("solve: x + 1 = 3", 2), ("solve {2x + 3 = 11}", 4),
                           ("x^2 - 5x + 6 = 0 : solve", "x = 2 or x = 3"), ("mean of [1, 2, 3, 4]", 2.5)]:
            self.assertEqual(ask(text)["output"], want, text)

    def test_latex(self):
        for text, want in [(r"\frac{3}{4} \cdot 8", 6), (r"\dfrac{\frac{1}{2}}{4}", 0.125), (r"$\sqrt{16} + 2^{3}$", 12),
                           (r"\sqrt[3]{27}", 3), (r"\frac{x}{2} = 4", 8), (r"\sin(\frac{\pi}{2})", 1),
                           (r"\frac{d}{dx}(x^3 + 2x)", "3x^2 + 2"), (r"\int_0^3 x^2 dx", 9), (r"\int_{0}^{\pi} \sin(x) \, dx", 2),
                           (r"\lim_{x \to 0} \frac{\sin(x)}{x}", 1), (r"\sum_{k=1}^{10} k^2", 385)]:
            self.assertEqual(ask(text)["output"], want, text)
        r = ask(r"\frac{3}{4} \cdot 8")
        self.assertEqual(r["read_as"], "(3/4) * 8")  # no brackets around bare numbers
        self.assertIn("LaTeX read as plain maths", r["say"])

    def test_other_ways_of_writing_symbols(self):
        for text, want in [("10 ÷ 4", 2.5), ("2³ + 1", 9), ("½ + ¼", 0.75), ("1½ * 2", 3), ("x² − 5x + 6 = 0", "x = 2 or x = 3"),
                           ("√49 + 1", 8), ("2 ∶ 4", 0.5), ("$5 + $3", 8), ("“2 + 2”", 4), ("2​+​2", 4)]:
            self.assertEqual(ask(text)["output"], want, text)

    def test_colons_ratios_and_times(self):
        r = ask("the ratio 3:4")
        self.assertEqual(r["output"], 0.75)
        self.assertIn("a ratio a:b read as a/b", r["say"])
        self.assertEqual(ask("ratio 3:4 of 20")["output"], 15)
        self.assertEqual(ask("(3/4) of 20")["output"], 15)
        r = ask("at 3:30 what is 2+2")  # a time of day is not part of the sum
        self.assertEqual(r["output"], 4)
        self.assertIn("the time 3:30 left out", r["say"])
        self.assertEqual(ask("f(x) = x^2 + 1 at x = 3")["output"], 10)
        self.assertEqual(ask("f(3) where f(x)=2x+1")["output"], 7)

    def test_unbalanced_brackets_are_mended_and_said(self):
        for text, note in [("((2+3)*4", "1 missing ')' added"), ("(2+3))*4", "1 extra ')' dropped")]:
            r = ask(text)
            self.assertEqual(r["output"], 20, text)
            self.assertIn(note, r["say"])
        self.assertEqual(ms.tidy("(2)(3)"), "(2)(3)")  # brackets that mean something stay
        self.assertEqual(ask("2(3)")["output"], 6)

    def test_what_it_cannot_read_is_named(self):
        for text, char in [("what is 3 # 4", "#"), ("5 @ 2", "@")]:
            with self.assertRaises(ms.MathError) as cm:
                ask(text)
            self.assertIn(f"'{char}'", str(cm.exception))
        out = run_tool("what is 3 # 4")  # said plainly, as the order's fault, not the tool's
        self.assertTrue(out["bad_input"] and out["not_found"])
        self.assertIn("'#'", out["error"])
        self.assertNotIn("trace", out)
        t = ms.tidy("{2+3}*[4-1]")
        self.assertEqual((t, t.notes), ("(2+3)*(4-1)", []))


class LoopBackTests(unittest.TestCase):
    def test_it_asks_a_director_for_what_it_does_not_know(self):
        with self.assertRaises(ms.NeedToKnow) as caught:
            ask("what is the number of bones in the human body times 2")
        self.assertEqual(caught.exception.ask, "What is the number of bones in the human body?")
        out = run_tool("what is the speed of light times 3")
        self.assertEqual(out["query"]["ask"], "What is the speed of light?")
        told = [{"ask": "What is the speed of light?", "answer": "Light travels at about 299,792 kilometres per second.",
                 "source": "memory node mem-0001"}]
        out = run_tool("what is the speed of light times 3", told)
        self.assertEqual(out["output"], 899376)
        self.assertIn("from memory node mem-0001", out["say"])
        out = run_tool("what is the speed of light times 3", [{"ask": "What is the speed of light?", "answer": None,
                                                               "source": "nothing"}])
        self.assertIn("could not give one", out["error"])

    def test_calculator2_shares_the_mathematics(self):
        v2 = ROOT / "examples" / "incoming_plugins" / "calculator2" / "tool.py"
        self.assertEqual(run_tool("what is 15% of 80", folder=v2)["output"], 12)
        self.assertEqual(run_tool("solve x^2 - 9 = 0", folder=v2)["output"], "x = -3 or x = 3")
        self.assertEqual((ROOT / "plugins" / "calculator" / "mathsolver.py").read_text(),
                         (ROOT / "examples" / "incoming_plugins" / "calculator2" / "mathsolver.py").read_text())


if __name__ == "__main__":
    unittest.main()
