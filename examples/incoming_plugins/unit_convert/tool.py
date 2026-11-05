"""Unit conversion tool."""
import json
import re
import sys

LINEAR = {}
for names, dim, factor in [
    (("km", "kilometer", "kilometers", "kilometre", "kilometres"), "length", 1000.0),
    (("m", "meter", "meters", "metre", "metres"), "length", 1.0),
    (("cm", "centimeter", "centimeters"), "length", 0.01),
    (("mile", "miles", "mi"), "length", 1609.344),
    (("ft", "foot", "feet"), "length", 0.3048),
    (("yard", "yards", "yd"), "length", 0.9144),
    (("kg", "kilogram", "kilograms", "kilo", "kilos"), "mass", 1.0),
    (("g", "gram", "grams"), "mass", 0.001),
    (("lb", "lbs", "pound", "pounds"), "mass", 0.45359237),
    (("oz", "ounce", "ounces"), "mass", 0.028349523125),
    (("l", "liter", "liters", "litre", "litres"), "volume", 1.0),
    (("ml", "milliliter", "milliliters"), "volume", 0.001),
    (("gallon", "gallons", "gal"), "volume", 3.785411784),
]:
    for n in names:
        LINEAR[n] = (dim, factor)
TEMPS = {"c": "c", "celsius": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}


def to_kelvin(v, u):
    return v + 273.15 if u == "c" else (v - 32) * 5 / 9 + 273.15 if u == "f" else v


def from_kelvin(v, u):
    return v - 273.15 if u == "c" else (v - 273.15) * 9 / 5 + 32 if u == "f" else v


def main():
    text = json.load(sys.stdin)["input"].lower()
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+", text)
    src = None
    for i, tok in enumerate(tokens[:-1]):
        if tok[0].isdigit() and (tokens[i + 1] in LINEAR or tokens[i + 1] in TEMPS):
            src = (float(tok), tokens[i + 1], i + 1)
            break
    if src is None:
        raise ValueError("could not find a quantity with a unit")
    value, unit, pos = src
    target = next((t for j, t in enumerate(tokens) if j != pos and (t in LINEAR or t in TEMPS) and t != unit), None)
    if target is None:
        raise ValueError("could not find a target unit")
    if unit in TEMPS and target in TEMPS:
        result = from_kelvin(to_kelvin(value, TEMPS[unit]), TEMPS[target])
    elif unit in LINEAR and target in LINEAR and LINEAR[unit][0] == LINEAR[target][0]:
        result = value * LINEAR[unit][1] / LINEAR[target][1]
    else:
        raise ValueError(f"cannot convert {unit} to {target}")
    print(json.dumps({"output": f"{value:g} {unit} = {result:.4f} {target}".replace(".0000 ", " ")}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
