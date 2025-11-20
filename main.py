import os
import random
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime, timezone

app = FastAPI(title="Math Tutor API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# Models
# -----------------------------
class PracticeRequest(BaseModel):
    topic: str = Field(..., description="Topic key, e.g., 'arithmetic_addition'")
    difficulty: int = Field(1, ge=1, le=5)
    count: int = Field(5, ge=1, le=20)
    seed: Optional[int] = None
    curriculum_id: Optional[str] = Field(None, description="Optional curriculum document id to align content")
    lang: str = Field("en", description="Language code: en, fa, de, fr")

class PracticeProblem(BaseModel):
    id: str
    topic: str
    question: str
    answer: Any
    steps: List[str]

class PracticeResponse(BaseModel):
    topic: str
    problems: List[PracticeProblem]

class ExplainResponse(BaseModel):
    topic: str
    title: str
    summary: str
    key_points: List[str]
    examples: List[str]

class CurriculumDoc(BaseModel):
    id: str
    name: str
    level: Optional[str] = None
    region: Optional[str] = None
    size: int
    pages: Optional[int] = None
    text_preview: Optional[str] = None

class QuizStartRequest(BaseModel):
    user: str
    topic: str
    difficulty: int = Field(1, ge=1, le=5)
    count: int = Field(5, ge=1, le=20)
    lang: str = Field("en")

class QuizQuestion(BaseModel):
    id: str
    question: str

class QuizStartResponse(BaseModel):
    quiz_id: str
    user: str
    topic: str
    lang: str
    questions: List[QuizQuestion]

class QuizSubmitRequest(BaseModel):
    quiz_id: str
    answers: List[Any]

class QuizSubmitResponse(BaseModel):
    quiz_id: str
    user: str
    topic: str
    lang: str
    total: int
    correct: int
    score: float
    points: int

# -----------------------------
# i18n helpers
# -----------------------------
SUPPORTED_LANGS = {"en", "fa", "de", "fr"}

TOPIC_LABELS_EN = {
    "arithmetic_addition": "Addition",
    "arithmetic_subtraction": "Subtraction",
    "arithmetic_multiplication": "Multiplication",
    "arithmetic_division": "Division",
    "fractions_basic": "Fractions (Basics)",
    "fractions_operations": "Fractions (Add/Subtract)",
    "algebra_linear": "Algebra: Solve ax + b = c",
    "ratio_proportion": "Ratios & Proportions",
    "algebra_quadratic": "Quadratic Equations",
    "functions_polynomial": "Polynomials",
    "geometry_area": "Geometry: Area",
    "calculus_derivative": "Calculus: Derivatives (Polynomials)",
    "calculus_integral": "Calculus: Integrals (Polynomials)",
    "linear_algebra_vectors": "Linear Algebra: Vectors (Dot Product)",
    "linear_algebra_matrices": "Linear Algebra: Matrix Multiply (2x2)",
    "probability_basic": "Probability: Discrete (Dice/Coin)",
    "proof_techniques": "Proof Techniques (Concepts)",
}

TOPIC_LABELS_FA = {
    "arithmetic_addition": "جمع",
    "arithmetic_subtraction": "تفریق",
    "arithmetic_multiplication": "ضرب",
    "arithmetic_division": "تقسیم",
    "fractions_basic": "کسرها (مقدماتی)",
    "fractions_operations": "کسرها (جمع/تفریق)",
    "algebra_linear": "جبر: حل ax + b = c",
    "ratio_proportion": "نسبت و تناسب",
    "algebra_quadratic": "معادلات درجه دوم",
    "functions_polynomial": "چندجمله‌ای‌ها",
    "geometry_area": "هندسه: مساحت",
    "calculus_derivative": "حساب دیفرانسیل: مشتق‌ها",
    "calculus_integral": "حساب انتگرال: انتگرال‌ها",
    "linear_algebra_vectors": "جبر خطی: بردارها (ضرب داخلی)",
    "linear_algebra_matrices": "جبر خطی: ضرب ماتریس (۲×۲)",
    "probability_basic": "احتمال: گسسته (سکه/تاس)",
    "proof_techniques": "روش‌های اثبات (مفاهیم)",
}

TOPIC_LABELS_DE = {
    "arithmetic_addition": "Addition",
    "arithmetic_subtraction": "Subtraktion",
    "arithmetic_multiplication": "Multiplikation",
    "arithmetic_division": "Division",
    "fractions_basic": "Brüche (Grundlagen)",
    "fractions_operations": "Brüche (Add/Sub)",
    "algebra_linear": "Algebra: Löse ax + b = c",
    "ratio_proportion": "Verhältnisse & Proportionen",
    "algebra_quadratic": "Quadratische Gleichungen",
    "functions_polynomial": "Polynome",
    "geometry_area": "Geometrie: Fläche",
    "calculus_derivative": "Analysis: Ableitungen",
    "calculus_integral": "Analysis: Integrale",
    "linear_algebra_vectors": "Lineare Algebra: Vektoren (Skalarprodukt)",
    "linear_algebra_matrices": "Lineare Algebra: Matrixmultiplikation (2x2)",
    "probability_basic": "Wahrscheinlichkeit: Diskret (Würfel/Münze)",
    "proof_techniques": "Beweistechniken (Konzepte)",
}

TOPIC_LABELS_FR = {
    "arithmetic_addition": "Addition",
    "arithmetic_subtraction": "Soustraction",
    "arithmetic_multiplication": "Multiplication",
    "arithmetic_division": "Division",
    "fractions_basic": "Fractions (Bases)",
    "fractions_operations": "Fractions (Add/Sous)",
    "algebra_linear": "Algèbre : Résoudre ax + b = c",
    "ratio_proportion": "Rapports et Proportions",
    "algebra_quadratic": "Équations quadratiques",
    "functions_polynomial": "Polynômes",
    "geometry_area": "Géométrie : Aire",
    "calculus_derivative": "Calcul : Dérivées",
    "calculus_integral": "Calcul : Intégrales",
    "linear_algebra_vectors": "Algèbre linéaire : Vecteurs (Produit scalaire)",
    "linear_algebra_matrices": "Algèbre linéaire : Multiplication de matrices (2x2)",
    "probability_basic": "Probabilité : Discrète (Dé/Coin)",
    "proof_techniques": "Techniques de preuve (Concepts)",
}

LANG_TOPIC_LABELS = {
    "en": TOPIC_LABELS_EN,
    "fa": TOPIC_LABELS_FA,
    "de": TOPIC_LABELS_DE,
    "fr": TOPIC_LABELS_FR,
}

# -----------------------------
# Curriculum Map
# -----------------------------
LEVELS = [
    "2nd Grade",
    "3rd-5th Grade",
    "Middle School",
    "High School",
    "Undergraduate",
    "Masters",
]

TOPICS_BY_LEVEL_KEYS: Dict[str, List[str]] = {
    "2nd Grade": ["arithmetic_addition", "arithmetic_subtraction"],
    "3rd-5th Grade": ["arithmetic_multiplication", "arithmetic_division", "fractions_basic"],
    "Middle School": ["fractions_operations", "algebra_linear", "ratio_proportion"],
    "High School": ["algebra_quadratic", "functions_polynomial", "geometry_area"],
    "Undergraduate": ["calculus_derivative", "calculus_integral", "linear_algebra_vectors", "linear_algebra_matrices"],
    "Masters": ["probability_basic", "proof_techniques"],
}

# -----------------------------
# Generators & Solvers
# -----------------------------

def gen_addition(d: int):
    base = 10 ** d
    a = random.randint(1, base - 1)
    b = random.randint(1, base - 1)
    s = a + b
    steps = [
        "Line up the numbers by place value.",
        f"Add ones: {(a % 10)} + {(b % 10)}.",
        "Carry if needed, then add tens, hundreds, ...",
        f"Total: {a} + {b} = {s}",
    ]
    return f"{a} + {b} = ?", s, steps


def gen_subtraction(d: int):
    base = 10 ** d
    a = random.randint(1, base - 1)
    b = random.randint(0, a)
    s = a - b
    steps = [
        "Line up the numbers by place value.",
        "Subtract ones, borrow if needed.",
        f"Compute: {a} - {b} = {s}",
    ]
    return f"{a} - {b} = ?", s, steps


def gen_multiplication(d: int):
    a = random.randint(2, 10 ** d)
    b = random.randint(2, 10 ** d)
    p = a * b
    steps = [
        "Multiply ones, then tens, etc.",
        "Add partial products.",
        f"Compute: {a} × {b} = {p}",
    ]
    return f"{a} × {b} = ?", p, steps


def gen_division(d: int):
    b = random.randint(2, 10 ** d)
    q = random.randint(2, 10 ** d)
    a = b * q
    steps = [
        f"We choose numbers so division is exact.",
        f"{a} ÷ {b} = {q}",
    ]
    return f"{a} ÷ {b} = ?", q, steps


def simplify_fraction(n: int, d: int):
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x

    g = gcd(n, d)
    return n // g, d // g, g


def gen_fraction_basic(difficulty: int):
    n = random.randint(1, 9 * difficulty)
    d = random.randint(n + 1, 10 * difficulty + n)
    sn, sd, g = simplify_fraction(n, d)
    steps = [
        f"Find gcd({n}, {d}) = {g}.",
        f"Divide numerator and denominator by {g}.",
        f"{n}/{d} simplifies to {sn}/{sd}.",
    ]
    return f"Simplify {n}/{d}", f"{sn}/{sd}", steps


def gen_fraction_add(difficulty: int):
    a, b = random.randint(1, 9 * difficulty), random.randint(2, 10 * difficulty)
    c, d_ = random.randint(1, 9 * difficulty), random.randint(2, 10 * difficulty)
    num = a * d_ + c * b
    den = b * d_
    sn, sd, g = simplify_fraction(num, den)
    steps = [
        f"Common denominator: {b}×{d_} = {den}.",
        f"Convert: {a}/{b} = {a*d_}/{den}, {c}/{d_} = {c*b}/{den}.",
        f"Add numerators: {a*d_} + {c*b} = {num}.",
        f"Simplify by gcd {g}: {num}/{den} -> {sn}/{sd}.",
    ]
    q = f"{a}/{b} + {c}/{d_} = ?"
    return q, f"{sn}/{sd}", steps


def gen_linear_equation(difficulty: int):
    a = random.randint(1, 8 * difficulty)
    x = random.randint(-10 * difficulty, 10 * difficulty)
    b = random.randint(-10 * difficulty, 10 * difficulty)
    c = a * x + b
    steps = [
        f"Start with {a}x + {b} = {c}.",
        f"Subtract {b} both sides: {a}x = {c - b}.",
        f"Divide by {a}: x = {(c - b) / a}.",
    ]
    return f"Solve: {a}x + {b} = {c}", x, steps


def gen_quadratic(difficulty: int):
    # Generate factorable quadratics: (x+r1)(x+r2)
    r1 = random.randint(-5 * difficulty, 5 * difficulty)
    r2 = random.randint(-5 * difficulty, 5 * difficulty)
    a = 1
    b = r1 + r2
    c = r1 * r2
    disc = b * b - 4 * a * c
    steps = [
        f"Equation: x^2 + {b}x + {c} = 0.",
        f"Discriminant Δ = b^2 - 4ac = {disc}.",
        "Since a=1 and Δ is a perfect square for chosen roots.",
        f"Roots are r1={r1}, r2={r2}.",
    ]
    q = f"Solve: x^2 + {b}x + {c} = 0"
    ans = [r1, r2]
    return q, ans, steps


def gen_poly_derivative(difficulty: int):
    # f(x) = a_n x^n + ... + a0 with small degrees
    deg = random.randint(1, min(5, 1 + difficulty))
    coeffs = [random.randint(-5, 5) for _ in range(deg + 1)]
    while coeffs[-1] == 0:
        coeffs[-1] = random.randint(1, 5)

    def poly_str(cs):
        parts = []
        for i, a in enumerate(cs):
            if a == 0:
                continue
            if i == 0:
                parts.append(str(a))
            elif i == 1:
                parts.append(f"{a}x")
            else:
                parts.append(f"{a}x^{i}")
        return " + ".join(parts) if parts else "0"

    deriv = [i * coeffs[i] for i in range(1, len(coeffs))]
    q = f"Differentiate f(x) = {poly_str(coeffs)}"
    ans = poly_str(deriv)
    steps = [
        "Use power rule: d/dx x^n = n x^(n-1).",
        "Differentiate each term and add results.",
        f"f'(x) = {ans}",
    ]
    return q, ans, steps


def gen_poly_integral(difficulty: int):
    deg = random.randint(0, min(4, difficulty + 1))
    coeffs = [random.randint(-5, 5) for _ in range(deg + 1)]
    while coeffs[-1] == 0:
        coeffs[-1] = random.randint(1, 5)

    def poly_str(cs):
        parts = []
        for i, a in enumerate(cs):
            if a == 0:
                continue
            if i == 0:
                parts.append(str(a))
            elif i == 1:
                parts.append(f"{a}x")
            else:
                parts.append(f"{a}x^{i}")
        return " + ".join(parts) if parts else "0"

    antideriv_terms = []
    for i, a in enumerate(coeffs):
        n = i + 1
        antideriv_terms.append((a / n, n))

    def poly_terms_to_str(ts):
        parts = []
        for a, n in ts:
            if a == 0:
                continue
            if n == 1:
                parts.append(f"{a}x")
            else:
                parts.append(f"{a}x^{n}")
        return " + ".join(parts) if parts else "0"

    q = f"Integrate F(x) = ∫({poly_str(coeffs)}) dx"
    ans = poly_terms_to_str(antideriv_terms) + " + C"
    steps = [
        "Use power rule for integrals: ∫x^n dx = x^(n+1)/(n+1).",
        "Integrate each term and add constant C.",
        f"F(x) = {ans}",
    ]
    return q, ans, steps


def gen_vector_dot(difficulty: int):
    n = random.randint(2, min(5, 2 + difficulty))
    v = [random.randint(-5, 5) for _ in range(n)]
    w = [random.randint(-5, 5) for _ in range(n)]
    dot = sum(vi * wi for vi, wi in zip(v, w))
    steps = [
        f"Compute sum of products: {v} · {w} = " + " + ".join([f"{vi}×{wi}" for vi, wi in zip(v, w)]) + f" = {dot}",
    ]
    return f"Find v·w for v={v}, w={w}", dot, steps


def gen_matrix_mult_2x2(_: int):
    A = [[random.randint(-3, 3) for _ in range(2)], [random.randint(-3, 3) for _ in range(2)]]
    B = [[random.randint(-3, 3) for _ in range(2)], [random.randint(-3, 3) for _ in range(2)]]
    C = [[A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
         [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
    steps = [
        "Use (AB)_{ij} = sum_k A_{ik}B_{kj} for 2x2.",
        f"Result: {C}",
    ]
    return f"Compute A·B for A={A}, B={B}", C, steps


def gen_probability_basic(difficulty: int):
    # Simple: probability of event in fair dice/coin experiments
    if random.random() < 0.5:
        # coin flips
        n = random.randint(1, 4 + difficulty)
        k = random.randint(0, n)
        from math import comb
        ways = comb(n, k)
        total = 2 ** n
        p = ways / total
        steps = [
            f"Number of ways to get {k} heads in {n} flips is C({n},{k}) = {ways}.",
            f"Total outcomes: 2^{n} = {total}.",
            f"Probability = {ways}/{total} = {p}",
        ]
        return f"In {n} fair coin flips, probability of exactly {k} heads?", p, steps
    else:
        # dice
        n = random.randint(1, 2 + difficulty)
        target = random.randint(1, 6)
        p = 1/6 if n == 1 else None
        if n == 1:
            steps = [
                "A fair die has 6 equally likely outcomes.",
                "Probability of a specific face is 1/6.",
            ]
            return f"Roll one die. Probability of a {target}?", p, steps
        else:
            from math import comb
            k = random.randint(0, n)
            ways = comb(n, k) * (1) * (5 ** (n - k))
            total = 6 ** n
            p = ways / total
            steps = [
                f"Treat getting {target} as success (p=1/6).",
                f"Ways to get exactly {k} successes in {n} trials: C({n},{k}) (1)^{k} (5)^{n-k}.",
                f"Total outcomes: 6^{n} = {total}.",
                f"Probability = {ways}/{total} = {p}",
            ]
            return f"Roll {n} dice. Probability of exactly {k} times showing {target}?", p, steps


def gen_ratio_proportion(difficulty: int):
    a = random.randint(2, 8 * difficulty)
    b = random.randint(2, 8 * difficulty)
    k = random.randint(2, 6)
    x = k * a
    steps = [
        f"Given ratio {a}:{b}, scaling by k={k} gives {x}:{k*b}.",
        f"Unknown is x in x:{k*b} = {a}:{b} -> x = {k}×{a} = {x}",
    ]
    return f"If a:b = {a}:{b}, find x such that x:{k*b} = {a}:{b}", x, steps


def gen_geometry_area(difficulty: int):
    # Mix rectangle and triangle
    if random.random() < 0.5:
        w = random.randint(2, 10 * difficulty)
        h = random.randint(2, 10 * difficulty)
        area = w * h
        steps = [
            "Area of rectangle = width × height.",
            f"Compute: {w} × {h} = {area}",
        ]
        return f"Find area of rectangle with width={w}, height={h}", area, steps
    else:
        b = random.randint(2, 10 * difficulty)
        h = random.randint(2, 10 * difficulty)
        area = 0.5 * b * h
        steps = [
            "Area of triangle = 1/2 × base × height.",
            f"Compute: 1/2 × {b} × {h} = {area}",
        ]
        return f"Find area of triangle with base={b}, height={h}", area, steps


def gen_polynomial_eval(difficulty: int):
    deg = random.randint(1, min(4, difficulty + 1))
    coeffs = [random.randint(-3, 5) for _ in range(deg + 1)]
    x = random.randint(-3, 3)

    def poly_str(cs):
        parts = []
        for i, a in enumerate(cs):
            if a == 0:
                continue
            if i == 0:
                parts.append(str(a))
            elif i == 1:
                parts.append(f"{a}x")
            else:
                parts.append(f"{a}x^{i}")
        return " + ".join(parts) if parts else "0"

    val = sum(a * (x ** i) for i, a in enumerate(coeffs))
    steps = [
        f"Evaluate term by term at x={x}.",
        f"Result: {val}",
    ]
    return f"Evaluate f(x)={poly_str(coeffs)} at x={x}", val, steps

# registry of generators
GENERATOR_MAP = {
    "arithmetic_addition": lambda diff: gen_addition(max(1, min(3, diff))),
    "arithmetic_subtraction": lambda diff: gen_subtraction(max(1, min(3, diff))),
    "arithmetic_multiplication": lambda diff: gen_multiplication(max(1, min(2, diff))),
    "arithmetic_division": lambda diff: gen_division(max(1, min(2, diff))),
    "fractions_basic": gen_fraction_basic,
    "fractions_operations": gen_fraction_add,
    "algebra_linear": gen_linear_equation,
    "algebra_quadratic": gen_quadratic,
    "functions_polynomial": gen_polynomial_eval,
    "geometry_area": gen_geometry_area,
    "calculus_derivative": gen_poly_derivative,
    "calculus_integral": gen_poly_integral,
    "linear_algebra_vectors": gen_vector_dot,
    "linear_algebra_matrices": gen_matrix_mult_2x2,
    "ratio_proportion": gen_ratio_proportion,
    "probability_basic": gen_probability_basic,
}

EXPLANATIONS: Dict[str, ExplainResponse] = {}

# Populate explanations with concise concept overviews (English source)
EXPLANATIONS_DATA_EN = {
    "arithmetic_addition": {
        "title": "Addition",
        "summary": "Combine two or more numbers to find their total.",
        "key_points": [
            "Add by place value (ones, tens, hundreds).",
            "Order doesn't matter (commutative).",
            "Group numbers for easier sums (associative).",
        ],
        "examples": ["23 + 45 = 68", "300 + 120 = 420"],
    },
    "arithmetic_subtraction": {
        "title": "Subtraction",
        "summary": "Find how much remains or the difference between numbers.",
        "key_points": [
            "Line up by place value.",
            "Borrow from next place when needed.",
        ],
        "examples": ["54 - 29 = 25", "1000 - 456 = 544"],
    },
    "fractions_basic": {
        "title": "Fractions (Basics)",
        "summary": "A fraction a/b represents a parts out of b equal parts.",
        "key_points": [
            "Simplify by dividing by gcd(a,b).",
            "Equivalent fractions represent the same value.",
        ],
        "examples": ["6/8 = 3/4", "2/3 = 4/6"],
    },
    "algebra_linear": {
        "title": "Linear Equations",
        "summary": "Solve for x in ax+b=c by inverse operations.",
        "key_points": [
            "Add/subtract to isolate the term with x.",
            "Divide by the coefficient of x.",
        ],
        "examples": ["3x + 4 = 19 -> x = 5"],
    },
    "algebra_quadratic": {
        "title": "Quadratic Equations",
        "summary": "Equations of the form ax^2+bx+c=0 solved by factoring or quadratic formula.",
        "key_points": [
            "Discriminant Δ=b^2-4ac determines number of roots.",
            "Factorable quadratics have integer roots.",
        ],
        "examples": ["x^2-5x+6=0 -> (x-2)(x-3)=0"],
    },
    "calculus_derivative": {
        "title": "Derivatives of Polynomials",
        "summary": "Rate of change; apply power rule term-by-term.",
        "key_points": [
            "d/dx x^n = n x^(n-1).",
            "Constants differentiate to 0.",
        ],
        "examples": ["d/dx (3x^2 + 2x + 5) = 6x + 2"],
    },
    "calculus_integral": {
        "title": "Integrals of Polynomials",
        "summary": "Antiderivative; reverse of differentiation.",
        "key_points": [
            "∫ x^n dx = x^(n+1)/(n+1) + C (n≠-1).",
            "Add constant of integration C.",
        ],
        "examples": ["∫ (6x+2) dx = 3x^2 + 2x + C"],
    },
    "linear_algebra_vectors": {
        "title": "Vector Dot Product",
        "summary": "Multiply corresponding components and sum.",
        "key_points": [
            "v·w = Σ v_i w_i.",
            "Measures alignment between vectors.",
        ],
        "examples": ["[1,2]·[3,4] = 11"],
    },
    "linear_algebra_matrices": {
        "title": "Matrix Multiplication (2x2)",
        "summary": "(AB)_{ij} = Σ A_{ik} B_{kj}.",
        "key_points": [
            "Row-by-column multiplication.",
            "Order matters (not commutative).",
        ],
        "examples": ["[[1,2],[3,4]]·[[0,1],[1,0]] = [[2,1],[4,3]]"],
    },
    "probability_basic": {
        "title": "Basic Discrete Probability",
        "summary": "Count favorable outcomes over total equally likely outcomes.",
        "key_points": [
            "Use combinations for counts.",
            "For independent trials, multiply.",
        ],
        "examples": ["P(heads in one flip)=1/2", "P(sum=7 on two dice)=6/36"],
    },
    "ratio_proportion": {
        "title": "Ratios & Proportions",
        "summary": "Two ratios are proportional if they are equal.",
        "key_points": [
            "Solve by cross-multiplying or scaling.",
        ],
        "examples": ["a:b = 2:3, then 4:6 is equivalent"],
    },
    "functions_polynomial": {
        "title": "Evaluating Polynomials",
        "summary": "Substitute x and compute term by term.",
        "key_points": [
            "Use order of operations.",
        ],
        "examples": ["f(x)=x^2+2x+1, f(3)=16"],
    },
    "geometry_area": {
        "title": "Areas of Simple Shapes",
        "summary": "Use known formulas for rectangles and triangles.",
        "key_points": [
            "Rectangle: A=wh.",
            "Triangle: A=bh/2.",
        ],
        "examples": ["w=3,h=4 => A=12", "b=5,h=6 => A=15"],
    },
    "proof_techniques": {
        "title": "Proof Techniques (Overview)",
        "summary": "Common strategies to prove mathematical statements.",
        "key_points": [
            "Direct proof, contrapositive, contradiction, induction.",
            "State assumptions, show steps clearly, conclude logically.",
        ],
        "examples": ["Prove √2 is irrational by contradiction.", "Use induction to prove sum of first n integers is n(n+1)/2."],
    },
}

# For brevity, reuse EN content for other languages; titles will use translated topic labels
EXPLANATIONS_DATA = EXPLANATIONS_DATA_EN
for key, data in EXPLANATIONS_DATA.items():
    EXPLANATIONS[key] = ExplainResponse(topic=key, **data)

# -----------------------------
# Curriculum Upload & Management
# -----------------------------
@app.post("/api/curriculum/upload", response_model=CurriculumDoc)
async def upload_curriculum(
    file: UploadFile = File(...),
    level: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save file to disk
    file_id = str(uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    pages = None
    text_preview = None

    # Try to extract basic text preview and page count
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(dest_path)
        pages = len(reader.pages)
        texts = []
        for i, page in enumerate(reader.pages[:5]):
            txt = page.extract_text() or ""
            if txt:
                texts.append(txt.strip())
        combined = "\n\n".join(texts)
        text_preview = combined[:2000] if combined else None
    except Exception:
        pass

    # Store metadata to database if available
    meta = {
        "id": file_id,
        "name": file.filename,
        "level": level,
        "region": region,
        "size": len(content),
        "pages": pages,
        "text_preview": text_preview,
        "path": dest_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        from database import create_document
        create_document("curriculum", meta)
    except Exception:
        # proceed without DB
        pass

    return CurriculumDoc(**{k: meta[k] for k in ["id", "name", "level", "region", "size", "pages", "text_preview"]})


@app.get("/api/curriculum", response_model=List[CurriculumDoc])
async def list_curriculum():
    items: List[CurriculumDoc] = []
    # Try DB first
    try:
        from database import get_documents
        docs = get_documents("curriculum", {}, limit=100)
        for d in docs:
            items.append(CurriculumDoc(**{k: d.get(k) for k in ["id", "name", "level", "region", "size", "pages", "text_preview"] if k in d}))
        return items
    except Exception:
        # Fallback: scan uploads dir
        for fname in os.listdir(UPLOAD_DIR):
            if fname.endswith('.pdf'):
                fid = os.path.splitext(fname)[0]
                path = os.path.join(UPLOAD_DIR, fname)
                try:
                    size = os.path.getsize(path)
                except Exception:
                    size = 0
                items.append(CurriculumDoc(id=fid, name=fname, level=None, region=None, size=size, pages=None, text_preview=None))
        return items


@app.delete("/api/curriculum/{doc_id}")
async def delete_curriculum(doc_id: str):
    # remove from db if possible
    try:
        from database import db
        if db is not None:
            db["curriculum"].delete_one({"id": doc_id})
    except Exception:
        pass
    # remove file
    pdf_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception:
            pass
    return {"ok": True}

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Math Tutor Backend running"}

@app.get("/api/levels", response_model=List[str])
def get_levels(lang: str = Query("en")):
    # Levels shown in English for now; could add translation here if needed
    return LEVELS

@app.get("/api/topics")
def get_topics(level: str = Query(..., description="One of LEVELS"), lang: str = Query("en")):
    if level not in TOPICS_BY_LEVEL_KEYS:
        raise HTTPException(status_code=400, detail="Unknown level")
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    labels = LANG_TOPIC_LABELS.get(lang, TOPIC_LABELS_EN)
    keys = TOPICS_BY_LEVEL_KEYS[level]
    return [{"key": k, "label": labels.get(k, TOPIC_LABELS_EN.get(k, k))} for k in keys]

@app.get("/api/explain", response_model=ExplainResponse)
def explain(topic: str = Query(...), lang: str = Query("en")):
    if topic not in EXPLANATIONS:
        raise HTTPException(status_code=404, detail="Explanation not found for topic")
    data = EXPLANATIONS[topic]
    if lang in LANG_TOPIC_LABELS:
        # Replace title with translated topic label, keep body for brevity
        translated_title = LANG_TOPIC_LABELS[lang].get(topic, data.title)
        return ExplainResponse(topic=topic, title=translated_title, summary=data.summary, key_points=data.key_points, examples=data.examples)
    return data

@app.post("/api/practice", response_model=PracticeResponse)
def practice(req: PracticeRequest):
    if req.topic not in GENERATOR_MAP:
        raise HTTPException(status_code=400, detail="Unknown topic")
    if req.seed is not None:
        random.seed(req.seed)

    curriculum_note = None
    if req.curriculum_id:
        # Try to fetch curriculum name for alignment note
        try:
            from database import db
            doc = db["curriculum"].find_one({"id": req.curriculum_id}) if db is not None else None
            if doc and doc.get("name"):
                curriculum_note = f"Aligned with your uploaded course: {doc.get('name')}"
        except Exception:
            pass

    problems: List[PracticeProblem] = []
    generator = GENERATOR_MAP[req.topic]
    for i in range(req.count):
        q, a, steps = generator(req.difficulty)
        if curriculum_note:
            steps = [curriculum_note] + steps
        problems.append(
            PracticeProblem(
                id=f"{req.topic}-{i}", topic=req.topic, question=q, answer=a, steps=steps
            )
        )
    return PracticeResponse(topic=req.topic, problems=problems)

# -----------------------------
# Quiz & Report Card
# -----------------------------
@app.post("/api/quiz/start", response_model=QuizStartResponse)
def quiz_start(req: QuizStartRequest):
    if req.topic not in GENERATOR_MAP:
        raise HTTPException(status_code=400, detail="Unknown topic")
    quiz_id = str(uuid4())
    gen = GENERATOR_MAP[req.topic]
    questions = []
    answers = []
    for i in range(req.count):
        q, a, _ = gen(req.difficulty)
        qid = f"{quiz_id}-{i}"
        questions.append({"id": qid, "question": q})
        answers.append(a)

    # Persist quiz session with correct answers
    try:
        from database import db
        if db is not None:
            db["quiz"].insert_one({
                "id": quiz_id,
                "user": req.user,
                "topic": req.topic,
                "difficulty": req.difficulty,
                "count": req.count,
                "lang": req.lang,
                "questions": questions,
                "answers": answers,
                "created_at": datetime.now(timezone.utc)
            })
    except Exception:
        pass

    return QuizStartResponse(quiz_id=quiz_id, user=req.user, topic=req.topic, lang=req.lang, questions=[QuizQuestion(**q) for q in questions])

@app.post("/api/quiz/submit", response_model=QuizSubmitResponse)
def quiz_submit(req: QuizSubmitRequest):
    try:
        from database import db
        doc = None
        if db is not None:
            doc = db["quiz"].find_one({"id": req.quiz_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Quiz session not found")
        correct = 0
        details = []
        for i, user_ans in enumerate(req.answers):
            gold = doc.get("answers", [None]*len(req.answers))[i]
            is_correct = False
            # tolerant comparison for numbers and arrays
            try:
                if isinstance(gold, (int, float)):
                    is_correct = abs(float(user_ans) - float(gold)) < 1e-6
                elif isinstance(gold, list):
                    is_correct = list(user_ans) == gold
                else:
                    is_correct = str(user_ans).strip() == str(gold).strip()
            except Exception:
                is_correct = str(user_ans).strip() == str(gold).strip()
            correct += 1 if is_correct else 0
            details.append({"q": doc["questions"][i]["question"], "your": user_ans, "answer": gold, "correct": is_correct})
        total = len(doc.get("questions", []))
        score = (correct / total) * 100 if total else 0.0
        points = int(score)
        # Save result
        if db is not None:
            db["quiz_result"].insert_one({
                "quiz_id": req.quiz_id,
                "user": doc["user"],
                "topic": doc["topic"],
                "lang": doc.get("lang", "en"),
                "total": total,
                "correct": correct,
                "score": score,
                "points": points,
                "details": details,
                "created_at": datetime.now(timezone.utc)
            })
        return QuizSubmitResponse(quiz_id=req.quiz_id, user=doc["user"], topic=doc["topic"], lang=doc.get("lang", "en"), total=total, correct=correct, score=score, points=points)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to grade quiz")

@app.get("/api/report/{user}")
def report_card(user: str):
    summary: Dict[str, Any] = {"user": user, "points": 0, "quizzes": 0, "per_topic": {}, "recent": []}
    try:
        from database import db
        if db is None:
            # No DB; return empty summary
            summary["overall"] = {"score": 0.0, "grade": "-"}
            summary["final_value"] = 0.0
            return summary
        cursor = db["quiz_result"].find({"user": user}).sort("created_at", -1)
        results = list(cursor)
        summary["quizzes"] = len(results)
        total_score_sum = 0.0
        for r in results:
            s = float(r.get("score", 0))
            summary["points"] += int(r.get("points", 0))
            total_score_sum += s
            t = r.get("topic")
            pt = summary["per_topic"].setdefault(t, {"attempts": 0, "best": 0.0, "avg": 0.0})
            pt["attempts"] += 1
            pt["avg"] = ((pt["avg"] * (pt["attempts"] - 1)) + s) / pt["attempts"]
            pt["best"] = max(pt["best"], s)
        # overall average score across all quizzes
        overall_score = (total_score_sum / summary["quizzes"]) if summary["quizzes"] else 0.0
        # map to grade
        def to_grade(x: float) -> str:
            if x >= 90: return "A"
            if x >= 80: return "B"
            if x >= 70: return "C"
            if x >= 60: return "D"
            return "F"
        summary["overall"] = {"score": overall_score, "grade": to_grade(overall_score)}
        # final_value can be used by tutor to finalize a numeric value for report card
        summary["final_value"] = round(overall_score, 1)
        summary["recent"] = [{
            "quiz_id": r.get("quiz_id"),
            "topic": r.get("topic"),
            "score": r.get("score"),
            "points": r.get("points"),
        } for r in results[:10]]
        return summary
    except Exception:
        summary["overall"] = {"score": 0.0, "grade": "-"}
        summary["final_value"] = 0.0
        return summary

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
