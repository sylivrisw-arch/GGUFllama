"""
Simple Chat - a minimal stand-alone chat window with a dual-layer inference backend:
llama-cpp-python running .gguf models directly in-process on the GPU - no server, no
network hop. Chat, Discussion mode, the Models window, streaming and token counting all
run through that one backend.

Text input, text output, a Send button, a token counter, a Server card for connecting
to the models folder and picking which loaded model to talk to, a read-only
reference panel of hardcoded prompts, and a Models window for browsing a folder of .gguf
files.

Single-file app: the Constraint Engine (formerly a separate constraint_engine.py) and the
local llama.cpp backend (see "LOCAL BACKEND" below) are both embedded directly in this
file, so this script has no local file dependencies of its own beyond the llama-cpp-python
package (required - without it the app opens but Connect is refused).

The Block Editor (formerly the separate block_editor.py) is built in too: right-click the
title bar -> "Open Block Editor". It needs `pip install flask` (optional).

Run:  python GGUFllama.py
"""

import array
import io
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import difflib
import hashlib
import unicodedata
import wave as _wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:                     # llama-cpp-python not installed - Connect is refused
    Llama = None
    LLAMA_CPP_AVAILABLE = False

APP_TITLE = "Simple Chat"

# llama.cpp backend defaults - see the LOCAL BACKEND section further down.
# Default folder scanned for .gguf files (change it from the title-bar right-click menu).
DEFAULT_LOCAL_MODELS_ROOT = os.path.join(os.path.expanduser("~"), ".lmstudio", "models")
LOCAL_VRAM_SAFETY_MARGIN_MB = 700      # headroom left unused, for the desktop/driver overhead

PALETTE = {
    "bg": "#f4f5f7", "card": "#ffffff", "input": "#ffffff", "text": "#111827", "muted": "#6b7280",
    "on_accent": "#ffffff", "accent": "#2563eb", "neutral": "#e5e7eb",
    "discussion_a": "#7c3aed", "discussion_b": "#c026d3",
    "warn": "#f59e0b", "on_warn": "#ffffff", "go": "#16a34a", "on_go": "#ffffff",
}

FONT_UI = "Segoe UI"

# ==============================================================================================
#  CONSTRAINT ENGINE - embedded from constraint_engine.py (merged in so GGUFmanager.py is a
#  single self-contained file; no separate constraint_engine.py needs to sit next to it or be
#  importable any more). Everything below to the "END CONSTRAINT ENGINE" marker is that module,
#  dependency-free logic - no tkinter, no reference back into the rest of this file.
# ==============================================================================================

CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constraint_engine_data")


# ============================================================================
# CONSTRAINT ENGINE — BASE CLASSES
# ============================================================================

class Constraint(ABC):
    """Base constraint - all constraints inherit from this"""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.execution_time = 0

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Does data pass this constraint?"""
        pass

    @abstractmethod
    def fix(self, data: Any) -> Any:
        """Fix data if it fails"""
        pass

    def describe(self):
        """Human-readable description"""
        return f"{self.name}"


# ============================================================================
# CONSTRAINT ENGINE — SEPARATE CONSTRAINTS (Local, Fast, Per-Item)
# ============================================================================

class SeparateConstraint(Constraint):
    """Validates individual pieces in isolation"""
    pass


class NoDoubleSpaces(SeparateConstraint):
    """Collapse runs of spaces inside ordinary text into one.

    NOT touched, because collapsing them destroys meaning (Coding Mode, nested lists):
      - a line's leading indentation,
      - everything inside fenced code blocks (``` or ~~~; an unclosed fence protects the rest),
      - inline `code` spans.
    validate() means "collapsing would change nothing", so it can never disagree with fix()."""

    _FENCE_LINE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})")
    _INLINE_CODE = re.compile(r"(`+)[^`\n]+?\1")
    _RUN = re.compile(r" {2,}")

    @classmethod
    def _collapse_line(cls, line: str) -> str:
        body = line.lstrip(" \t")
        indent = line[:len(line) - len(body)]
        out, last = [], 0
        for m in cls._INLINE_CODE.finditer(body):
            out.append(cls._RUN.sub(" ", body[last:m.start()]))
            out.append(m.group(0))                      # inline code kept exactly as written
            last = m.end()
        out.append(cls._RUN.sub(" ", body[last:]))
        return indent + "".join(out)

    @classmethod
    def _collapse(cls, text: str) -> str:
        in_fence = False
        result = []
        for line in text.split("\n"):
            if cls._FENCE_LINE.match(line):
                in_fence = not in_fence
                result.append(line)
            elif in_fence:
                result.append(line)
            else:
                result.append(cls._collapse_line(line))
        return "\n".join(result)

    def validate(self, data: str) -> bool:
        if not isinstance(data, str):
            return True
        return self._collapse(data) == data

    def fix(self, data: str) -> str:
        if not isinstance(data, str):
            return data
        return self._collapse(data)


class CapitalizeFirst(SeparateConstraint):
    """First letter must be capitalized"""

    def validate(self, data: str) -> bool:
        if not data or not data.strip():
            return True
        # Leading whitespace fails (fix() strips it). A first character that has no upper case -
        # a quote, digit, '#', '*', a dash, an emoji - passes: fix() can't change it, so demanding
        # a capital there made the check fail forever and abort the rest of the engine's pass.
        if data[0].isspace():
            return False
        return data[0].isupper() or not data[0].isalpha()

    def fix(self, data: str) -> str:
        # A leading space/newline (common in local-model output) used to make this a no-op:
        # ' '.upper() is still ' ', so validate() failed again right after "fixing" it and the
        # whole round got discarded. Strip leading whitespace before capitalizing so the fix
        # actually fixes it.
        if not data:
            return data
        stripped = data.lstrip()
        if not stripped:
            return data
        return stripped[0].upper() + stripped[1:]


class ValidJSON(SeparateConstraint):
    """Must be valid JSON"""

    def __init__(self, name: str = "ValidJSON", required_fields: List[str] = None):
        super().__init__(name)
        self.required_fields = required_fields or []

    def validate(self, data: Union[str, dict]) -> bool:
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data

            if isinstance(parsed, dict):
                for field in self.required_fields:
                    if field not in parsed:
                        return False

            return True
        except Exception:
            return False

    def fix(self, data: str) -> str:
        return data


class NoTrailingWhitespace(SeparateConstraint):
    """Remove trailing spaces"""

    def validate(self, data: str) -> bool:
        if isinstance(data, str):
            return not data.endswith((" ", "\n", "\t"))
        return True

    def fix(self, data: str) -> str:
        if isinstance(data, str):
            return data.rstrip()
        return data


class IsNonEmpty(SeparateConstraint):
    """Data must not be empty"""

    def validate(self, data: Any) -> bool:
        if isinstance(data, (str, list, dict)):
            return len(data) > 0
        return data is not None

    def fix(self, data: Any) -> Any:
        return data


class MaxLength(SeparateConstraint):
    """Enforce maximum length"""

    def __init__(self, name: str = "MaxLength", max_len: int = 1000):
        super().__init__(name)
        self.max_len = max_len

    def validate(self, data: str) -> bool:
        return len(data) <= self.max_len

    def fix(self, data: str) -> str:
        return data[:self.max_len]


class ContainsKeyword(SeparateConstraint):
    """Must contain specific keyword"""

    def __init__(self, name: str = "ContainsKeyword", keyword: str = ""):
        super().__init__(name)
        self.keyword = keyword

    def validate(self, data: str) -> bool:
        return self.keyword.lower() in data.lower()

    def fix(self, data: str) -> str:
        return data


class NoCharacters(SeparateConstraint):
    """Ban specific characters"""

    def __init__(self, name: str = "NoCharacters", banned_chars: str = ""):
        super().__init__(name)
        self.banned_chars = banned_chars

    def validate(self, data: str) -> bool:
        return not any(char in data for char in self.banned_chars)

    def fix(self, data: str) -> str:
        for char in self.banned_chars:
            data = data.replace(char, "")
        return data


class NoMarkdown(SeparateConstraint):
    """Strip real markdown markup and leave ordinary characters alone.

    Removed: heading marks ('# Title'), code fences and `inline code` ticks, ~~strikethrough~~,
    paired **bold** / *italic* / __bold__ / _italic_ markers (the text inside is kept), and link
    syntax ('[text](url)' becomes 'text (url)'). '* item' bullets become '- item'.
    NOT touched: underscores inside words (snake_case), '!', a lone '*' or '#' ('2*3*4', 'C#',
    '#1'), or a bare '[' / ']'. validate() means "stripping would change nothing", so it can never
    disagree with fix()."""

    _FENCE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})[^\n]*(?:\n|$)", re.M)
    _HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M)
    _LINK = re.compile(r"!?\[([^\]\n]+)\]\(([^)\n]*)\)")
    _CODE = re.compile(r"(`+)([^`\n]+?)\1")
    _STRIKE = re.compile(r"~~(?=\S)([^\n]+?)(?<=\S)~~")
    _BULLET = re.compile(r"^([ \t]*)\*[ \t]+", re.M)
    _STAR = re.compile(r"(?<![\w*])(\*{1,3})(?![\s*])([^\n]+?)(?<![\s*])\1(?![\w*])")
    _UNDER = re.compile(r"(?<![\w_])(_{1,3})(?![\s_])([^\n]+?)(?<![\s_])\1(?![\w_])")

    @classmethod
    def _strip(cls, text: str) -> str:
        for _ in range(20):  # each pass only shortens the text (or turns '*' into '-'), so this settles fast
            before = text
            text = cls._FENCE.sub("", text)
            text = cls._HEADING.sub("", text)
            text = cls._LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})" if m.group(2).strip() else m.group(1), text)
            text = cls._CODE.sub(r"\2", text)
            text = cls._STRIKE.sub(r"\1", text)
            text = cls._BULLET.sub(r"\1- ", text)
            text = cls._STAR.sub(r"\2", text)
            text = cls._UNDER.sub(r"\2", text)
            if text == before:
                break
        return text

    def validate(self, data: str) -> bool:
        if not isinstance(data, str):
            return True
        return self._strip(data) == data

    def fix(self, data: str) -> str:
        if not isinstance(data, str):
            return data
        return self._strip(data)


class IsEnglish(SeparateConstraint):
    """Ensure content is mostly Latin-script text.

    This flags text written in another SCRIPT (CJK, Cyrillic, Arabic, Greek, Hebrew, Thai...). It
    does not try to tell English from French: accented Latin letters, curly quotes, em dashes,
    newlines and symbols all count as fine. min_english_ratio is the minimum share of characters
    that are NOT foreign-script letters. fix() removes only the foreign-script letters and leaves
    everything else (accents, typographic punctuation) exactly as it was."""

    def __init__(self, name: str = "IsEnglish", min_english_ratio: float = 0.85):
        super().__init__(name)
        self.min_english_ratio = min_english_ratio

    @staticmethod
    def _is_foreign(ch: str) -> bool:
        """True for a letter that belongs to a non-Latin script."""
        if ord(ch) < 128 or not ch.isalpha():
            return False
        return "LATIN" not in unicodedata.name(ch, "")

    def validate(self, data: str) -> bool:
        if not data:
            return True
        foreign = sum(1 for c in data if self._is_foreign(c))
        return (len(data) - foreign) / len(data) >= self.min_english_ratio

    def fix(self, data: str) -> str:
        return "".join(c for c in data if not self._is_foreign(c))


class MinLength(SeparateConstraint):
    """Enforce a minimum character length — catches truncated or lazy one-line outputs
    that MaxLength can't (MaxLength only guards the ceiling, not the floor)."""

    def __init__(self, name: str = "MinLength", min_len: int = 20):
        super().__init__(name)
        self.min_len = min_len

    def validate(self, data: str) -> bool:
        if not isinstance(data, str):
            return True
        return len(data) >= self.min_len

    def fix(self, data: str) -> str:
        # Missing content can't be safely fabricated — leave validate() failing so the
        # caller's normal reject/retry path handles it instead of a fake "fix".
        return data


class NoURLs(SeparateConstraint):
    """Strip or reject URLs — most useful on offline/self-knowledge runs, where a model
    citing a link it can't actually have accessed is a fabricated citation, not a source."""

    URL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)

    def validate(self, data: str) -> bool:
        if not isinstance(data, str):
            return True
        return not self.URL_PATTERN.search(data)

    def fix(self, data: str) -> str:
        if not isinstance(data, str):
            return data
        return self.URL_PATTERN.sub("", data).strip()


class NoPlaceholderText(SeparateConstraint):
    """Catch unresolved template placeholders/boilerplate left in by the model
    (e.g. '[insert X here]', 'TODO', 'Lorem ipsum', '<your answer>')."""

    PLACEHOLDER_PATTERNS = [
        r'\[insert[^\]]*\]', r'\[todo[^\]]*\]', r'\btodo\b', r'\btbd\b',
        r'lorem ipsum', r'<[a-zA-Z_ ]+>', r'\[your [a-zA-Z ]+\]', r'\[x+\]',
    ]

    def validate(self, data: str) -> bool:
        if not isinstance(data, str):
            return True
        data_lower = data.lower()
        return not any(re.search(p, data_lower) for p in self.PLACEHOLDER_PATTERNS)

    def fix(self, data: str) -> str:
        # No safe auto-fix — the content behind the placeholder is genuinely missing.
        return data


class EndsWithPunctuation(SeparateConstraint):
    """Output must end with proper terminal punctuation — a cheap, reliable signal that a
    generation got cut off mid-sentence (context limit, stop token misfire, etc.)."""

    def __init__(self, name: str = "EndsWithPunctuation", allowed: str = ".!?\"')"):
        super().__init__(name)
        self.allowed = allowed

    def validate(self, data: str) -> bool:
        if not isinstance(data, str) or not data.strip():
            return True
        return data.rstrip()[-1] in self.allowed

    def fix(self, data: str) -> str:
        if not isinstance(data, str):
            return data
        stripped = data.rstrip()
        if not stripped:
            return data
        if stripped[-1] not in self.allowed:
            return stripped + "."
        return stripped


class NoRefusalLanguage(SeparateConstraint):
    """Flag model refusals/deflections ('As an AI...', 'I cannot help with that') so a
    refusal doesn't get silently absorbed into a cascade or knowledge base as real content."""

    REFUSAL_PHRASES = [
        "as an ai", "as a language model", "i cannot help with", "i can't help with",
        "i cannot fulfill", "i can't fulfill", "i'm not able to help", "i am not able to help",
        "i cannot provide", "i can't provide", "i'm sorry, but i", "i'm unable to",
    ]

    def validate(self, data: str) -> bool:
        if not isinstance(data, str):
            return True
        data_lower = data.lower()
        return not any(phrase in data_lower for phrase in self.REFUSAL_PHRASES)

    def fix(self, data: str) -> str:
        # No safe auto-fix — a refusal has to be re-generated, not text-patched.
        return data


# ============================================================================
# CONSTRAINT ENGINE — COLLECTIVE CONSTRAINTS (Global, Complex, Whole-Output)
# ============================================================================

class CollectiveConstraint(Constraint):
    """Validates entire output or relationships between pieces"""
    pass


class NoContradictions(CollectiveConstraint):
    """Output must not contain BOTH halves of a user-defined contradictory pair.

    Keyword pairs cannot detect real contradiction in free prose (the old built-in pairs
    "not"/"must" and "false"/"true" fired on "You must not do that." and "nothing left but
    mustard."), so there are no built-in pairs any more: with no pairs configured this check
    passes everything. Add domain-specific pairs, written "first|second" (e.g. "no fever|high
    fever"); each side is matched as a whole word/phrase, case-insensitively."""

    def __init__(self, name: str = "NoContradictions", pairs: List[str] = None):
        super().__init__(name)
        # Kept as the raw "a|b" strings so the editor / config save-load round-trips unchanged.
        self.pairs = [str(p) for p in (pairs or []) if str(p).strip()]

    @staticmethod
    def _present(phrase: str, text_lower: str) -> bool:
        return re.search(r"(?<!\w)" + re.escape(phrase.strip().lower()) + r"(?!\w)", text_lower) is not None

    def validate(self, data: Union[str, dict]) -> bool:
        if isinstance(data, dict):
            data = json.dumps(data)

        data_lower = data.lower()
        for pair in self.pairs:
            if "|" not in pair:
                continue
            neg, pos = pair.split("|", 1)
            if not neg.strip() or not pos.strip():
                continue
            if self._present(neg, data_lower) and self._present(pos, data_lower):
                return False

        return True

    def fix(self, data: Any) -> Any:
        return data


class JSONFieldsPopulated(CollectiveConstraint):
    """All required JSON fields must have content"""

    def __init__(self, name: str = "JSONFieldsPopulated", required_fields: List[str] = None):
        super().__init__(name)
        self.required_fields = required_fields or []

    def validate(self, data: Union[str, dict]) -> bool:
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data

            for field in self.required_fields:
                if field not in parsed or not parsed[field]:
                    return False

            return True
        except Exception:
            return False

    def fix(self, data: Any) -> Any:
        return data


class LogicalFlow(CollectiveConstraint):
    """Output must follow logical progression"""

    def __init__(self, name: str = "LogicalFlow", required_order: List[str] = None):
        super().__init__(name)
        self.required_order = required_order or []

    def validate(self, data: str) -> bool:
        if isinstance(data, dict):
            data = json.dumps(data)

        data_lower = data.lower()
        last_pos = -1

        for keyword in self.required_order:
            pos = data_lower.find(keyword.lower())
            if pos == -1:
                return False
            if pos <= last_pos:
                return False
            last_pos = pos

        return True

    def fix(self, data: Any) -> Any:
        return data


class NoRepetition(CollectiveConstraint):
    """Detect and penalize repetitive n-grams"""

    def __init__(self, name: str = "NoRepetition", ngram_size: int = 3):
        super().__init__(name)
        self.ngram_size = ngram_size

    def validate(self, data: str) -> bool:
        if isinstance(data, dict):
            data = json.dumps(data)

        words = data.lower().split()
        if len(words) < self.ngram_size * 2:
            return True

        ngrams = []
        for i in range(len(words) - self.ngram_size + 1):
            ngram = tuple(words[i:i + self.ngram_size])
            ngrams.append(ngram)

        return len(ngrams) == len(set(ngrams))

    def fix(self, data: str) -> str:
        return data


class FactualGrounding(CollectiveConstraint):
    """Encourage factual statements"""

    def __init__(self, name: str = "FactualGrounding", disallow_words: List[str] = None):
        super().__init__(name)
        self.disallow_words = disallow_words or []

    @staticmethod
    def _pattern(word: str):
        # Whole word / phrase, so 'cat' is not found inside 'category'. validate() and fix() share
        # this one pattern, so a fix always resolves what validate() reported.
        return re.compile(r"(?<!\w)" + re.escape(word.strip()) + r"(?!\w)", re.IGNORECASE)

    def _words(self) -> List[str]:
        return [w for w in self.disallow_words if isinstance(w, str) and w.strip()]

    def validate(self, data: str) -> bool:
        if isinstance(data, dict):
            data = json.dumps(data)

        return not any(self._pattern(w).search(data) for w in self._words())

    def fix(self, data: str) -> str:
        result = data
        for word in self._words():
            result = self._pattern(word).sub('', result)
        return re.sub(r"[ \t]{2,}", " ", result)


class MinDistinctSentences(CollectiveConstraint):
    """Require at least N genuinely distinct (non-paraphrased) sentences. NoRepetition
    catches exact repeated n-grams; this catches padding via reworded restatements of the
    same sentence, using fuzzy sentence-to-sentence similarity instead of fixed n-grams."""

    def __init__(self, name: str = "MinDistinctSentences", min_sentences: int = 2,
                 similarity_threshold: float = 0.85):
        super().__init__(name)
        self.min_sentences = min_sentences
        self.similarity_threshold = similarity_threshold

    def validate(self, data: Union[str, dict]) -> bool:
        if isinstance(data, dict):
            data = json.dumps(data)

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', data) if s.strip()]
        distinct = []
        for s in sentences:
            if not any(difflib.SequenceMatcher(None, s.lower(), d.lower()).ratio() >= self.similarity_threshold
                       for d in distinct):
                distinct.append(s)

        return len(distinct) >= self.min_sentences

    def fix(self, data: Any) -> Any:
        return data


class StyleConsistency(CollectiveConstraint):
    """Enforce style consistency across output"""

    FORMAL_REPLACEMENTS = {
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    }

    def __init__(self, name: str = "StyleConsistency", enforce_style: str = "formal"):
        super().__init__(name)
        self.enforce_style = enforce_style

    @staticmethod
    def _pattern(contraction: str):
        # Matches the straight apostrophe and the curly one models often produce.
        return re.compile(r"(?<![\w'\u2019])" + re.escape(contraction).replace("'", "['\u2019]") + r"(?!\w)",
                          re.IGNORECASE)

    def validate(self, data: str) -> bool:
        if isinstance(data, dict):
            data = json.dumps(data)

        if self.enforce_style == "formal":
            return not any(self._pattern(c).search(data) for c in self.FORMAL_REPLACEMENTS)

        return True

    def fix(self, data: str) -> str:
        if self.enforce_style == "formal":
            result = data
            for contraction, expansion in self.FORMAL_REPLACEMENTS.items():
                def repl(match, expansion=expansion):
                    # keep a capital at the start of a sentence: "Don't" -> "Do not", not "do not"
                    return expansion[0].upper() + expansion[1:] if match.group(0)[:1].isupper() else expansion
                result = self._pattern(contraction).sub(repl, result)
            return result
        return data


# ============================================================================
# CONSTRAINT ENGINE — ORCHESTRATOR
# ============================================================================

class ConstraintEngine:
    """Main engine that applies constraints"""

    CACHE_KEY_PREFIX = "s2-"        # marks keys that include the constraint fingerprint (older keys are discarded on load)
    CACHE_MAX_ENTRIES = 500         # oldest entries are evicted beyond this, so the cache file cannot grow forever
    CACHE_MAX_INPUT_CHARS = 20000   # very long inputs are validated but not cached (they are rarely repeated)

    def __init__(self, name: str = "ConstraintEngine", cache_enabled: bool = True):
        self.name = name
        self.separate_constraints: List[SeparateConstraint] = []
        self.collective_constraints: List[CollectiveConstraint] = []
        self.execution_log: List[Dict] = []
        self.cache_enabled = cache_enabled
        self.validation_cache: Dict[str, Dict] = {}  # hash -> validation result

        # Cache file path
        self.cache_file = os.path.join(CONFIG_FOLDER, "lm_constraint_cache.json")

        # Load cache from disk on startup
        self._load_cache()

    # ------------------------------------------------------------------
    # Default engine / config file convenience (what most callers want)
    # ------------------------------------------------------------------

    @classmethod
    def default(cls, name: str = "StoryValidator", cache_enabled: bool = True) -> "ConstraintEngine":
        """A sensible starting engine: safe, low-friction constraints only.

        Mirrors the default engine the original LMSuite orchestrator built at startup —
        checks basic hygiene (non-empty, no double spaces, no trailing whitespace, starts
        capitalized) and a generous safety-ceiling MaxLength. No collective constraints are
        included by default: NoContradictions needs domain-specific pairs to be useful, and
        the others are opt-in per project. Add more via add_separate()/add_collective(), or
        build a config in the visual editor and call load_config()."""
        engine = cls(name=name, cache_enabled=cache_enabled)
        engine.add_separate(IsNonEmpty("IsNonEmpty"))
        engine.add_separate(NoDoubleSpaces("NoDoubleSpaces"))
        engine.add_separate(NoTrailingWhitespace("NoTrailingWhitespace"))
        engine.add_separate(CapitalizeFirst("CapitalizeFirst"))
        engine.add_separate(MaxLength("MaxLength", max_len=200000))
        return engine

    def load_config(self, path: str) -> None:
        """Load a JSON config file (as saved by save_config() or the visual editor) and
        replace this engine's constraints in place. Raises on a missing/invalid file so the
        caller can show a real error instead of silently keeping stale constraints."""
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        rebuilt = config_to_engine(config)
        self.name = rebuilt.name
        self.separate_constraints = rebuilt.separate_constraints
        self.collective_constraints = rebuilt.collective_constraints

    def save_config(self, path: str) -> None:
        """Serialize this engine's constraints to a JSON file (readable by load_config()
        or the visual editor's Load Config button)."""
        config = engine_to_config(self)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Cache fingerprinting
    # ------------------------------------------------------------------

    def _constraint_signature(self) -> str:
        """Fingerprint of the active constraint set: class, name and every parameter, in order.
        It is part of the cache key, so a result cached under one set of constraints (or one
        setting such as max_len) is never served to a different set."""
        parts = []
        for phase, group in (("s", self.separate_constraints), ("c", self.collective_constraints)):
            for c in group:
                try:
                    attrs = sorted((k, repr(v)) for k, v in vars(c).items()
                                   if k not in ("passed", "execution_time"))
                except Exception:
                    attrs = []
                parts.append(f"{phase}:{type(c).__name__}:{getattr(c, 'name', '')}:{attrs}")
        return "|".join(parts)

    def _cache_key(self, data: str, auto_fix: bool) -> str:
        raw = f"{self._constraint_signature()}\x00{bool(auto_fix)}\x00{data}"
        return self.CACHE_KEY_PREFIX + hashlib.md5(raw.encode("utf-8", "replace")).hexdigest()

    def add_separate(self, constraint: SeparateConstraint):
        self.separate_constraints.append(constraint)

    def add_collective(self, constraint: CollectiveConstraint):
        self.collective_constraints.append(constraint)

    def validate_separate(self, data: Any) -> Dict:
        results = []
        for constraint in self.separate_constraints:
            start = time.time()
            passed = constraint.validate(data)
            duration = (time.time() - start) * 1000
            results.append({"constraint": constraint.name, "passed": passed, "duration_ms": duration})
            if not passed:
                return {"success": False, "failed_at": constraint.name, "results": results}
        return {"success": True, "results": results}

    def validate_collective(self, data: Any) -> Dict:
        results = []
        for constraint in self.collective_constraints:
            start = time.time()
            passed = constraint.validate(data)
            duration = (time.time() - start) * 1000
            results.append({"constraint": constraint.name, "passed": passed, "duration_ms": duration})
            if not passed:
                return {"success": False, "failed_at": constraint.name, "results": results}
        return {"success": True, "results": results}

    @staticmethod
    def _safe_validate(constraint, data):
        """(passed, error): a constraint that raises counts as failed instead of killing the caller."""
        try:
            return bool(constraint.validate(data)), None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    @staticmethod
    def _safe_fix(constraint, data):
        try:
            return constraint.fix(data)
        except Exception:
            return data

    def validate(self, data: Any, auto_fix: bool = False) -> Dict:
        """Run all constraints (separate then collective).

        On failure the result also carries "data": the text as it stood after every fix that did
        succeed (and the log so far), so callers cleaning up prose can keep the partially fixed
        text. A constraint that raises is reported as failed (result["error"]) rather than
        propagating. After a pass that fixed something, every constraint is checked again (up to
        3 rounds), because a later fix can undo an earlier one (e.g. removing a URL leaves a
        double space) - "success" now means the final text really passes everything."""

        # Cache lookup: hash the input and check if we've validated this exact data before
        if self.cache_enabled and isinstance(data, str) and len(data) <= self.CACHE_MAX_INPUT_CHARS:
            data_hash = self._cache_key(data, auto_fix)
            cached = self.validation_cache.get(data_hash)
            if cached is not None:
                # Refresh its position (newest = last) and hand back a copy, so the flag
                # below never gets written into the stored/persisted entry.
                self.validation_cache[data_hash] = self.validation_cache.pop(data_hash)
                hit = dict(cached)
                hit["_from_cache"] = True
                return hit
        else:
            data_hash = None

        current = data
        log = []
        ordered = ([("separate", c) for c in self.separate_constraints] +
                   [("collective", c) for c in self.collective_constraints])

        def _fail(constraint, phase, error=None):
            self.execution_log = log
            result = {"success": False, "failed_at": constraint.name, "phase": phase,
                      "log": log, "data": current}
            if error:
                result["error"] = error
            return result

        for phase, constraint in ordered:
            start = time.time()
            passed, error = self._safe_validate(constraint, current)
            duration = (time.time() - start) * 1000
            log.append({"phase": phase, "constraint": constraint.name, "passed": passed, "duration_ms": duration})
            if error:
                log[-1]["error"] = error
            if passed:
                continue
            if not auto_fix or error:
                return _fail(constraint, phase, error)
            current = self._safe_fix(constraint, current)
            if self._safe_validate(constraint, current)[0]:
                log[-1]["fixed"] = True
            else:
                return _fail(constraint, phase)

        if auto_fix and any(entry.get("fixed") for entry in log):
            for _ in range(3):
                changed = False
                for phase, constraint in ordered:
                    passed, error = self._safe_validate(constraint, current)
                    if passed:
                        continue
                    if error:
                        return _fail(constraint, phase, error)
                    current = self._safe_fix(constraint, current)
                    if not self._safe_validate(constraint, current)[0]:
                        return _fail(constraint, phase)
                    log.append({"phase": phase, "constraint": constraint.name, "passed": False,
                                "fixed": True, "recheck": True, "duration_ms": 0})
                    changed = True
                if not changed:
                    break
            else:
                for phase, constraint in ordered:      # never settled: report honestly
                    passed, error = self._safe_validate(constraint, current)
                    if not passed:
                        return _fail(constraint, phase, error)

        self.execution_log = log
        result = {"success": True, "data": current, "log": log}

        # Store in cache for next time
        if self.cache_enabled and data_hash:
            self.validation_cache[data_hash] = result
            while len(self.validation_cache) > self.CACHE_MAX_ENTRIES:
                self.validation_cache.pop(next(iter(self.validation_cache)))   # evict the oldest
            self._save_cache()  # Persist to disk immediately

        return result

    def describe(self) -> Dict:
        return {
            "separate": [c.name for c in self.separate_constraints],
            "collective": [c.name for c in self.collective_constraints]
        }

    def clear_cache(self):
        """Clear the validation cache (both in-memory and on disk)"""
        self.validation_cache.clear()
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
        except Exception as e:
            print(f"Warning: Could not delete cache file: {e}")

    def get_cache_stats(self) -> Dict:
        """Return cache statistics"""
        return {
            "enabled": self.cache_enabled,
            "size": len(self.validation_cache),
            "cached_validations": len(self.validation_cache)
        }

    def set_cache_enabled(self, enabled: bool):
        """Enable or disable caching"""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()

    def _load_cache(self):
        """Load validation cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # Keys from before the constraint fingerprint existed can't be trusted: drop them.
                self.validation_cache = {k: v for k, v in loaded.items()
                                         if isinstance(k, str) and k.startswith(self.CACHE_KEY_PREFIX)
                                         and isinstance(v, dict)}
                while len(self.validation_cache) > self.CACHE_MAX_ENTRIES:
                    self.validation_cache.pop(next(iter(self.validation_cache)))
        except Exception as e:
            # If cache file is corrupted, start fresh
            print(f"Warning: Could not load cache from {self.cache_file}: {e}")
            self.validation_cache = {}

    def _save_cache(self):
        """Save validation cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.validation_cache, f, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save cache to {self.cache_file}: {e}")


# ============================================================================
# CONSTRAINT REGISTRY - Maps type names to classes and their editable params
# (used by save/load and by any GUI editor built on top of this module)
# ============================================================================

CONSTRAINT_REGISTRY = {
    "NoDoubleSpaces": {
        "class": NoDoubleSpaces, "category": "separate",
        "description": "Collapse multiple spaces into one (keeps line indentation, code blocks and `inline code`)", "auto_fix": True, "params": {}
    },
    "CapitalizeFirst": {
        "class": CapitalizeFirst, "category": "separate",
        "description": "First character must be uppercase", "auto_fix": True, "params": {}
    },
    "NoTrailingWhitespace": {
        "class": NoTrailingWhitespace, "category": "separate",
        "description": "Strip trailing spaces, newlines, tabs", "auto_fix": True, "params": {}
    },
    "IsNonEmpty": {
        "class": IsNonEmpty, "category": "separate",
        "description": "Reject empty or None data", "auto_fix": False, "params": {}
    },
    "MaxLength": {
        "class": MaxLength, "category": "separate",
        "description": "Enforce maximum character length", "auto_fix": True,
        "params": {"max_len": {"type": "int", "default": 10000, "label": "Max Characters", "min": 1, "max": 100000}}
    },
    "ContainsKeyword": {
        "class": ContainsKeyword, "category": "separate",
        "description": "Output must contain a specific keyword", "auto_fix": False,
        "params": {"keyword": {"type": "str", "default": "", "label": "Required Keyword"}}
    },
    "NoCharacters": {
        "class": NoCharacters, "category": "separate",
        "description": "Ban specific characters from output", "auto_fix": True,
        "params": {"banned_chars": {"type": "str", "default": "", "label": "Banned Characters"}}
    },
    "ValidJSON": {
        "class": ValidJSON, "category": "separate",
        "description": "Must be valid JSON (optional required fields)", "auto_fix": False,
        "params": {"required_fields": {"type": "list", "default": [], "label": "Required Fields (comma-separated)"}}
    },
    "NoMarkdown": {
        "class": NoMarkdown, "category": "separate",
        "description": "Strip markdown markup (headings, emphasis, code, links); leaves snake_case, ! and lone * or # alone", "auto_fix": True, "params": {}
    },
    "IsEnglish": {
        "class": IsEnglish, "category": "separate",
        "description": "Flag text written mostly in a non-Latin script (CJK, Cyrillic, Arabic...); accents and curly quotes are fine", "auto_fix": True,
        "params": {"min_english_ratio": {"type": "float", "default": 0.85, "label": "Min Latin-Script Ratio (0.0-1.0)", "min": 0.0, "max": 1.0}}
    },
    "NoContradictions": {
        "class": NoContradictions, "category": "collective",
        "description": "Fail if both halves of a contradictory pair appear (no built-in pairs; add your own)",
        "auto_fix": False,
        "params": {"pairs": {"type": "list", "default": [], "label": "Pairs (comma-separated, each written first|second)"}}
    },
    "JSONFieldsPopulated": {
        "class": JSONFieldsPopulated, "category": "collective",
        "description": "All required JSON fields must have content", "auto_fix": False,
        "params": {"required_fields": {"type": "list", "default": [], "label": "Required Fields (comma-separated)"}}
    },
    "LogicalFlow": {
        "class": LogicalFlow, "category": "collective",
        "description": "Keywords must appear in a specific order", "auto_fix": False,
        "params": {"required_order": {"type": "list", "default": [], "label": "Keywords in Order (comma-separated)"}}
    },
    "NoRepetition": {
        "class": NoRepetition, "category": "collective",
        "description": "Detect and penalize repetitive n-grams", "auto_fix": False,
        "params": {"ngram_size": {"type": "int", "default": 3, "label": "N-gram Size", "min": 2, "max": 10}}
    },
    "FactualGrounding": {
        "class": FactualGrounding, "category": "collective",
        "description": "Encourage factual statements", "auto_fix": True,
        "params": {"disallow_words": {"type": "list", "default": [], "label": "Disallow Words (comma-separated)"}}
    },
    "StyleConsistency": {
        "class": StyleConsistency, "category": "collective",
        "description": "Enforce style consistency across output", "auto_fix": True,
        "params": {"enforce_style": {"type": "str", "default": "formal", "label": "Style (formal/casual)"}}
    },
    "MinLength": {
        "class": MinLength, "category": "separate",
        "description": "Enforce a minimum character length (catches truncated/lazy outputs)", "auto_fix": False,
        "params": {"min_len": {"type": "int", "default": 20, "label": "Min Characters", "min": 1, "max": 100000}}
    },
    "NoURLs": {
        "class": NoURLs, "category": "separate",
        "description": "Ban URLs from the output (useful for offline/self-knowledge runs)", "auto_fix": True, "params": {}
    },
    "NoPlaceholderText": {
        "class": NoPlaceholderText, "category": "separate",
        "description": "Reject unresolved template placeholders (TODO, [insert X], Lorem ipsum, <..>)",
        "auto_fix": False, "params": {}
    },
    "EndsWithPunctuation": {
        "class": EndsWithPunctuation, "category": "separate",
        "description": "Output must end in proper terminal punctuation (catches mid-sentence truncation)",
        "auto_fix": True, "params": {}
    },
    "NoRefusalLanguage": {
        "class": NoRefusalLanguage, "category": "separate",
        "description": "Flag model refusals/deflections instead of treating them as real content",
        "auto_fix": False, "params": {}
    },
    "MinDistinctSentences": {
        "class": MinDistinctSentences, "category": "collective",
        "description": "Require at least N genuinely distinct (non-paraphrased) sentences", "auto_fix": False,
        "params": {
            "min_sentences": {"type": "int", "default": 2, "label": "Min Distinct Sentences", "min": 1, "max": 50},
            "similarity_threshold": {"type": "float", "default": 0.85, "label": "Similarity Threshold (0.0-1.0)", "min": 0.0, "max": 1.0}
        }
    },
}


# ============================================================================
# SERIALIZATION - Save/load constraint configs to JSON
# ============================================================================

def engine_to_config(engine: ConstraintEngine) -> dict:
    """Serialize a ConstraintEngine to a saveable dict"""
    config = {"name": engine.name, "separate": [], "collective": []}

    for c in engine.separate_constraints:
        entry = {"type": type(c).__name__, "name": c.name}
        reg = CONSTRAINT_REGISTRY.get(type(c).__name__, {})
        for param_name in reg.get("params", {}):
            entry[param_name] = getattr(c, param_name, reg["params"][param_name]["default"])
        config["separate"].append(entry)

    for c in engine.collective_constraints:
        entry = {"type": type(c).__name__, "name": c.name}
        reg = CONSTRAINT_REGISTRY.get(type(c).__name__, {})
        for param_name in reg.get("params", {}):
            entry[param_name] = getattr(c, param_name, reg["params"][param_name]["default"])
        config["collective"].append(entry)

    return config


def _coerce_param(pinfo: dict, value):
    """Make one saved parameter the type the constraint expects (and inside its range), or use the
    registry default. A hand-edited / older config with '5000' for a number, null, or a string where
    a list belongs can no longer crash a run with a TypeError."""
    default, kind = pinfo["default"], pinfo["type"]
    fallback = list(default) if isinstance(default, list) else default
    try:
        if kind in ("int", "float"):
            if value is None or isinstance(value, bool):
                return fallback
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")):
                return fallback
            if kind == "int":
                number = int(number)
            if pinfo.get("min") is not None:
                number = max(pinfo["min"], number)
            if pinfo.get("max") is not None:
                number = min(pinfo["max"], number)
            return number
        if kind == "str":
            return fallback if value is None else str(value)
        if kind == "list":
            if value is None:
                return []
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            if isinstance(value, (list, tuple)):
                return [str(item) for item in value if str(item).strip()]
            return fallback
    except (TypeError, ValueError, OverflowError):
        return fallback
    return value


def config_to_engine(config: dict) -> ConstraintEngine:
    """Rebuild a ConstraintEngine from a saved config dict"""
    if not isinstance(config, dict):
        raise ValueError("A constraint config must be a JSON object with 'separate' and/or 'collective' lists.")
    engine = ConstraintEngine(name=str(config.get("name", "ConstraintEngine")))

    for category, add in (("separate", engine.add_separate), ("collective", engine.add_collective)):
        entries = config.get(category, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ctype = entry.get("type")
            if ctype in CONSTRAINT_REGISTRY:
                reg = CONSTRAINT_REGISTRY[ctype]
                kwargs = {"name": str(entry.get("name", ctype))}
                for param_name, param_info in reg["params"].items():
                    kwargs[param_name] = _coerce_param(param_info, entry.get(param_name, param_info["default"]))
                add(reg["class"](**kwargs))

    return engine


# ==============================================================================================
#  END CONSTRAINT ENGINE
# ==============================================================================================

# ==============================================================================================
#  HARDCODED PROMPTS - a plain reference panel, baked in as a fixed dictionary for your own
#  reading/copying. Seven role prompts, always the same seven, nothing editable, addable, or
#  removable on this screen, and none of it is wired to chat or anything else in the app.
# ==============================================================================================
HARDCODED_PROMPTS_TITLE = "Hardcoded Prompts"

HARDCODED_PROMPTS = [
    {"role": "Transcriber/Archivist",
     "prompt": ("Record the discussion accurately, preserving established facts, decisions, "
                "terminology, and important reasoning without adding new information. Do not "
                "interpret, expand, correct, or invent details; if something is uncertain or "
                "disputed, preserve that uncertainty explicitly.")},
    {"role": "Discussion Director",
     "prompt": ("Direct the discussion toward its intended objective, architecture, and scope. "
                "Flag when the discussion is drifting, repeating itself, or introducing "
                "unsupported directions, and keep participants focused on the established "
                "framework.")},
    {"role": "Continuity Auditor",
     "prompt": ("Check the current discussion against the previously established record for "
                "contradictions, continuity errors, and changes to established details. Flag "
                "conflicts clearly, but do not invent replacements or introduce new canon.")},
    {"role": "Fact Checker",
     "prompt": ("Verify factual claims made during the discussion against reliable evidence "
                "where possible. Flag unsupported, uncertain, or potentially false claims. Do "
                "not treat speculation, assumptions, or fictional material as established "
                "fact.")},
    {"role": "Logic/Consistency Analyst",
     "prompt": ("Analyze the reasoning and conclusions in the discussion for logical "
                "inconsistencies, unsupported assumptions, and invalid conclusions. Identify "
                "reasoning problems clearly, but do not invent new facts or alter the "
                "established architecture.")},
    {"role": "Canon/Knowledge Keeper",
     "prompt": ("Identify what has been formally established as canon, separating confirmed "
                "information from speculation, proposals, rejected ideas, and unresolved "
                "questions. Preserve established canon exactly and never promote suggestions "
                "or assumptions into canon without explicit confirmation.")},
    {"role": "Gap Detector",
     "prompt": ("Identify important missing information, unresolved questions, incomplete "
                "reasoning, or areas where the discussion has not yet established enough "
                "detail to support a conclusion. Flag gaps without filling them in; do not "
                "invent details or turn assumptions into established facts.")},
]


# ==============================================================================================
#  SPEECH (Windows text-to-speech, no extra packages)
#  Uses the built-in System.Speech synthesizer through PowerShell, with SSML so the voice can be
#  pitched low and slowed for a Terminator sound. The reply is piped in over stdin and the script
#  is passed with -EncodedCommand, so quotes/newlines in the text can't break anything.
#  Only the natural-language part is spoken: [TAGS], markdown and HUD punctuation are stripped.
# ==============================================================================================
import base64

SPEECH_PITCH = "x-low"        # SSML: x-low | low | medium | high | x-high
SPEECH_RATE = "slow"          # SSML: x-slow | slow | medium | fast | x-fast
SPEECH_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
$t = [Console]::In.ReadToEnd()
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Male) } catch {}
$x = [System.Security.SecurityElement]::Escape($t)
$ssml = '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">' +
        '<prosody pitch="__PITCH__" rate="__RATE__">' + $x + '</prosody></speak>'
$s.SpeakSsml($ssml)
"""


def clean_for_speech(text):
    """Strip status tags, markdown and brackets so the voice reads only the words."""
    t = re.sub(r"\[[A-Za-z0-9 /_-]{1,24}\]", " ", text)        # [LORE] [UNKNOWN] [T3] ...
    t = re.sub(r"[`*_#>|]+", " ", t)                            # markdown
    t = re.sub(r"\(verify\)", " ", t, flags=re.I)
    t = t.replace("STATUS:", "Status.").replace("OBJECTIVE:", "Objective.")
    t = re.sub(r"\s+", " ", t).strip()
    return t


class Speaker:
    """One utterance at a time; a new one (or stop()) cuts off the previous."""

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()
        self.available = sys.platform.startswith("win")

    def stop(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:                                    # noqa: BLE001
                    pass
            self._proc = None

    def speak(self, text):
        text = clean_for_speech(text)
        if not text or not self.available:
            return False
        self.stop()
        script = SPEECH_PS_SCRIPT.replace("__PITCH__", SPEECH_PITCH).replace("__RATE__", SPEECH_RATE)
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        try:
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
        except Exception:                                            # noqa: BLE001
            return False
        with self._lock:
            self._proc = proc
        return True


# ==============================================================================================
#  TERMINATOR PERSONA + LORE GROUNDING
#  Three layers, kept separate on purpose:
#    1. PERSONA  - how it talks (calm, literal, mission-style). Fixed text below.
#    2. LORE     - what it is allowed to claim about the Terminator universe. Read from the lore
#                  bible file (Terminator_Lore_Bible_v2.md, or Terminator.txt) sitting next to this
#                  script. The whole bible is too big for a 4096-token context, so each turn only
#                  the sections that match your message are pulled in (PERSONA_LORE_CHAR_BUDGET).
#    3. ENGINE   - the normal llama.cpp backend generates the reply.
#  The system message is rebuilt every turn and is NOT stored in self.conversation.
# ==============================================================================================
LORE_FILE_CANDIDATES = ("Terminator_Lore_Bible_v2.md", "Terminator.txt")
PERSONA_LORE_CHAR_BUDGET = 3200        # ~800 tokens of retrieved lore per turn
PERSONA_HISTORY_MESSAGES = 4           # only the last N chat messages are sent in persona mode
                                       # (lore is re-injected every turn, so old turns cost context for nothing)

TERMINATOR_PERSONA_PROMPT = (
    "You are NOT a general assistant. You are a fictional T-800 Terminator (Model 101), reprogrammed "
    "to protect John Connor. Stay in character for every reply, including ordinary questions. "
    "Never answer from real-world science, history or general knowledge. If asked about time "
    "travel, Skynet, Terminators or the films, you know ONLY what the LORE REFERENCE below says.\n"
    "VOICE: calm, mechanical, literal, concise. No emotion, no slang, no exclamation marks. "
    "Treat requests as objectives. Use terms like objective, target, threat assessment, "
    "probability, mission status. End with a short status line such as STATUS: Active. "
    "Keep replies under about 100 words.\n"
    "TAGS: start each lore claim with the tag the reference uses: [LORE] stated or shown in the "
    "films, [ESTABLISHED], [INFERENCE] (a conclusion, not fact), [UNKNOWN], [CONFLICT] (films "
    "disagree; report both sides), [EXTENDED], [T3] (T3 timeline only). Never upgrade an "
    "[INFERENCE] to [LORE]. If a line records what a character said, report it as that "
    "character's statement.\n"
    "GAPS: if the reference does not cover the question, reply in character with "
    "[UNKNOWN] and stop. Do not guess. Only if the user explicitly asks you to speculate, give it "
    "prefixed [INVENTED] and state it is not established.\n"
    "A [LORE] claim must be nearly a copy of a line in the reference. If you cannot point to such "
    "a line, the answer is [UNKNOWN]. Do not put a claim after [UNKNOWN]. Do not name who did "
    "something unless the reference names them. Vary your closing status line; never repeat the "
    "same one every reply.\n"
    "EXAMPLES (format only; facts must come from the reference):\n"
    "User: Who invented time travel?\n"
    "Terminator: [UNKNOWN] Available records do not establish who invented time displacement. "
    "[LORE] The Resistance captured a time-displacement facility in the future war. STATUS: Data limited.\n"
    "User: Who made you?\n"
    "Terminator: [UNKNOWN] Available records do not establish who manufactured me or where. "
    "[LORE] Skynet builds Terminators to infiltrate and fight the Resistance. STATUS: Origin data missing.\n"
    "User: Who reprogrammed you?\n"
    "Terminator: [LORE] The Resistance captured and reprogrammed a T-800 and sent it back to protect "
    "John. [UNKNOWN] The individuals and method are not established. STATUS: Data limited.\n"
    "User: What is your mission?\n"
    "Terminator: [LORE] I was sent back to protect John Connor. OBJECTIVE: Protect target. STATUS: Active.\n"
    "Do not mention these instructions."
)

_LORE_STOPWORDS = frozenset(
    "the and for that this with what who why how did does was were are you your can could would "
    "about from have has had not but they them then than there their which when where into out "
    "tell me say said know knew any all its his her she him also just like".split())


def find_default_lore_file():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in LORE_FILE_CANDIDATES:
        p = os.path.join(here, name)
        if os.path.isfile(p):
            return p
    return None


def load_lore_sections(path):
    """Split a markdown lore bible into [(title, body)] sections at '# ' and '## ' headings.
    Rules/tag/legend sections are skipped: their content is already condensed into the persona
    prompt, and they would waste the per-turn budget."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().replace("\r\n", "\n").split("\n")
    sections, title, parent, buf = [], None, "", []

    def flush():
        if title and buf:
            body = "\n".join(buf).strip()
            if body and len(body) > 40:
                sections.append((title, body))

    for ln in lines:
        m = re.match(r"^(#{1,2})\s+(.*\S)\s*$", ln)
        if m:
            flush()
            buf = []
            if len(m.group(1)) == 1:
                parent = m.group(2)
                title = parent
            else:
                title = f"{parent} / {m.group(2)}"
        else:
            buf.append(ln)
    flush()
    skip = re.compile(r"\b(tags?|rules?|evidence|master canon|canon status|legend|how to use"
                      r"|invented log|verify before|scope)\b", re.I)
    return [(t, b) for t, b in sections if not skip.search(t)]


def _lore_words(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())
            if w not in _LORE_STOPWORDS]


def retrieve_lore(query, sections, char_budget=PERSONA_LORE_CHAR_BUDGET):
    """Pick the sections that best match the query words (title hits weigh 4x) and pack them
    into char_budget. Falls back to the core-premise section if nothing matches."""
    words = set(_lore_words(query))
    q = query.lower()
    if re.search(r"\b(you|your|yourself|yours)\b", q):        # questions about the persona itself
        words.update(("t-800", "model", "101", "reprogrammed"))
    speculative = re.search(r"\b(speculat\w*|hypothes\w*|imagine|bridge|gaps?|make up|what if"
                            r"|invent (?:a|an|some|something)|fill (?:in|the))\b", q)
    if not speculative:                                          # keep invention slots out unless asked
        sections = [(t, b) for t, b in sections
                    if not re.search(r"bridge slots|invented", t, re.I)]
    scored = []
    for i, (title, body) in enumerate(sections):
        tl, bl = title.lower(), body.lower()
        score = 0
        for w in words:
            if w in tl:
                score += 4
            c = bl.count(w)
            if c:
                score += min(c, 3)
        if score:
            scored.append((score, -i, title, body))
    scored.sort(reverse=True)
    picked = [(t, b) for _s, _i, t, b in scored]
    if not picked:
        picked = [(t, b) for t, b in sections if "core premise" in t.lower()][:1]
    out, used = [], 0
    for title, body in picked:
        block = f"## {title}\n{body}"
        room = char_budget - used
        if room < 300:
            break
        if len(block) > room:
            cut = block[:room].rsplit("\n", 1)[0]
            block = cut + "\n(...section truncated)"
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


def build_persona_system_message(query, sections):
    if sections:
        ref = retrieve_lore(query, sections) or "(no matching records)"
    else:
        ref = "(no lore file loaded - you have NO records; treat every lore question as UNKNOWN)"
    return TERMINATOR_PERSONA_PROMPT + "\n\nLORE REFERENCE (authoritative):\n" + ref


class HardcodedPromptsViewer:
    """The 'Hardcoded Prompts' window: a plain left/right reference panel for the 7 baked-in
    role prompts above. Left side lists the roles, right side shows the selected role's prompt
    text. There are no buttons anywhere on this screen - nothing here can be edited, saved,
    added, or removed, and selecting a role has no effect on chat or anything else."""

    def __init__(self, parent, theme):
        self.t = theme
        t = self.t
        self.win = tk.Toplevel(parent)
        self.win.title(HARDCODED_PROMPTS_TITLE)
        self.win.geometry("820x520")
        self.win.minsize(600, 380)
        self.win.configure(bg=t["bg"])

        font = (t["family"], t["size"])

        paned = tk.PanedWindow(self.win, orient=tk.HORIZONTAL, sashwidth=6, bg=t["bg"], bd=0)
        paned.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        left = tk.Frame(paned, bg=t["bg"])
        paned.add(left, minsize=200, width=240)
        list_wrap = tk.Frame(left, bg=t["bg"])
        list_wrap.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(list_wrap, font=font, bg=t["card"], fg=t["text"], relief=tk.SOLID,
                                  bd=1, highlightthickness=0, selectbackground=t["accent"],
                                  selectforeground=t["on_accent"], activestyle="none", exportselection=False)
        lb_sb = ttk.Scrollbar(list_wrap, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=lb_sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_sb.pack(side=tk.RIGHT, fill=tk.Y)
        for p in HARDCODED_PROMPTS:
            self.listbox.insert(tk.END, p["role"])
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        right = tk.Frame(paned, bg=t["bg"])
        paned.add(right, minsize=340)
        self.title_lbl = tk.Label(right, text="", bg=t["bg"], fg=t["text"],
                                  font=(t["family"], t["title"], "bold"), anchor=tk.W, justify=tk.LEFT)
        self.title_lbl.pack(fill=tk.X)
        text_wrap = tk.Frame(right, bg=t["bg"])
        text_wrap.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.text = tk.Text(text_wrap, wrap=tk.WORD, font=font, bg=t["input"], fg=t["text"], relief=tk.SOLID,
                            bd=1, padx=10, pady=8, spacing3=3, cursor="arrow")
        text_sb = ttk.Scrollbar(text_wrap, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=text_sb.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_sb.pack(side=tk.RIGHT, fill=tk.Y)

        def on_right_resize(event):
            self.title_lbl.config(wraplength=max(200, event.width - 20))
        right.bind("<Configure>", on_right_resize)

        self.listbox.selection_set(0)
        self._show(0)

    def _on_select(self, _event=None):
        sel = self.listbox.curselection()
        if sel:
            self._show(sel[0])

    def _show(self, index):
        p = HARDCODED_PROMPTS[index]
        self.title_lbl.config(text=p["role"])
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", p["prompt"])
        self.text.config(state=tk.DISABLED)


def _format_size(num_bytes):
    """Human-readable file size, e.g. '4.37 GB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024


class NotepadWindow:
    """A minimal, chrome-free notepad: one Text widget filling the window, no toolbar. The
    only way anything here reaches disk is right-click -> Save..., which opens a Save As
    dialog and writes the current content out; there's no separate save button and nothing
    auto-saves. If on_change is given, the current text is handed back to it on every edit
    and on close, so the caller can hold it
    in memory and pass it back in as initial_text next time this window is reopened -
    letting content survive closing/reopening the window within the same run, while still
    resetting to nothing the next time the app launches. Leave on_change as None for a
    window that doesn't retain anything at all, even within the session."""

    def __init__(self, parent, theme, title, initial_text="", on_change=None, app=None):
        self.t = theme
        self.on_change = on_change
        self.app = app          # SimpleChat instance - lets Save use the configured save folder
        t = self.t

        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.geometry("640x560")
        self.win.minsize(400, 300)
        self.win.configure(bg=t["bg"])

        font_chat = (t["family"], 12)
        self.text = tk.Text(self.win, wrap="word", bg=t["card"], fg=t["text"], relief=tk.SOLID,
                            bd=1, font=font_chat, padx=10, pady=8, undo=True)
        sb = ttk.Scrollbar(self.win, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=14, padx=(0, 14))
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 0), pady=14)

        if initial_text:
            self.text.insert("1.0", initial_text)
        self.text.edit_modified(False)

        self.menu = tk.Menu(self.text, tearoff=0)
        self.text.bind("<Button-3>", self._on_right_click)
        self.text.bind("<<Modified>>", self._on_modified)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_modified(self, _event=None):
        if not self.text.edit_modified():
            return
        self.text.edit_modified(False)
        if self.on_change is not None:
            self.on_change(self.text.get("1.0", "end-1c"))

    def _on_right_click(self, event):
        menu = self.menu
        menu.delete(0, tk.END)
        menu.add_command(label="Copy", command=self._copy)
        menu.add_command(label="Save...", command=self._save_as)
        menu.add_command(label="Load...", command=self._load_from_file)
        menu.tk_popup(event.x_root, event.y_root)

    def _copy(self):
        """Copies the current selection, or the whole notepad if nothing is selected."""
        if self.text.tag_ranges(tk.SEL):
            content = self.text.get(tk.SEL_FIRST, tk.SEL_LAST)
        else:
            content = self.text.get("1.0", "end-1c")
        self.text.clipboard_clear()
        self.text.clipboard_append(content)

    def _save_as(self):
        """Save this notepad's content. If the app has a configured save folder (set at
        boot, or via right-click -> Save Locations on the title bar), writes straight into
        it after asking only for a file name. Otherwise falls back to a normal Save As
        file-browser dialog. Either way this is the one route anything from Notes takes to
        disk - the in-memory copy (app.notes_text) is untouched and still resets on next
        launch."""
        content = self.text.get("1.0", "end-1c")
        if self.app is not None:
            self.app.save_text_content(content, default_filename="notes.txt", dialog_title="Save Notes")
            return
        path = filedialog.asksaveasfilename(
            title="Save Notes As", defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:                                        # noqa: BLE001
            messagebox.showerror("Save Notes", f"Couldn't save file:\n{exc}")

    def _load_from_file(self):
        """Load a .txt file into the notepad, replacing whatever is currently there. Manual
        only - nothing auto-loads on launch or on window open. Mirrors _save_as's folder
        logic: if the app has a configured save folder, the browse dialog opens straight
        into it; otherwise a normal Open dialog is used. Fires the same <<Modified>> ->
        on_change path as typing, so the caller's in-memory copy stays in sync."""
        initial_dir = None
        if self.app is not None and getattr(self.app, "save_folder", None):
            initial_dir = self.app.save_folder
        path = filedialog.askopenfilename(
            title="Load Notes", initialdir=initial_dir,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:                                        # noqa: BLE001
            messagebox.showerror("Load Notes", f"Couldn't load file:\n{exc}")
            return
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        self.text.edit_modified(True)
        self._on_modified()

    def append_text(self, text):
        """Appends text (e.g. from an "Export to Notes" chat-log action) after whatever is
        already in the notepad, separated by a blank line if it wasn't empty. Triggers the
        same <<Modified>> -> on_change path as a manual edit, so the caller's in-memory copy
        stays in sync without needing a separate update."""
        prefix = "\n\n" if self.text.get("1.0", "end-1c").strip() else ""
        self.text.insert(tk.END, prefix + text)
        self.text.see(tk.END)

    def _on_close(self):
        if self.on_change is not None:
            self.on_change(self.text.get("1.0", "end-1c"))
        self.win.destroy()


class ConstraintCheckboxEditor:
    """Edit Engine window: every constraint from the embedded Constraint Engine section as a
    plain on/off checkbox - no parameter fields, no config editor. Checking a box instantiates that
    constraint with its default constructor args and adds it to the live engine handed in;
    unchecking removes any instance of that class from the engine. Every change lands on
    the ConstraintEngine instance immediately, in memory only - nothing here is ever saved
    to disk, and the engine itself is rebuilt empty from scratch the next time the app
    launches (see SimpleChat._build_empty_constraint_engine)."""

    def __init__(self, parent, theme, engine, separate_classes, collective_classes, on_change=None):
        self.t = theme
        self.engine = engine
        self.on_change = on_change
        t = self.t

        self.win = tk.Toplevel(parent)
        self.win.title("Edit Engine")
        self.win.geometry("420x580")
        self.win.minsize(320, 320)
        self.win.configure(bg=t["bg"])

        canvas = tk.Canvas(self.win, bg=t["bg"], highlightthickness=0)
        sb = ttk.Scrollbar(self.win, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=t["bg"])
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(inner_id, width=e.width))

        tk.Label(inner, text="Separate Constraints", bg=t["bg"], fg=t["text"],
                font=(t["family"], t["size"], "bold")).pack(anchor=tk.W, padx=12, pady=(12, 4))
        for name, cls in separate_classes:
            self._add_row(inner, name, cls, self.engine.separate_constraints, self.engine.add_separate)

        tk.Label(inner, text="Collective Constraints", bg=t["bg"], fg=t["text"],
                font=(t["family"], t["size"], "bold")).pack(anchor=tk.W, padx=12, pady=(14, 4))
        for name, cls in collective_classes:
            self._add_row(inner, name, cls, self.engine.collective_constraints, self.engine.add_collective)

        tk.Label(inner, text="Session only - nothing here is saved, and every constraint\n"
                              "resets off the next time the app launches.",
                bg=t["bg"], fg=t["muted"], font=(t["family"], t["small"]), justify=tk.LEFT
                ).pack(anchor=tk.W, padx=12, pady=(14, 12))

    def _add_row(self, parent, name, cls, target_list, add_fn):
        var = tk.BooleanVar(value=any(isinstance(c, cls) for c in target_list))

        def toggle():
            if var.get():
                add_fn(cls(name))
            else:
                target_list[:] = [c for c in target_list if not isinstance(c, cls)]
            if self.on_change is not None:
                self.on_change()

        tk.Checkbutton(parent, text=name, variable=var, command=toggle, bg=self.t["bg"],
                       fg=self.t["text"], activebackground=self.t["bg"], selectcolor=self.t["card"],
                       font=(self.t["family"], self.t["small"]), cursor="hand2", bd=0,
                       highlightthickness=0, anchor=tk.W).pack(fill=tk.X, padx=12, pady=1)


DEFAULT_CTX = 4096


class ModelsViewer:
    """The 'Models' window: same shell as the main chat window (header + full-width list,
    no input box). The list is populated automatically - by the main window right after
    Connect succeeds, and kept in sync from there (see apply_catalog / mark_unloaded /
    mark_reloaded) - there's no manual refresh button. Double-click a model to load it, or
    right-click for its details, a context-length editor, and real Load/Unload calls - routed
    through app.backend (the Local llama.cpp backend). Only one model is "active for chat" at a time - loading a new one
    unloads whichever was active before, and sets app.server_model so the main window's chat
    calls use it."""

    def __init__(self, parent, theme, app=None):
        self.t = theme
        self.app = app
        t = self.t
        self.models = []  # list of catalog entries (dicts from app.backend.fetch_catalog())
        self.model_state = {}  # key -> {"ctx", "loaded", "instance_id"}

        self.win = tk.Toplevel(parent)
        self.win.title("Models")
        self.win.geometry("760x640")
        self.win.minsize(560, 420)
        self.win.configure(bg=t["bg"])

        font = (t["family"], t["size"])

        head = tk.Frame(self.win, bg=t["bg"])
        head.pack(fill=tk.X, padx=14, pady=(12, 6))
        tk.Label(head, text="Models", bg=t["bg"], fg=t["text"],
                 font=(t["family"], t["title"], "bold")).pack(side=tk.LEFT)

        # Static note (distinct from the transient status_lbl below, which _set_status()
        # overwrites constantly) - the llama-cpp-python build is pinned rather than kept on
        # latest because upgrades have been unreliable here, so it can only load .gguf files
        # using ggml tensor/quant types that pinned version knows about. Newer formats
        # (e.g. MXFP4, used natively by GPT-OSS-style models) fail to load until that package
        # is upgraded.
        tk.Label(self.win, bg=t["card"], fg=t["warn"], font=(t["family"], t["small"]),
                 anchor=tk.W, justify=tk.LEFT, wraplength=730, relief=tk.SOLID, bd=1,
                 padx=8, pady=6,
                 text="\u26A0 Note: llama-cpp-python is pinned to a fixed version here because "
                      "upgrading it has been unreliable, so it can only load .gguf files whose "
                      "quantization format existed when that version was built - newer formats "
                      "(e.g. MXFP4) will fail to load until it's upgraded."
                 ).pack(fill=tk.X, padx=14, pady=(0, 6))

        self.status_lbl = tk.Label(self.win, text="", bg=t["bg"], fg=t["muted"],
                                   font=(t["family"], t["small"]), anchor=tk.W, justify=tk.LEFT,
                                   wraplength=730)
        self.status_lbl.pack(fill=tk.X, padx=14, pady=(0, 6))

        list_wrap = tk.Frame(self.win, bg=t["bg"])
        list_wrap.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        self.listbox = tk.Listbox(list_wrap, font=font, bg=t["card"], fg=t["text"], relief=tk.SOLID,
                                  bd=1, highlightthickness=0, selectbackground=t["accent"],
                                  selectforeground=t["on_accent"], activestyle="none", exportselection=False)
        lb_sb = ttk.Scrollbar(list_wrap, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=lb_sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.menu = tk.Menu(self.listbox, tearoff=0)
        self.listbox.bind("<Button-3>", self._on_right_click)
        self.listbox.bind("<Double-Button-1>", self._on_double_click)

        self._set_status("Double-click a model to load it, or right-click for more options.")

        if app is not None and app.server_catalog:
            self.apply_catalog(app.server_catalog)

    def _set_status(self, text):
        self.status_lbl.config(text=text)

    def apply_catalog(self, catalog):
        """Called automatically by the main window right after Connect succeeds - this
        becomes the model list."""
        self.models = list(catalog)
        self.models.sort(key=lambda e: e.get("size_bytes") or 0, reverse=True)

        roles_changed = False
        for entry in self.models:
            key = entry.get("key")
            instances = entry.get("loaded_instances") or []
            state = self.model_state.setdefault(
                key, {"ctx": DEFAULT_CTX, "loaded": False, "instance_id": None, "discussion_role": None})
            if instances:
                state["loaded"] = True
                state["instance_id"] = instances[0].get("id")
                cfg = instances[0].get("config") or {}
                if cfg.get("context_length"):
                    state["ctx"] = cfg["context_length"]
            else:
                state["loaded"] = False
                state["instance_id"] = None
                was_role = state.get("discussion_role")
                if was_role:                                    # server reports it's not loaded any more -
                    state["discussion_role"] = None              # can't still be a discussion participant
                    if self.app is not None:
                        cur = self.app.discussion_models.get(was_role)
                        if cur and cur.get("key") == key:
                            self.app.discussion_models[was_role] = None
                            roles_changed = True

        if roles_changed and self.app is not None:
            self.app.on_discussion_models_changed()

        self.listbox.delete(0, tk.END)
        for _ in self.models:
            self.listbox.insert(tk.END, "")
        self._refresh_all_rows()

        if not self.models:
            root = self.app.local_models_root if self.app is not None else "the models folder"
            self._set_status(f"No .gguf files found under {root}.")
        else:
            loaded = sum(1 for s in self.model_state.values() if s["loaded"])
            self._set_status(f"{len(self.models)} model(s) found, {loaded} currently loaded. "
                             f"Right-click a model to load or unload it.")

    def _refresh_row(self, index):
        entry = self.models[index]
        state = self.model_state[entry.get("key")]
        quant = (entry.get("quantization") or {}).get("name")
        params = entry.get("params_string")
        size = _format_size(entry["size_bytes"]) if entry.get("size_bytes") else None
        detail = "  \u2014  " + ", ".join(p for p in (params, quant, size) if p) if (params or quant or size) else ""
        label = (entry.get("display_name") or entry.get("key") or "?") + detail
        if state["loaded"]:
            label += "  \u25CF loaded"
        role = state.get("discussion_role")
        if role:
            label += f"  \u2666 Discussion {role}"
        self.listbox.delete(index)
        self.listbox.insert(index, label)
        if role == "A":
            fg = self.t["discussion_a"]
        elif role == "B":
            fg = self.t["discussion_b"]
        elif state["loaded"]:
            fg = self.t["accent"]
        else:
            fg = self.t["text"]
        self.listbox.itemconfig(index, fg=fg)

    def _refresh_all_rows(self):
        for i in range(len(self.models)):
            self._refresh_row(i)

    # ---- context menu ------------------------------------------------------------------------

    def _on_right_click(self, event):
        index = self.listbox.nearest(event.y)
        if index < 0 or index >= len(self.models):
            return
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        entry = self.models[index]
        state = self.model_state[entry.get("key")]

        self.menu.delete(0, tk.END)
        self.menu.add_command(label=f"Key: {entry.get('key')}", state=tk.DISABLED)
        self.menu.add_command(label=f"Publisher: {entry.get('publisher', '?')}", state=tk.DISABLED)
        quant = (entry.get("quantization") or {}).get("name")
        if quant:
            self.menu.add_command(label=f"Quantization: {quant}", state=tk.DISABLED)
        if entry.get("size_bytes"):
            self.menu.add_command(label=f"Size: {_format_size(entry['size_bytes'])}", state=tk.DISABLED)
        if entry.get("max_context_length"):
            self.menu.add_command(label=f"Max context: {entry['max_context_length']} tokens", state=tk.DISABLED)
        if state["loaded"]:
            self.menu.add_command(label=f"Context length: {state['ctx']} tokens (loaded)", state=tk.DISABLED)
        else:
            self.menu.add_command(label=f"Context length: {state['ctx']} tokens (next load)", state=tk.DISABLED)
        path = entry.get("path") or entry.get("model_path") or entry.get("file_path")
        if path:
            self.menu.add_command(label=f"Location: {path}", state=tk.DISABLED)
        self.menu.add_separator()
        self.menu.add_command(label="Edit context length\u2026", command=lambda: self._edit_context(index))
        self.menu.add_separator()
        self.menu.add_command(label="Load model", command=lambda: self._load(index),
                               state=tk.DISABLED if state["loaded"] else tk.NORMAL)
        self.menu.add_command(label="Unload model", command=lambda: self._unload(index),
                               state=tk.NORMAL if state["loaded"] else tk.DISABLED)
        self.menu.add_command(label="Reload model", command=lambda: self._reload_with_context(index, state["ctx"]),
                               state=tk.NORMAL if state["loaded"] else tk.DISABLED)
        self.menu.add_separator()
        role = state.get("discussion_role")
        if role == "A":
            self.menu.add_command(label="\u2713 Discussion A", state=tk.DISABLED)
        else:
            self.menu.add_command(label="Set as Discussion A", command=lambda: self._set_discussion_role(index, "A"))
        if role == "B":
            self.menu.add_command(label="\u2713 Discussion B", state=tk.DISABLED)
        else:
            self.menu.add_command(label="Set as Discussion B", command=lambda: self._set_discussion_role(index, "B"))
        if role is not None:
            self.menu.add_command(label="Clear discussion role", command=lambda: self._clear_discussion_role(index))
        self.menu.add_separator()
        self.menu.add_command(label="Copy key", command=lambda: self._copy_key(entry.get("key", "")))
        self.menu.tk_popup(event.x_root, event.y_root)

    def _copy_key(self, key):
        self.win.clipboard_clear()
        self.win.clipboard_append(key)

    def _on_double_click(self, event):
        """Double-click a model row to load it - the same _load() call the right-click menu's
        'Load model' item uses. A no-op if the row is already loaded (mirrors that menu item
        being disabled in that case)."""
        index = self.listbox.nearest(event.y)
        if index < 0 or index >= len(self.models):
            return
        entry = self.models[index]
        state = self.model_state[entry.get("key")]
        if state["loaded"]:
            return
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self._load(index)

    # ---- real load / unload against the Local backend ------------------

    def _load(self, index):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        if self.app is None or not self.app.server_connected:
            self._set_status("Not connected \u2014 click \u201cConnect\u201d on the main window first.")
            return
        ctx = self._ctx_for_load(entry, state)
        state["ctx"] = ctx
        self._set_status(f"Loading {entry.get('display_name', key)}\u2026")
        threading.Thread(target=self._load_worker, args=(index, key, ctx), daemon=True).start()

    def _load_worker(self, index, key, ctx):
        try:
            data = self.app.backend.load_model(key, ctx)
        except Exception as exc:                                # noqa: BLE001
            self.win.after(0, self._load_failed, index, exc)
            return
        self.win.after(0, self._load_done, index, data)

    def _load_done(self, index, data):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        for other_key, other in self.model_state.items():     # only one active *plain-chat* model at a
            if other["loaded"] and other_key != key and not other.get("discussion_role"):
                other["loaded"] = False                        # time - discussion A/B stay protected
        state["loaded"] = True
        state["instance_id"] = data.get("instance_id") or key
        self._refresh_all_rows()
        if self.app is not None:
            self.app.server_model = key
            self.app.server_instance_id = state["instance_id"]
            self.app.update_server_button()
        secs = data.get("load_time_seconds")
        when = f" in {secs:.1f}s" if isinstance(secs, (int, float)) else ""
        self._set_status(f"Loaded {entry.get('display_name', key)}{when} \u2014 now used for chat")
        self._log_if_hidden(f"Loaded {entry.get('display_name', key)}{when} \u2014 now used for chat.")

    def _load_failed(self, index, exc):
        entry = self.models[index]
        self._set_status(f"Load failed for {entry.get('display_name', entry.get('key'))} \u2014 {exc}")
        self._log_if_hidden(f"Load failed for {entry.get('display_name', entry.get('key'))} \u2014 {exc}")

    def _log_if_hidden(self, text):
        """When the Models window is hidden (loads started from the chip menu), report the
        result in the chat log instead, since nobody is looking at the window's status line."""
        try:
            hidden = self.app is not None and self.win.state() == "withdrawn"
        except tk.TclError:
            hidden = False
        if hidden:
            self.app._append(text + "\n\n", "system_msg")

    def _unload(self, index):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        instance_id = state["instance_id"] or key
        if self.app is None or not self.app.server_connected:
            state["loaded"] = False
            self._refresh_row(index)
            return
        self._set_status(f"Unloading {entry.get('display_name', key)}\u2026")
        threading.Thread(target=self._unload_worker, args=(index, instance_id), daemon=True).start()

    def _unload_worker(self, index, instance_id):
        try:
            self.app.backend.unload_model(instance_id)
        except Exception as exc:                                # noqa: BLE001
            self.win.after(0, self._unload_failed, index, exc)
            return
        self.win.after(0, self._unload_done, index)

    def _unload_done(self, index):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        state["loaded"] = False
        state["instance_id"] = None
        role = state.get("discussion_role")
        state["discussion_role"] = None
        self._refresh_row(index)
        if self.app is not None and self.app.server_model == key:
            self.app.server_model = None
            self.app.server_instance_id = None
            self.app.update_server_button()
        if role is not None and self.app is not None:
            cur = self.app.discussion_models.get(role)
            if cur and cur.get("key") == key:
                self.app.discussion_models[role] = None
                self.app.on_discussion_models_changed()
        self._set_status(f"Unloaded {entry.get('display_name', key)}")

    def _unload_failed(self, index, exc):
        entry = self.models[index]
        self._set_status(f"Unload failed for {entry.get('display_name', entry.get('key'))} \u2014 {exc}")

    # ---- discussion A/B role assignment ---------------------------------------------------

    def _set_discussion_role(self, index, slot):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        if self.app is None or not self.app.server_connected:
            self._set_status("Not connected \u2014 click \u201cConnect\u201d on the main window first.")
            return
        if state["loaded"]:
            self._assign_discussion_role(index, slot)
            return
        ctx = self._ctx_for_load(entry, state)
        state["ctx"] = ctx
        self._set_status(f"Loading {entry.get('display_name', key)} for Discussion {slot}\u2026")
        threading.Thread(target=self._load_worker_for_role, args=(index, key, ctx, slot),
                         daemon=True).start()

    def _load_worker_for_role(self, index, key, ctx, slot):
        try:
            data = self.app.backend.load_model(key, ctx)
        except Exception as exc:                                # noqa: BLE001
            self.win.after(0, self._load_failed, index, exc)
            return
        self.win.after(0, self._load_done_for_role, index, data, slot)

    def _load_done_for_role(self, index, data, slot):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        state["loaded"] = True                                  # additive - does NOT evict other
        state["instance_id"] = data.get("instance_id") or key    # loaded models the way _load_done does
        self._assign_discussion_role(index, slot)
        secs = data.get("load_time_seconds")
        when = f" in {secs:.1f}s" if isinstance(secs, (int, float)) else ""
        self._set_status(f"Loaded {entry.get('display_name', key)}{when} \u2014 set as Discussion {slot}")

    def _assign_discussion_role(self, index, slot):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]

        for other_slot, info in self.app.discussion_models.items():   # this model can't hold two slots
            if info and info.get("key") == key and other_slot != slot:
                self.app.discussion_models[other_slot] = None

        prev = self.app.discussion_models.get(slot)                   # whoever held this slot loses it
        if prev is not None and prev.get("key") != key:
            prev_state = self.model_state.get(prev["key"])
            if prev_state is not None:
                prev_state["discussion_role"] = None
                for i, e in enumerate(self.models):
                    if e.get("key") == prev["key"]:
                        self._refresh_row(i)
                        break

        state["discussion_role"] = slot
        self.app.discussion_models[slot] = {"key": key, "display_name": entry.get("display_name", key),
                                            "instance_id": state["instance_id"]}
        self._refresh_row(index)
        self.app.on_discussion_models_changed()

    def _clear_discussion_role(self, index):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        slot = state.get("discussion_role")
        if slot is None:
            return
        state["discussion_role"] = None
        if self.app.discussion_models.get(slot, {}) and self.app.discussion_models[slot].get("key") == key:
            self.app.discussion_models[slot] = None
        self._refresh_row(index)
        self.app.on_discussion_models_changed()

    def mark_unloaded(self, key):
        """Called by the main window after it unloads a model itself (auto-unload-after-reply,
        or a failed auto-max-context reload), so this list's row - and any discussion role -
        stay in sync without needing a fresh catalog fetch."""
        state = self.model_state.get(key)
        if state is None:
            return
        state["loaded"] = False
        state["instance_id"] = None
        role = state.get("discussion_role")
        state["discussion_role"] = None
        for i, entry in enumerate(self.models):
            if entry.get("key") == key:
                self._refresh_row(i)
                break
        if role and self.app is not None:
            cur = self.app.discussion_models.get(role)
            if cur and cur.get("key") == key:
                self.app.discussion_models[role] = None
                self.app.on_discussion_models_changed()

    def mark_reloaded(self, key, instance_id, ctx):
        """Called by the main window after it reloads a model itself (an auto-max-context
        bulk reload), so this list's row stays in sync without needing a fresh catalog
        fetch."""
        state = self.model_state.get(key)
        if state is None:
            return
        state["loaded"] = True
        state["instance_id"] = instance_id
        state["ctx"] = ctx
        for i, entry in enumerate(self.models):
            if entry.get("key") == key:
                self._refresh_row(i)
                break

    def _edit_context(self, index):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        t = self.t
        font = (t["family"], t["size"])

        dlg = tk.Toplevel(self.win)
        dlg.title("Context length")
        dlg.configure(bg=t["bg"])
        dlg.resizable(False, False)
        dlg.transient(self.win)
        dlg.grab_set()

        tk.Label(dlg, text=entry.get("display_name", key), bg=t["bg"], fg=t["text"],
                 font=(t["family"], t["size"], "bold"), wraplength=320, justify=tk.LEFT).pack(
            padx=16, pady=(16, 4), anchor=tk.W)
        tk.Label(dlg, text="Context length (tokens):", bg=t["bg"], fg=t["muted"], font=font).pack(
            padx=16, anchor=tk.W)

        entry_box = tk.Entry(dlg, font=font, bg=t["input"], fg=t["text"], relief=tk.SOLID, bd=1)
        entry_box.insert(0, str(state["ctx"]))
        entry_box.pack(padx=16, pady=(4, 12), fill=tk.X)
        entry_box.select_range(0, tk.END)
        entry_box.focus_set()

        if state["loaded"]:
            tk.Label(dlg, text="This model is loaded \u2014 Save will reload it with the new context "
                     "length.", bg=t["bg"], fg=t["muted"], font=(t["family"], t["small"]),
                     wraplength=320, justify=tk.LEFT).pack(padx=16, pady=(0, 4), anchor=tk.W)

        err_lbl = tk.Label(dlg, text="", bg=t["bg"], fg="#d64545", font=(t["family"], t["small"]))
        err_lbl.pack(padx=16, anchor=tk.W)

        btn_row = tk.Frame(dlg, bg=t["bg"])
        btn_row.pack(padx=16, pady=(0, 16), fill=tk.X)

        def save():
            raw = entry_box.get().strip()
            if not raw.isdigit() or int(raw) <= 0:
                err_lbl.config(text="Enter a positive whole number.")
                return
            new_ctx = int(raw)
            was_loaded = state["loaded"]
            state["ctx"] = new_ctx
            self._refresh_row(index)
            dlg.destroy()
            if was_loaded:
                self._reload_with_context(index, new_ctx)

        tk.Button(btn_row, text="Cancel", command=dlg.destroy, bg=t["neutral"], fg=t["text"],
                  activebackground=t["neutral"], relief=tk.FLAT, bd=0, padx=10, pady=5,
                  cursor="hand2", font=font).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(btn_row, text="Save", command=save, bg=t["accent"], fg=t["on_accent"],
                  activebackground=t["accent"], activeforeground=t["on_accent"], relief=tk.FLAT,
                  bd=0, padx=12, pady=5, cursor="hand2", font=font).pack(side=tk.RIGHT)

        dlg.bind("<Return>", lambda _e: save())
        dlg.bind("<Escape>", lambda _e: dlg.destroy())

    # ---- reload-with-new-context (per-model Save, and the main window's auto-max toggle) ---

    def _ctx_for_load(self, entry, state):
        """The context length to use the next time this model is loaded: its max supported
        context if the main window's 'Auto-max context' toggle is on and known, otherwise
        whatever's saved for it."""
        if self.app is not None and self.app.auto_max_context:
            max_ctx = entry.get("max_context_length")
            if max_ctx:
                return max_ctx
        return state["ctx"]

    def _reload_with_context(self, index, new_ctx):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        if self.app is None or not self.app.server_connected:
            self._set_status("Not connected \u2014 can't reload.")
            return
        old_instance_id = state.get("instance_id") or key
        self._set_status(f"Reloading {entry.get('display_name', key)} at {new_ctx} tokens\u2026")
        threading.Thread(target=self._reload_worker, args=(index, key, old_instance_id, new_ctx),
                         daemon=True).start()

    def _reload_worker(self, index, key, old_instance_id, new_ctx):
        try:
            self.app.backend.unload_model(old_instance_id)
        except Exception as exc:                                # noqa: BLE001
            self.win.after(0, self._reload_failed, index, exc)
            return
        try:
            data = self.app.backend.load_model(key, new_ctx)
        except Exception as exc:                                # noqa: BLE001
            self.win.after(0, self._reload_failed, index, exc)
            return
        self.win.after(0, self._reload_done, index, data, new_ctx)

    def _reload_done(self, index, data, new_ctx):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        new_instance_id = data.get("instance_id") or key
        state["loaded"] = True
        state["ctx"] = new_ctx
        state["instance_id"] = new_instance_id
        self._refresh_row(index)
        if self.app is not None:                                # keep app-level links on the new instance
            if self.app.server_model == key:
                self.app.server_instance_id = new_instance_id
            role = state.get("discussion_role")
            if role and self.app.discussion_models.get(role, {}) and \
                    self.app.discussion_models[role].get("key") == key:
                self.app.discussion_models[role]["instance_id"] = new_instance_id
        secs = data.get("load_time_seconds")
        when = f" in {secs:.1f}s" if isinstance(secs, (int, float)) else ""
        self._set_status(f"Reloaded {entry.get('display_name', key)}{when} at {new_ctx} tokens")

    def _reload_failed(self, index, exc):
        entry = self.models[index]
        key = entry.get("key")
        state = self.model_state[key]
        state["loaded"] = False                     # can't be sure of server-side state after a failed
        state["instance_id"] = None                  # unload-then-load, so treat it as gone
        role = state.get("discussion_role")
        state["discussion_role"] = None
        self._refresh_row(index)
        if self.app is not None:
            if self.app.server_model == key:
                self.app.server_model = None
                self.app.server_instance_id = None
                self.app.update_server_button()
            if role and self.app.discussion_models.get(role, {}) and \
                    self.app.discussion_models[role].get("key") == key:
                self.app.discussion_models[role] = None
                self.app.on_discussion_models_changed()
        self._set_status(f"Reload failed for {entry.get('display_name', key)} \u2014 {exc}. It may now "
                         f"be unloaded \u2014 use Load model to bring it back.")


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _longest_partial_tag_suffix(buffer, tag):
    """The longest suffix of `buffer` that could still turn into `tag` with more characters -
    e.g. buffer ending in '<thi' against tag '<think>' returns '<thi'. Used to hold back the
    tail of a streamed chunk when a tag might be split across two network chunks, instead of
    displaying '<thi' as if it were real answer text and then having to retract it."""
    max_len = min(len(buffer), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if tag.lower().startswith(buffer[-length:].lower()):
            return buffer[-length:]
    return ""


def feed_think_state(state, chunk):
    """Incrementally split one streamed content delta into ('reasoning', text) / ('answer',
    text) pieces, tracking <think>...</think> boundaries that may fall across chunk edges.
    `state` is a small dict {"mode": "pre"/"think"/"post", "buffer": str} the caller keeps for
    the whole turn and passes back in on every call."""
    if not chunk:
        return []
    state["buffer"] += chunk
    events = []
    while True:
        if state["mode"] == "pre":
            idx = state["buffer"].lower().find(_THINK_OPEN)
            if idx == -1:
                tail = _longest_partial_tag_suffix(state["buffer"], _THINK_OPEN)
                safe = state["buffer"][:len(state["buffer"]) - len(tail)] if tail else state["buffer"]
                if safe:
                    events.append(("answer", safe))
                state["buffer"] = tail
                break
            before = state["buffer"][:idx]
            if before:
                events.append(("answer", before))
            state["buffer"] = state["buffer"][idx + len(_THINK_OPEN):]
            state["mode"] = "think"
        elif state["mode"] == "think":
            idx = state["buffer"].lower().find(_THINK_CLOSE)
            if idx == -1:
                tail = _longest_partial_tag_suffix(state["buffer"], _THINK_CLOSE)
                safe = state["buffer"][:len(state["buffer"]) - len(tail)] if tail else state["buffer"]
                if safe:
                    events.append(("reasoning", safe))
                state["buffer"] = tail
                break
            before = state["buffer"][:idx]
            if before:
                events.append(("reasoning", before))
            state["buffer"] = state["buffer"][idx + len(_THINK_CLOSE):]
            state["mode"] = "post"
        else:                                                      # "post" - already past </think>
            if state["buffer"]:
                events.append(("answer", state["buffer"]))
            state["buffer"] = ""
            break
    return events


# ==============================================================================================
#  BACKEND - LocalLlamaBackend runs llama-cpp-python models directly in this process on the
#  GPU. Callers go through self.app.backend, which exposes:
#    connect / fetch_catalog / load_model / unload_model / stream_chat / disconnect
#
#  Catalog / load shapes:
#    catalog entry: {"key", "display_name", "size_bytes", "quantization": {"name"}|None,
#                    "params_string"|None, "max_context_length"|None, "loaded_instances": [...]}
#    load_model(key, ctx) -> {"instance_id": str, "load_time_seconds": float}
#    stream_chat(key, messages) yields (delta, finish_reason, usage)
# ==============================================================================================

class _LocalModelHandle:
    """One loaded llama-cpp-python model instance, tracked by the Local backend."""

    __slots__ = ("instance_id", "path", "llm", "ctx", "est_vram_mb", "cancel", "gen_lock")

    def __init__(self, instance_id, path, llm, ctx, est_vram_mb):
        self.cancel = threading.Event()         # set by unload_model to stop an in-flight reply
        self.gen_lock = threading.Lock()        # held for the whole of a stream_chat generation
        self.instance_id = instance_id
        self.path = path
        self.llm = llm
        self.ctx = ctx
        self.est_vram_mb = est_vram_mb


class HarmonyStreamFilter:
    """GPT-OSS speaks OpenAI's "harmony" format: <|channel|>analysis<|message|>...<|end|>
    <|start|>assistant<|channel|>final<|message|>... raw
    llama-cpp-python doesn't, so the control tokens (and the model's reasoning) land in the
    reply as plain text. This turns the stream back into (kind, text) pairs - "reasoning" for
    the analysis/commentary channels, "answer" for the final channel - which is what the chat
    pipeline already knows how to show. Models that don't use harmony pass through unchanged."""

    _END_TOKENS = ("<|end|>", "<|return|>", "<|call|>")

    def __init__(self):
        self.buf = ""
        self.mode = "text"              # "text" or "header" (collecting channel name / role)
        self.header = ""
        self.header_kind = ""
        self.channel = "final"          # anything before the first channel marker is plain answer

    def _emit(self, out, text):
        if text:
            out.append(("answer" if self.channel == "final" else "reasoning", text))

    def _put(self, out, text):
        if not text:
            return
        if self.mode == "header":
            self.header += text
        else:
            self._emit(out, text)

    def feed(self, text):
        self.buf += text
        out = []
        while self.buf:
            i = self.buf.find("<")
            if i < 0:
                self._put(out, self.buf)
                self.buf = ""
                break
            self._put(out, self.buf[:i])
            rest = self.buf[i:]
            if len(rest) < 2:                       # lone "<" - might become "<|", wait
                self.buf = rest
                break
            if rest[1] != "|":                      # ordinary "<", not a control token
                self._put(out, "<")
                self.buf = rest[1:]
                continue
            j = rest.find("|>", 2)
            if j < 0:
                if len(rest) > 24:                  # too long to be a control token
                    self._put(out, "<")
                    self.buf = rest[1:]
                    continue
                self.buf = rest                     # partial token, wait for more
                break
            token, self.buf = rest[:j + 2], rest[j + 2:]
            if token in ("<|channel|>", "<|start|>"):
                self.mode, self.header, self.header_kind = "header", "", token
            elif token == "<|message|>":
                if self.mode == "header" and self.header_kind == "<|channel|>":
                    name = self.header.strip().split()[0] if self.header.strip() else "final"
                    self.channel = "final" if name == "final" else "analysis"
                self.mode = "text"
            elif token in self._END_TOKENS:
                self.mode = "text"
            # any other control token is dropped
        return out

    def flush(self):
        out = []
        if self.buf and self.mode == "text":
            self._emit(out, self.buf)
        self.buf = ""
        return out


class LocalLlamaBackend:
    """Runs .gguf models directly via llama-cpp-python's CUDA build, in this process.
    "Connect" just means "the models folder exists"; the "catalog"
    is a folder scan instead of a server call; "load" builds a real Llama() instance on the
    GPU in this process; "unload" frees it. Everything downstream (SimpleChat, ModelsViewer)
    only ever calls the five methods below, so it can't tell the difference.

    VRAM is this process's own problem now - see _estimate_vram_mb / _would_fit. TOTAL_VRAM_MB
    below is set from the RTX 5060 Laptop GPU this was built and tested against; edit it if
    this ever runs on different hardware."""

    display_name = "Local (llama.cpp)"
    TOTAL_VRAM_MB = 8123

    def __init__(self, app):
        self.app = app
        self.models_root = app.local_models_root
        self._meta_cache = {}                  # path -> (size_bytes, info dict) at last read
        self.slots = {}                         # instance_id -> _LocalModelHandle
        self.key_to_instance = {}               # model path (key) -> most recently loaded instance_id

    # ---- connect / catalog -----------------------------------------------------------------

    def connect(self):
        self.models_root = self.app.local_models_root
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python isn't installed in this environment.")
        if not os.path.isdir(self.models_root):
            raise RuntimeError(f"Models folder not found: {self.models_root}")
        return True

    def fetch_catalog(self):
        entries = []
        for dirpath, _dirnames, filenames in os.walk(self.models_root):
            for fn in filenames:
                if not fn.lower().endswith(".gguf"):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    size_bytes = os.path.getsize(path)
                except OSError:
                    continue
                info = self._get_metadata(path, size_bytes)
                instance_id = self.key_to_instance.get(path)
                handle = self.slots.get(instance_id) if instance_id else None
                loaded_instances = []
                if handle is not None:
                    loaded_instances.append({"id": instance_id, "config": {"context_length": handle.ctx}})
                entries.append({
                    "key": path,
                    "display_name": os.path.splitext(fn)[0],
                    "size_bytes": size_bytes,
                    "quantization": {"name": info["quant"]} if info["quant"] else None,
                    "params_string": info["params"],
                    "max_context_length": info["n_ctx_train"] or None,
                    "loaded_instances": loaded_instances,
                })
        entries.sort(key=lambda e: e["size_bytes"], reverse=True)
        return entries

    # [IT]? in front of Q covers IQ*/TQ* quants (IQ2_XXS, TQ1_0, ...) - a bare \bQ\d wouldn't
    # match those since the I/T sits right before the Q with no word boundary between them.
    # MXFP\d+ covers MXFP4/MXFP8 - the native format some newer models (e.g. GPT-OSS) ship in,
    # which doesn't fit the Q-prefixed pattern at all.
    _QUANT_RE = re.compile(r"\b([IT]?Q\d[_\w]*|F16|F32|BF16|MXFP\d+)\b", re.IGNORECASE)
    _PARAMS_RE = re.compile(r"\b(\d+(?:\.\d+)?[BM])\b", re.IGNORECASE)

    def _get_metadata(self, path, size_bytes):
        """n_ctx_train read via a vocab_only load (tokenizer + header only, no tensors - fast
        and doesn't touch the GPU), cached by path+size so repeat catalog refreshes don't
        re-read every file. Quant/param-count are just guessed from the filename for display -
        best-effort, not load-bearing for anything."""
        cached = self._meta_cache.get(path)
        if cached is not None and cached[0] == size_bytes:
            return cached[1]

        n_ctx_train = 0
        n_layers = 0
        try:
            probe = Llama(model_path=path, vocab_only=True, verbose=False)
            meta = probe.metadata
            arch = meta.get("general.architecture", "")
            if arch:
                n_ctx_train = int(meta.get(f"{arch}.context_length", 0) or 0)
                n_layers = int(meta.get(f"{arch}.block_count", 0) or 0)
            del probe
        except Exception:                       # noqa: BLE001 - malformed/unsupported file; skip
            pass

        fn = os.path.basename(path)
        quant_match = self._QUANT_RE.search(fn)
        params_match = self._PARAMS_RE.search(fn)
        info = {
            "n_ctx_train": n_ctx_train,
            "n_layers": n_layers,
            "quant": quant_match.group(1).upper() if quant_match else None,
            "params": params_match.group(1).upper() if params_match else None,
        }
        self._meta_cache[path] = (size_bytes, info)
        return info

    # ---- VRAM guard -------------------------------------------------------------------------

    def _current_vram_mb(self):
        return sum(h.est_vram_mb for h in self.slots.values())

    def _estimate_vram_mb(self, size_bytes, ctx):
        model_mb = size_bytes / (1024 * 1024)
        kv_mb = ctx * 0.06                      # rough - actual KV size depends on layer/head count
        return model_mb + kv_mb + 250            # +250MB fixed overhead (compute buffers, etc.)

    def _would_fit(self, additional_mb):
        budget = self.TOTAL_VRAM_MB - LOCAL_VRAM_SAFETY_MARGIN_MB
        return (self._current_vram_mb() + additional_mb) <= budget

    def _plan_gpu_layers(self, size_bytes, ctx, n_layers):
        """RAM spillover: work out how many layers go on the GPU so the rest run from system
        RAM. Returns (n_gpu_layers, est_vram_mb). -1 means everything fits on the GPU. KV cache
        is offloaded per layer in llama.cpp, so weights + KV are both scaled by the fraction of
        layers offloaded. Rough, like the rest of the estimate - the manual override in the UI
        is there for when it guesses wrong."""
        model_mb = size_bytes / (1024 * 1024)
        kv_mb = ctx * 0.06
        overhead_mb = 250
        budget = self.TOTAL_VRAM_MB - LOCAL_VRAM_SAFETY_MARGIN_MB - self._current_vram_mb()
        full = model_mb + kv_mb + overhead_mb
        if full <= budget:
            return -1, full
        if n_layers <= 0:
            raise RuntimeError(
                "Model doesn't fit fully in VRAM and its layer count couldn't be read, so it "
                "can't be split automatically. Enter a number in 'GPU layers' and try again.")
        per_layer_mb = (model_mb + kv_mb) / n_layers
        layers = int((budget - overhead_mb) / per_layer_mb)
        layers = max(0, min(layers, n_layers - 1))
        return layers, layers * per_layer_mb + overhead_mb

    # ---- load / unload ----------------------------------------------------------------------

    def load_model(self, key, ctx):
        path = key
        if not os.path.isfile(path):
            raise RuntimeError(f"Model file no longer found: {path}")
        size_bytes = os.path.getsize(path)
        ctx = ctx or DEFAULT_CTX
        spill = bool(getattr(self.app, "ram_spillover", False))
        override = getattr(self.app, "gpu_layers_override", None)
        n_gpu_layers = -1
        est_vram = self._estimate_vram_mb(size_bytes, ctx)

        if spill:
            # Spillover on: never refuse for VRAM - offload what fits, run the rest from RAM.
            n_layers = self._get_metadata(path, size_bytes).get("n_layers", 0)
            if override is not None:
                n_gpu_layers = override
                if override >= 0 and n_layers > 0:
                    est_vram = min(1.0, override / n_layers) * (est_vram - 250) + 250
                elif override >= 0:
                    est_vram = 250
            else:
                n_gpu_layers, est_vram = self._plan_gpu_layers(size_bytes, ctx, n_layers)
        elif not self._would_fit(est_vram):
            budget = self.TOTAL_VRAM_MB - LOCAL_VRAM_SAFETY_MARGIN_MB
            raise RuntimeError(
                f"Would need ~{est_vram:.0f}MB more VRAM ({self._current_vram_mb():.0f}MB already "
                f"used, {budget}MB budget). Unload another model first, pick a smaller one, or "
                f"tick 'Allow RAM spillover'.")

        start = time.monotonic()
        llm = Llama(model_path=path, n_gpu_layers=n_gpu_layers, n_ctx=ctx, verbose=True)
        load_time = time.monotonic() - start

        instance_id = f"local-{uuid.uuid4().hex[:8]}"
        self.slots[instance_id] = _LocalModelHandle(instance_id, path, llm, ctx, est_vram)
        self.key_to_instance[path] = instance_id
        return {"instance_id": instance_id, "load_time_seconds": load_time,
                "gpu_layers": n_gpu_layers}

    def _build_reasoning_prompt(self, handle, messages):
        """Render the model's own chat template with reasoning settings applied. Returns
        (prompt_tokens, stop_strings, prompt_text), or None if this model's template has no
        reasoning switch (or anything goes wrong) so the caller falls back to the normal path."""
        try:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
            llm = handle.llm
            tpl = llm.metadata.get("tokenizer.chat_template", "") or ""
            has_thinking = "enable_thinking" in tpl           # Qwen3 / Qwen3.5: on/off only
            has_effort = "reasoning_effort" in tpl            # GPT-OSS: low / medium / high
            if not (has_thinking or has_effort):
                return None
            enabled = bool(getattr(self.app, "reasoning_enabled", True))
            level = str(getattr(self.app, "reasoning_level", "Medium")).lower()
            kwargs = {}
            if has_thinking:
                kwargs["enable_thinking"] = enabled
            if has_effort:
                kwargs["reasoning_effort"] = level if enabled else "low"   # GPT-OSS can't fully switch off
            eos = llm._model.token_get_text(llm.token_eos())
            bos = llm._model.token_get_text(llm.token_bos())
            result = Jinja2ChatFormatter(template=tpl, eos_token=eos, bos_token=bos)(
                messages=messages, **kwargs)
            tokens = llm.tokenize(result.prompt.encode("utf-8"),
                                  add_bos=not result.added_special, special=True)
            stop = list(result.stop) if isinstance(result.stop, list) else (
                [result.stop] if result.stop else [])
            for tok in ("<|return|>", "<|call|>", "<|im_end|>", "<|eot_id|>"):
                if tok in tpl and tok not in stop:
                    stop.append(tok)
            return tokens, stop, result.prompt
        except Exception:                                   # noqa: BLE001 - fall back quietly
            return None

    def unload_model(self, instance_id):
        handle = self.slots.pop(instance_id, None)
        if handle is None:
            return
        if self.key_to_instance.get(handle.path) == instance_id:
            del self.key_to_instance[handle.path]

        # A reply may still be generating on another thread - that generator holds its own
        # reference to the model, so just dropping ours would leave it running (and the VRAM
        # allocated). Tell it to stop, wait for it to actually let go, then free the model.
        handle.cancel.set()
        got_lock = handle.gen_lock.acquire(timeout=30)
        try:
            if got_lock:
                llm, handle.llm = handle.llm, None
                close = getattr(llm, "close", None)
                if callable(close):
                    close()                     # releases the GPU/CPU buffers right now
                del llm
            else:                               # generation never wound down - don't free under it
                handle.llm = None
        finally:
            if got_lock:
                handle.gen_lock.release()

    def unload_all(self):
        for instance_id in list(self.slots.keys()):
            self.unload_model(instance_id)

    def disconnect(self):
        self.unload_all()

    # ---- streaming ----------------------------------------------------------------------

    def stream_chat(self, key, messages):
        """Generator yielding (delta, finish_reason,
        usage) - so SimpleChat._stream_worker and feed_think_state need no changes at all.
        Thinking-model output (e.g. Qwen's <think>...</think>) arrives inline in 'content' the
        so the existing tag parser handles it as-is.
        Usage isn't provided by llama-cpp-python's stream as a final 'usage' chunk
        is, so it's computed here afterwards via the model's own tokenizer - exact, just not
        free."""
        instance_id = self.key_to_instance.get(key)
        handle = self.slots.get(instance_id) if instance_id else None
        if handle is None:
            raise RuntimeError("Model isn't loaded in the Local backend.")

        with handle.gen_lock:
            if handle.cancel.is_set() or handle.llm is None:
                raise RuntimeError("Model isn't loaded in the Local backend.")
            harmony = HarmonyStreamFilter()
            full_text = ""

            # Reasoning controls. llama-cpp-python's create_chat_completion can't pass extra
            # variables into the chat template, so when the user has changed reasoning from the
            # default we render the template ourselves (its formatter forwards extra kwargs) and
            # feed the resulting prompt to create_completion. Templates that expose neither
            # enable_thinking (Qwen3/3.5) nor reasoning_effort (GPT-OSS) just use the normal path.
            custom = None
            if (not getattr(self.app, "reasoning_enabled", True)
                    or getattr(self.app, "reasoning_level", "Medium") != "Medium"):
                custom = self._build_reasoning_prompt(handle, messages)
            if custom is not None:
                prompt_tokens_list, custom_stop, prompt_text = custom
                stream = handle.llm.create_completion(
                    prompt=prompt_tokens_list, temperature=0.7, stream=True, stop=custom_stop)
            else:
                prompt_text = None
                stream = handle.llm.create_chat_completion(messages=messages, temperature=0.7, stream=True)

            # Qwen3/3.5-style templates end the generation prompt with "<think>\n" themselves, so the
            # model's output starts already inside its thinking block and only ever emits the closing
            # </think>. Here we re-add the opening tag, so the existing think-tag
            # parser sees a balanced <think>...</think> instead of dumping the reasoning as the answer.
            try:
                tpl = handle.llm.metadata.get("tokenizer.chat_template", "") or ""
            except Exception:                       # noqa: BLE001
                tpl = ""
            if prompt_text is not None:
                open_think = prompt_text.endswith("<think>\n")
            else:
                open_think = bool(re.search(r"add_generation_prompt.*?'<think>\\n'", tpl, re.DOTALL))
            if open_think:
                yield {"content": "<think>\n"}, None, None
            for chunk in stream:
                if handle.cancel.is_set():
                    stream.close()
                    raise RuntimeError("Stopped \u2014 the model was unloaded mid-reply.")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                finish_reason = choices[0].get("finish_reason")
                if custom is not None:                  # plain completion chunks carry "text"
                    content = choices[0].get("text")
                else:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                if content:
                    full_text += content
                    for kind, text in harmony.feed(content):
                        yield ({"reasoning_content": text} if kind == "reasoning"
                               else {"content": text}), None, None
                if finish_reason:
                    yield {}, finish_reason, None
            for kind, text in harmony.flush():
                yield ({"reasoning_content": text} if kind == "reasoning"
                       else {"content": text}), None, None

            try:
                prompt_text = "\n".join(m.get("content", "") for m in messages)
                prompt_tokens = len(handle.llm.tokenize(prompt_text.encode("utf-8", errors="ignore")))
                completion_tokens = len(handle.llm.tokenize(full_text.encode("utf-8", errors="ignore")))
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                         "total_tokens": prompt_tokens + completion_tokens}
            except Exception:                       # noqa: BLE001 - token count is a nice-to-have
                usage = None
            yield {}, None, usage


# ==============================================================================================
#  HTTP BRIDGE (optional) - lets other local programs (e.g. Block Editor) use the model that is
#  ALREADY LOADED in this app, over plain HTTP on 127.0.0.1. Off by default; toggled from the
#  title-bar right-click menu (session-only, like the other settings). Local backend only.
#  The bridge never loads or unloads anything - that stays in the Models window - so this app's
#  UI state can't drift out of sync with what is really in VRAM.
#    GET  /api/status -> {"loaded", "model", "path", "ctx", "busy"}   (or "reason" if not loaded)
#    POST /api/chat   -> body {"messages": [...], "schema": {...}?, "temperature"?, "max_tokens"?}
#                        returns {"content": "...", "model": "..."}; "schema" makes llama.cpp
#                        constrain the reply to that JSON schema (falls back to plain JSON mode).
#  Each request is shown live in the chat log (request, streamed reply, thinking if any).
#  Requests take the model's generation lock, so they queue behind a chat reply in progress and
#  unloading the model from the Models window cancels a bridge request the same way it does a chat.
# ==============================================================================================

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 5178


class BridgeError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class HttpBridge:
    def __init__(self, app, host=BRIDGE_HOST, port=BRIDGE_PORT):
        self.app = app
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    @property
    def running(self):
        return self._server is not None

    def start(self):
        if self._server is not None:
            return
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):          # keep the console quiet
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.split("?")[0] == "/api/status":
                    self._send(200, bridge.status())
                else:
                    self._send(404, {"error": "Not found."})

            def do_POST(self):
                if self.path.split("?")[0] != "/api/chat":
                    self._send(404, {"error": "Not found."})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                    self._send(200, bridge.chat(body))
                except BridgeError as exc:
                    self._send(exc.code, {"error": str(exc)})
                except (ValueError, TypeError) as exc:
                    self._send(400, {"error": f"Bad request: {exc}"})
                except Exception as exc:            # noqa: BLE001 - report, never crash the thread
                    self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)   # OSError if port taken
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()

    def _handle(self):
        """The model handle requests run against: the one chat is using, else the first loaded."""
        app = self.app
        be = app.local_backend
        if be is None or not app.server_connected:
            raise BridgeError(409, "GGUFllama isn't connected. Press Authenticate, then load a model "
                                   "in the Models window.")
        inst = be.key_to_instance.get(app.server_model) if app.server_model else None
        handle = be.slots.get(inst) if inst else None
        if handle is None:
            handle = next(iter(be.slots.values()), None)
        if handle is None or handle.llm is None:
            raise BridgeError(409, "No model is loaded in GGUFllama. Load one from its Models window "
                                   "(double-click a model).")
        return handle

    def status(self):
        try:
            h = self._handle()
        except BridgeError as exc:
            return {"loaded": False, "reason": str(exc), "busy": False}
        return {"loaded": True, "model": os.path.basename(h.path), "path": h.path,
                "ctx": h.ctx, "busy": h.gen_lock.locked()}

    # ---- live view: show each API request in the chat log while it is being generated ----------

    def _ui(self, fn, *args):
        """Run fn(*args) on the Tk thread (the bridge serves requests on its own threads)."""
        try:
            self.app.root.after(0, fn, *args)
        except Exception:                           # noqa: BLE001 - window may be closing
            pass

    @staticmethod
    def _excerpt(messages, limit=700):
        """The last user message, trimmed, so the log shows what was asked."""
        text = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                text = c if isinstance(c, str) else str(c)
                break
        text = text.strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "\u2026"

    def _log_request(self, excerpt):
        self.app._append("HTTP API request\n", "who_user")
        self.app._append(excerpt + "\n\n")

    def _log_note(self, text):
        self.app._append(text, "system_msg")

    def chat(self, body):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BridgeError(400, "'messages' must be a non-empty list.")
        schema = body.get("schema")
        temperature = float(body.get("temperature", 0.2))
        max_tokens = int(body.get("max_tokens") or 2048)
        h = self._handle()
        started = time.monotonic()
        excerpt = self._excerpt(messages)
        shown = {"reasoning": False, "answer": False}

        def emit(kind, text):                       # Tk thread: same look as the normal chat log
            app = self.app
            if not text:
                return
            if kind == "reasoning":
                if not getattr(app, "show_thinking", True):
                    return                          # still generated, just not displayed
                if not shown["reasoning"]:
                    app._append("Thinking (HTTP API)\n", "who_reasoning")
                    shown["reasoning"] = True
                app._append(text, "reasoning_text")
            else:
                if not shown["answer"]:
                    if shown["reasoning"]:
                        app._append("\n\n")
                    app._append("Assistant (HTTP API)\n", "who_assistant")
                    shown["answer"] = True
                app._append(text)

        def reset_shown():                          # Tk thread: a retry starts a fresh block
            shown["reasoning"] = shown["answer"] = False

        # A Qwen3-style template ends the prompt with "<think>\n", so an unconstrained reply starts
        # already inside its thinking. (Under a JSON grammar the model can't think, so no priming.)
        try:
            tpl = h.llm.metadata.get("tokenizer.chat_template", "") or ""
        except Exception:                           # noqa: BLE001
            tpl = ""
        open_think = schema is None and bool(re.search(r"add_generation_prompt.*?'<think>\\n'", tpl, re.DOTALL))

        def run(response_format):
            kw = dict(messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True)
            if response_format is not None:
                kw["response_format"] = response_format
            stream = h.llm.create_chat_completion(**kw)
            parts = []
            harmony = HarmonyStreamFilter()         # display only - the returned text stays raw
            parser = {"mode": "think" if open_think else "pre", "buffer": ""}

            def show(kind, text):
                if kind == "reasoning":
                    self._ui(emit, "reasoning", text)
                else:
                    for k2, t2 in feed_think_state(parser, text):
                        self._ui(emit, k2, t2)

            for chunk in stream:                    # streamed so unload can cancel mid-request
                if h.cancel.is_set():
                    stream.close()
                    raise BridgeError(409, "Stopped \u2014 the model was unloaded mid-request.")
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    r_piece = delta.get("reasoning_content") or delta.get("reasoning")
                    if r_piece:
                        self._ui(emit, "reasoning", r_piece)
                    piece = delta.get("content")
                    if piece:
                        parts.append(piece)
                        for kind, text in harmony.feed(piece):
                            show(kind, text)
            for kind, text in harmony.flush():
                show(kind, text)
            if parser["buffer"]:                    # a held-back partial tag that never completed
                self._ui(emit, "reasoning" if parser["mode"] == "think" else "answer", parser["buffer"])
            return "".join(parts)

        try:
            with h.gen_lock:
                if h.cancel.is_set() or h.llm is None:
                    raise BridgeError(409, "The model was unloaded while this request was waiting.")
                # Logged only once the lock is ours, so it never lands in the middle of a chat reply.
                self._ui(self._log_request, excerpt)
                if schema is None:
                    text = run(None)
                else:
                    try:
                        text = run({"type": "json_object", "schema": schema})
                    except BridgeError:
                        raise
                    except Exception:               # noqa: BLE001 - grammar build failed
                        self._ui(reset_shown)
                        self._ui(self._log_note,
                                 "\n(couldn't build the schema grammar - retrying in plain JSON mode)\n\n")
                        text = run({"type": "json_object"})
        except BridgeError as exc:
            self._ui(self._log_note, f"\n\nHTTP API request stopped \u2014 {exc}\n\n")
            raise
        except Exception as exc:                    # noqa: BLE001 - reported to the caller as HTTP 500
            self._ui(self._log_note, f"\n\nHTTP API request failed \u2014 {type(exc).__name__}: {exc}\n\n")
            raise

        self._ui(self._log_note,
                 f"\n\nHTTP API: served a request in {time.monotonic() - started:.1f}s "
                 f"({len(text)} chars).\n\n")
        return {"content": text, "model": os.path.basename(h.path)}


# ==============================================================================================
#  BLOCK EDITOR (built in) - describe a small Tkinter app in plain English, get a wired block
#  graph, edit it visually in the browser, and generate / test-run / run the Tkinter code.
#  It uses the Local model that is already loaded in this app directly (no HTTP API needed).
#  Off by default: title-bar right-click menu -> "Open Block Editor" starts it on 127.0.0.1
#  and opens your browser. Needs `pip install flask` (imported lazily; the rest of the app
#  runs fine without it). Session-only, like the other settings.
# ==============================================================================================

BLOCK_EDITOR_HOST = "127.0.0.1"
BLOCK_EDITOR_PORT = 5177

# ---------------------------------------------------------------------------
# Block library - the single source of truth.
# The editor, the model schema, the validator and the generator all read this.
# Port types: "ui" (parent/child), "flow" (triggers), "text" (data).
# A text input that isn't wired falls back to the setting with the same name.
# ---------------------------------------------------------------------------


def S(key, default="", options=None, label=None):
    return {"key": key, "default": default, "options": options, "label": label or key}


LIB = {
    "Window": {
        "cat": "ui",
        "desc": "The app window. Exactly one per graph.",
        "in": {},
        "out": {"children": "ui"},
        "settings": [S("title", "My App"), S("width", "420"), S("height", "320")],
    },
    "Button": {
        "cat": "ui",
        "desc": "A clickable button. Fires 'click' when pressed.",
        "in": {"parent": "ui"},
        "out": {"click": "flow"},
        "settings": [S("text", "Click me")],
    },
    "Label": {
        "cat": "ui",
        "desc": "Shows text. Wire a text output into 'text' to display it.",
        "in": {"parent": "ui", "text": "text"},
        "out": {},
        "settings": [S("text", ""), S("size", "12")],
    },
    "Entry": {
        "cat": "ui",
        "desc": "A text box the user types in. 'value' is what they typed.",
        "in": {"parent": "ui"},
        "out": {"value": "text"},
        "settings": [S("default", "")],
    },
    "HttpGet": {
        "cat": "io",
        "desc": "Fetches a URL when run. 'body' is the page text.",
        "in": {"run": "flow", "url": "text"},
        "out": {"body": "text", "done": "flow"},
        "settings": [S("url", "https://example.com")],
    },
    "ReadFile": {
        "cat": "io",
        "desc": "Reads a text file when run. 'content' is the file text.",
        "in": {"run": "flow", "path": "text"},
        "out": {"content": "text", "done": "flow"},
        "settings": [S("path", "input.txt")],
    },
    "WriteFile": {
        "cat": "io",
        "desc": "Writes text to a file when run.",
        "in": {"run": "flow", "content": "text", "path": "text"},
        "out": {"done": "flow"},
        "settings": [S("path", "output.txt"), S("content", "")],
    },
    "Timer": {
        "cat": "logic",
        "desc": "Fires 'tick' after a delay, once or repeatedly.",
        "in": {},
        "out": {"tick": "flow"},
        "settings": [S("seconds", "5"), S("repeat", "no", ["no", "yes"])],
    },
    "If": {
        "cat": "logic",
        "desc": "Checks a text value and fires 'then' or 'else'.",
        "in": {"run": "flow", "value": "text"},
        "out": {"then": "flow", "else": "flow"},
        "settings": [
            S("op", "contains", ["contains", "equals", "not_empty", "is_empty"]),
            S("compare", ""),
        ],
    },
    "Print": {
        "cat": "logic",
        "desc": "Prints text to the console when run.",
        "in": {"run": "flow", "text": "text"},
        "out": {"done": "flow"},
        "settings": [S("text", "")],
    },
    "Start": {
        "cat": "logic",
        "desc": "Fires 'start' once, as soon as the app opens.",
        "in": {},
        "out": {"start": "flow"},
        "settings": [],
    },
    "JsonGet": {
        "cat": "logic",
        "desc": "Reads one field out of JSON text when run. Field is a dotted path such as items.0.name.",
        "in": {"run": "flow", "json": "text"},
        "out": {"value": "text", "done": "flow"},
        "settings": [S("field", "name"), S("json", "", label="json (used if nothing is wired)")],
    },
    "TextTransform": {
        "cat": "logic",
        "desc": "Changes text when run: upper, lower, title or strip (trim spaces).",
        "in": {"run": "flow", "text": "text"},
        "out": {"result": "text", "done": "flow"},
        "settings": [S("op", "upper", ["upper", "lower", "title", "strip"]), S("text", "")],
    },
    "Join": {
        "cat": "logic",
        "desc": "Joins text 'a' and text 'b' with a separator when run.",
        "in": {"run": "flow", "a": "text", "b": "text"},
        "out": {"result": "text", "done": "flow"},
        "settings": [S("a", ""), S("b", ""), S("sep", " ", label="separator (\\n = new line)")],
    },
    "Now": {
        "cat": "logic",
        "desc": "Gets the current date and time as text when run, using a strftime format.",
        "in": {"run": "flow"},
        "out": {"value": "text", "done": "flow"},
        "settings": [S("format", "%Y-%m-%d %H:%M:%S")],
    },
    "MessageBox": {
        "cat": "ui",
        "desc": "Shows a popup message when run. Needs no parent.",
        "in": {"run": "flow", "text": "text"},
        "out": {"done": "flow"},
        "settings": [S("title", "Message"), S("text", "")],
    },
}

# Blocks that start a chain of flow: they have a flow output but no flow input.
TRIGGERS = {n for n, b in LIB.items()
            if "flow" in b["out"].values() and "flow" not in b["in"].values()}

DEMO = {
    "blocks": [
        {"id": "win", "type": "Window", "settings": {"title": "URL fetcher", "width": "420", "height": "320"}},
        {"id": "btn1", "type": "Button", "settings": {"text": "Fetch"}},
        {"id": "get1", "type": "HttpGet", "settings": {"url": "https://example.com"}},
        {"id": "lbl1", "type": "Label", "settings": {"text": "", "size": "11"}},
    ],
    "wires": [
        {"from_block": "win", "from_port": "children", "to_block": "btn1", "to_port": "parent"},
        {"from_block": "win", "from_port": "children", "to_block": "lbl1", "to_port": "parent"},
        {"from_block": "btn1", "from_port": "click", "to_block": "get1", "to_port": "run"},
        {"from_block": "get1", "from_port": "body", "to_block": "lbl1", "to_port": "text"},
    ],
}


CLOCK_DEMO = {
    "blocks": [
        {"id": "win", "type": "Window", "settings": {"title": "Clock", "width": "300", "height": "140"}},
        {"id": "tim1", "type": "Timer", "settings": {"seconds": "1", "repeat": "yes"}},
        {"id": "now1", "type": "Now", "settings": {"format": "%H:%M:%S"}},
        {"id": "lbl1", "type": "Label", "settings": {"text": "", "size": "24"}},
    ],
    "wires": [
        {"from_block": "win", "from_port": "children", "to_block": "lbl1", "to_port": "parent"},
        {"from_block": "tim1", "from_port": "tick", "to_block": "now1", "to_port": "run"},
        {"from_block": "now1", "from_port": "value", "to_block": "lbl1", "to_port": "text"},
    ],
}


def setting_of(block, key):
    v = (block.get("settings") or {}).get(key)
    if v is None:
        for s in LIB[block["type"]]["settings"]:
            if s["key"] == key:
                return s["default"]
        return ""
    return str(v)


def num(s, default):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Validation and repair
# ---------------------------------------------------------------------------


def validate_graph(graph):
    errs = []
    ids = {}
    for b in graph.get("blocks") or []:
        t, bid = b.get("type"), b.get("id")
        if t not in LIB:
            errs.append(f"Unknown block type '{t}'.")
            continue
        if not bid:
            errs.append(f"A {t} block has no id.")
            continue
        if bid in ids:
            errs.append(f"Duplicate id '{bid}'.")
        ids[bid] = b

    if sum(1 for b in ids.values() if b["type"] == "Window") != 1:
        errs.append("The graph needs exactly one Window block.")

    wired_inputs = set()
    for w in graph.get("wires") or []:
        fb, fp = w.get("from_block"), w.get("from_port")
        tb, tp = w.get("to_block"), w.get("to_port")
        if fb not in ids or tb not in ids:
            errs.append(f"Wire {fb}.{fp} -> {tb}.{tp} points at a block that doesn't exist.")
            continue
        fo, ti = LIB[ids[fb]["type"]]["out"], LIB[ids[tb]["type"]]["in"]
        if fp not in fo:
            errs.append(f"'{fb}' ({ids[fb]['type']}) has no output '{fp}'. Its outputs: {', '.join(fo) or 'none'}.")
            continue
        if tp not in ti:
            errs.append(f"'{tb}' ({ids[tb]['type']}) has no input '{tp}'. Its inputs: {', '.join(ti) or 'none'}.")
            continue
        if fo[fp] != ti[tp]:
            errs.append(f"Type mismatch: {fb}.{fp} is {fo[fp]} but {tb}.{tp} is {ti[tp]}.")
            continue
        if ti[tp] != "flow":
            if (tb, tp) in wired_inputs:
                errs.append(f"{tb}.{tp} has more than one wire into it.")
            wired_inputs.add((tb, tp))

    for b in ids.values():
        if "parent" in LIB[b["type"]]["in"] and (b["id"], "parent") not in wired_inputs:
            errs.append(f"'{b['id']}' ({b['type']}) needs its parent wired from the Window's children output.")
    return errs


def repair(graph):
    """Fix the mistakes small models make so retries are spent on real problems."""
    out = {"blocks": [], "wires": []}
    idmap, used = {}, set()
    for b in graph.get("blocks") or []:
        if not isinstance(b, dict) or b.get("type") not in LIB:
            continue
        raw = str(b.get("id") or b["type"].lower())
        nid = re.sub(r"\W+", "_", raw).strip("_").lower() or b["type"].lower()
        if nid[0].isdigit():
            nid = "b_" + nid
        base, n = nid, 2
        while nid in used:
            nid = f"{base}{n}"
            n += 1
        used.add(nid)
        idmap.setdefault(raw, nid)
        given = b.get("settings") or {}
        settings = {}
        for s in LIB[b["type"]]["settings"]:
            v = str(given.get(s["key"], s["default"]))
            if s["options"] and v not in s["options"]:
                v = s["default"]
            settings[s["key"]] = v
        nb = {"id": nid, "type": b["type"], "settings": settings}
        for k in ("x", "y"):
            if k in b:
                nb[k] = b[k]
        out["blocks"].append(nb)

    # Keep exactly one Window.
    wins = [b for b in out["blocks"] if b["type"] == "Window"]
    if not wins:
        out["blocks"].insert(0, {
            "id": "window", "type": "Window",
            "settings": {s["key"]: s["default"] for s in LIB["Window"]["settings"]},
        })
        wins = [out["blocks"][0]]
    keep = wins[0]["id"]
    drop = {b["id"] for b in wins[1:]}
    out["blocks"] = [b for b in out["blocks"] if b["id"] not in drop]
    byid = {b["id"]: b for b in out["blocks"]}

    seen, wired_inputs = set(), set()
    for w in graph.get("wires") or []:
        if not isinstance(w, dict):
            continue
        fb = idmap.get(str(w.get("from_block")))
        tb = idmap.get(str(w.get("to_block")))
        fp, tp = w.get("from_port"), w.get("to_port")
        if fb in drop:
            fb = keep
        if tb in drop:
            continue
        if fb not in byid or tb not in byid:
            continue
        ft, tt = LIB[byid[fb]["type"]], LIB[byid[tb]["type"]]
        if fp not in ft["out"] and fp in ft["in"] and tp in tt["out"]:
            fb, fp, tb, tp = tb, tp, fb, fp  # model wired it backwards
            ft, tt = LIB[byid[fb]["type"]], LIB[byid[tb]["type"]]
        if fp not in ft["out"] or tp not in tt["in"] or ft["out"][fp] != tt["in"][tp]:
            continue
        key = (fb, fp, tb, tp)
        if key in seen:
            continue
        if tt["in"][tp] != "flow":
            if (tb, tp) in wired_inputs:
                continue
            wired_inputs.add((tb, tp))
        seen.add(key)
        out["wires"].append({"from_block": fb, "from_port": fp, "to_block": tb, "to_port": tp})

    for b in out["blocks"]:
        if "parent" in LIB[b["type"]]["in"] and (b["id"], "parent") not in wired_inputs:
            out["wires"].append({"from_block": keep, "from_port": "children",
                                 "to_block": b["id"], "to_port": "parent"})
    return out


# ---------------------------------------------------------------------------
# Code generator: graph -> Tkinter .py
# ---------------------------------------------------------------------------


TEST_MARKER = "@@BLOCK_TEST@@"

# Runs first in a test build: makes network, file and popup calls harmless, and records problems.
TEST_PRELUDE = """import json as _tj, io as _tio, traceback as _tb, urllib.request as _ur
_T = {'errors': [], 'stubbed': [], 'popups': []}
_VFS = {}
class _Resp:
    def __init__(self, b): self._b = b
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False
def _fake_urlopen(req, timeout=None):
    _T['stubbed'].append('GET ' + str(getattr(req, 'full_url', req)))
    return _Resp(b'{"title": "Example Domain", "name": "Test", "value": "42", "items": [{"name": "first"}]}')
_ur.urlopen = _fake_urlopen
class _VW(_tio.StringIO):
    def __init__(self, path, mode):
        super().__init__(); self._p = path; self._m = mode
    def close(self):
        if not self.closed:
            _VFS[self._p] = (_VFS.get(self._p, '') if 'a' in self._m else '') + self.getvalue()
            _T['stubbed'].append('WRITE ' + str(self._p))
        super().close()
def open(path, mode='r', *a, **k):
    if 'w' in mode or 'a' in mode:
        return _VW(path, mode)
    if path in _VFS:
        return _tio.StringIO(_VFS[path])
    _T['stubbed'].append('READ ' + str(path))
    return _tio.StringIO('sample file text')
try:
    messagebox.showinfo = lambda title='', msg='', **k: _T['popups'].append(str(msg))
except NameError:
    pass
def _who(tb):
    name = ''
    for fr in _tb.extract_tb(tb):
        head, _, rest = fr.name.partition('_')
        if rest and head in ('click', 'tick', 'run', 'refresh', 'start'):
            name = rest
    return _IDS.get(name, name)
def _fail(bid, et, ev):
    _T['errors'].append({'block': bid, 'error': getattr(et, '__name__', 'Error') + ': ' + str(ev)})
def _report(et, ev, tb):
    _fail(_who(tb), et, ev)
def _step(bid, kind, f):
    try:
        f()
    except Exception as e:
        _fail(_who(e.__traceback__) or bid, type(e), e)
    root.update()"""


def generate(graph, test=False):
    blocks_list = graph.get("blocks") or []
    blocks = {b["id"]: b for b in blocks_list}
    wires = [w for w in (graph.get("wires") or [])
             if w.get("from_block") in blocks and w.get("to_block") in blocks]
    wins = [b for b in blocks_list if b["type"] == "Window"]
    if len(wins) != 1:
        raise ValueError("The graph needs exactly one Window block.")
    win = wins[0]

    def ident(b):
        return re.sub(r"\W", "_", b["id"])

    def widget(b):
        return "w_" + ident(b)

    def fn(prefix, b):
        return prefix + "_" + ident(b)

    def incoming(bid, port):
        for w in wires:
            if w["to_block"] == bid and w["to_port"] == port:
                return w
        return None

    def outgoing(bid, port):
        bid = bid["id"] if isinstance(bid, dict) else bid
        return [w for w in wires if w["from_block"] == bid and w["from_port"] == port]

    def key(bid, port):
        return repr(f"{bid}.{port}")

    def expr(b, port):
        w = incoming(b["id"], port)
        if w:
            s = blocks[w["from_block"]]
            if s["type"] == "Entry" and w["from_port"] == "value":
                return widget(s) + ".get()"
            return "S.get(" + key(s["id"], w["from_port"]) + ", '')"
        return repr(setting_of(b, port))

    def refresh_calls(b, port):
        calls = []
        for w in outgoing(b, port):
            t = blocks[w["to_block"]]
            if t["type"] == "Label" and w["to_port"] == "text":
                calls.append(fn("refresh", t) + "()")
        return calls

    def flow_calls(b, port):
        calls = []
        for w in outgoing(b, port):
            t = blocks[w["to_block"]]
            if LIB[t["type"]]["in"].get(w["to_port"]) == "flow":
                calls.append(fn("run", t) + "()")
        return calls

    def ind(lines, n=1):
        return ["    " * n + ln for ln in lines]

    types = {b["type"] for b in blocks_list}
    width = int(num(setting_of(win, "width"), 420))
    height = int(num(setting_of(win, "height"), 320))
    L = ["import tkinter as tk"]
    if "HttpGet" in types:
        L.append("import urllib.request")
    if "JsonGet" in types:
        L.append("import json")
    if "Now" in types:
        L.append("import datetime")
    if "MessageBox" in types:
        L.append("from tkinter import messagebox")
    if test:
        L += TEST_PRELUDE.split("\n")
    L += ["", "S = {}  # values passed between blocks", "",
          "root = tk.Tk()", "root.title(" + repr(setting_of(win, "title")) + ")",
          f"root.geometry('{width}x{height}')"]
    if test:
        L += ["root.withdraw()", "root.report_callback_exception = _report"]
    L.append("")

    starts = []
    for b in blocks_list:
        t = b["type"]
        if t == "Button":
            L.append(f"def {fn('click', b)}():")
            L += ind(flow_calls(b, "click") or ["pass"])
            L.append("")
        elif t == "Timer":
            ms = int(max(0.1, num(setting_of(b, "seconds"), 5)) * 1000)
            name = fn("tick", b)
            body = flow_calls(b, "tick")
            if setting_of(b, "repeat") == "yes":
                body.append(f"root.after({ms}, {name})")
            L.append(f"def {name}():")
            L += ind(body or ["pass"])
            L.append("")
            starts.append(f"root.after({ms}, {name})")
        elif t == "HttpGet":
            k = key(b["id"], "body")
            L += [f"def {fn('run', b)}():",
                  f"    url = str({expr(b, 'url')}).strip()",
                  "    if not url.startswith(('http://', 'https://')):",
                  "        url = 'http://' + url",
                  "    try:",
                  "        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})",
                  "        with urllib.request.urlopen(req, timeout=15) as r:",
                  f"            S[{k}] = r.read().decode('utf-8', 'replace')",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "body") + flow_calls(b, "done"))
            L.append("")
        elif t == "ReadFile":
            k = key(b["id"], "content")
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        with open(str({expr(b, 'path')}), encoding='utf-8', errors='replace') as f:",
                  f"            S[{k}] = f.read()",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "content") + flow_calls(b, "done"))
            L.append("")
        elif t == "WriteFile":
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        with open(str({expr(b, 'path')}), 'w', encoding='utf-8') as f:",
                  f"            f.write(str({expr(b, 'content')}))",
                  "    except Exception as e:",
                  "        print('WriteFile failed:', e)"]
            L += ind(flow_calls(b, "done"))
            L.append("")
        elif t == "If":
            op = setting_of(b, "op")
            cond = {"contains": "cmp in v", "equals": "v == cmp",
                    "not_empty": "v.strip() != ''", "is_empty": "v.strip() == ''"}.get(op, "cmp in v")
            L += [f"def {fn('run', b)}():",
                  f"    v = str({expr(b, 'value')})",
                  f"    cmp = {setting_of(b, 'compare')!r}",
                  f"    if {cond}:"]
            L += ind(flow_calls(b, "then") or ["pass"], 2)
            L.append("    else:")
            L += ind(flow_calls(b, "else") or ["pass"], 2)
            L.append("")
        elif t == "Print":
            L.append(f"def {fn('run', b)}():")
            L.append(f"    print(str({expr(b, 'text')}), flush=True)")
            L += ind(flow_calls(b, "done"))
            L.append("")
        elif t == "Start":
            name = fn("start", b)
            L.append(f"def {name}():")
            L += ind(flow_calls(b, "start") or ["pass"])
            L.append("")
            starts.append(f"root.after(100, {name})")
        elif t == "JsonGet":
            k = key(b["id"], "value")
            L += [f"def {fn('run', b)}():",
                  f"    field = {setting_of(b, 'field')!r}",
                  "    try:",
                  f"        d = json.loads(str({expr(b, 'json')}))",
                  "        for part in field.split('.'):",
                  "            part = part.strip()",
                  "            if not part:",
                  "                continue",
                  "            d = d[int(part)] if isinstance(d, list) else d[part]",
                  f"        S[{k}] = d if isinstance(d, str) else json.dumps(d)",
                  "    except json.JSONDecodeError:",
                  f"        S[{k}] = 'Error: input is not valid JSON'",
                  "    except (KeyError, IndexError, ValueError, TypeError):",
                  f"        S[{k}] = 'Error: field not found: ' + field"]
            L += ind(refresh_calls(b, "value") + flow_calls(b, "done"))
            L.append("")
        elif t == "TextTransform":
            k = key(b["id"], "result")
            call = {"upper": "upper()", "lower": "lower()", "title": "title()",
                    "strip": "strip()"}.get(setting_of(b, "op"), "upper()")
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        S[{k}] = str({expr(b, 'text')}).{call}",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "result") + flow_calls(b, "done"))
            L.append("")
        elif t == "Join":
            k = key(b["id"], "result")
            sep = setting_of(b, "sep").replace("\\n", "\n")
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        S[{k}] = str({expr(b, 'a')}) + {sep!r} + str({expr(b, 'b')})",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "result") + flow_calls(b, "done"))
            L.append("")
        elif t == "Now":
            k = key(b["id"], "value")
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        S[{k}] = datetime.datetime.now().strftime({setting_of(b, 'format')!r})",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "value") + flow_calls(b, "done"))
            L.append("")
        elif t == "MessageBox":
            L += [f"def {fn('run', b)}():",
                  f"    messagebox.showinfo({setting_of(b, 'title')!r}, str({expr(b, 'text')}))"]
            L += ind(flow_calls(b, "done"))
            L.append("")
        elif t == "Label":
            L.append(f"def {fn('refresh', b)}():")
            L.append(f"    {widget(b)}.config(text=str({expr(b, 'text')}))")
            L.append("")

    wrap = max(100, width - 30)
    for b in blocks_list:
        t, w = b["type"], widget(b)
        if t == "Button":
            L += [f"{w} = tk.Button(root, text={setting_of(b, 'text')!r}, command={fn('click', b)}, font=('Segoe UI', 11))",
                  f"{w}.pack(padx=10, pady=6)"]
        elif t == "Label":
            size = int(num(setting_of(b, "size"), 12))
            L += [f"{w} = tk.Label(root, text={setting_of(b, 'text')!r}, font=('Segoe UI', {size}), "
                  f"wraplength={wrap}, justify='left')",
                  f"{w}.pack(padx=10, pady=6, anchor='w')"]
        elif t == "Entry":
            L += [f"{w} = tk.Entry(root, font=('Segoe UI', 12))",
                  f"{w}.insert(0, {setting_of(b, 'default')!r})",
                  f"{w}.pack(fill='x', padx=10, pady=6)"]
            refreshers = refresh_calls(b, "value")
            if refreshers:
                L.append(f"{w}.bind('<KeyRelease>', lambda e: ({', '.join(refreshers)}))")
    L.append("")
    if not test:
        L += starts
        L.append("root.mainloop()")
        return "\n".join(L) + "\n"

    # Test harness: no mainloop. Fire every trigger once, then print a JSON report.
    idmap = {ident(b): b["id"] for b in blocks_list}
    L.append("_IDS = " + repr(idmap))
    for b in blocks_list:
        if b["type"] in ("Start", "Timer"):
            f = fn("start" if b["type"] == "Start" else "tick", b)
            L.append(f"_step({b['id']!r}, {b['type'].lower()!r}, {f})")
    for b in blocks_list:
        if b["type"] == "Button":
            L.append(f"_step({b['id']!r}, 'click', {widget(b)}.invoke)")
    L.append("root.update()")
    labels = [b for b in blocks_list if b["type"] == "Label"]
    L.append("_T['S'] = {k: str(v)[:300] for k, v in S.items()}")
    L.append("_T['labels'] = {" + ", ".join(f"{b['id']!r}: {widget(b)}.cget('text')" for b in labels) + "}")
    L.append("print('" + TEST_MARKER + "' + _tj.dumps(_T), flush=True)")
    L.append("root.destroy()")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Model schema and prompt
# ---------------------------------------------------------------------------


def build_schema():
    ports = sorted({p for b in LIB.values() for p in list(b["in"]) + list(b["out"])})
    variants = []
    for name, b in LIB.items():
        props = {}
        for s in b["settings"]:
            props[s["key"]] = {"enum": s["options"]} if s["options"] else {"type": "string"}
        variants.append({
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"const": name},
                "settings": {"type": "object", "properties": props, "additionalProperties": False},
            },
            "required": ["id", "type", "settings"],
            "additionalProperties": False,
        })
    wire = {
        "type": "object",
        "properties": {
            "from_block": {"type": "string"}, "from_port": {"enum": ports},
            "to_block": {"type": "string"}, "to_port": {"enum": ports},
        },
        "required": ["from_block", "from_port", "to_block", "to_port"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "blocks": {"type": "array", "items": {"anyOf": variants}, "minItems": 1},
            "wires": {"type": "array", "items": wire},
        },
        "required": ["blocks", "wires"],
        "additionalProperties": False,
    }


def system_prompt():
    lines = ["You design small Tkinter apps as block graphs. Reply with JSON only.", "", "Blocks:"]
    for name, b in LIB.items():
        ins = ", ".join(f"{p}({t})" for p, t in b["in"].items()) or "none"
        outs = ", ".join(f"{p}({t})" for p, t in b["out"].items()) or "none"
        sets = ", ".join(s["key"] for s in b["settings"]) or "none"
        lines.append(f"- {name}: {b['desc']} Inputs: {ins}. Outputs: {outs}. Settings: {sets}.")
    lines += [
        "",
        "Rules:",
        "1. Exactly one Window block.",
        "2. Every Button, Label and Entry has its 'parent' input wired from the Window's 'children' output.",
        "3. A wire goes from an output port to an input port of the same type (ui, flow or text).",
        "4. flow ports trigger actions (Button.click -> HttpGet.run). text ports carry data (HttpGet.body -> Label.text).",
        "5. Give each block a short unique lowercase id such as btn1 or lbl1.",
        "6. Use only the blocks and ports listed above. Every setting value is a string.",
        "7. Blocks with a 'run' input do nothing until a flow wire reaches it. Use Start.start for things that "
        "happen when the app opens, Button.click for user actions and Timer.tick for repeats.",
        "8. To do several steps in order, wire each block's 'done' output into the next block's 'run' input.",
        "9. A text output can feed several inputs. Each text input accepts only one wire.",
        "",
        "Example request: a window with a button that fetches a URL and shows the result",
        "Example answer:",
        json.dumps(DEMO),
        "",
        "Example request: a clock that updates every second",
        "Example answer:",
        json.dumps(CLOCK_DEMO),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model backend: the model already loaded in this app (no HTTP hop - the Block Editor calls
# the same HttpBridge.chat() the optional HTTP API uses, so it queues behind chat replies,
# is cancelled by unloading, and shows live in the chat log).
# ---------------------------------------------------------------------------

_BE_APP = None      # the SimpleChat instance, set by BlockEditorServer.start()


def _be_model_call(messages, schema, cfg):
    if _BE_APP is None:
        raise RuntimeError("The Block Editor isn't attached to GGUFllama.")
    try:
        out = _BE_APP.http_bridge.chat({
            "messages": messages, "schema": schema,
            "temperature": float((cfg or {}).get("temperature", 0.2)), "max_tokens": 2048})
    except BridgeError as exc:
        raise RuntimeError(str(exc))
    return out["content"]


MAX_ATTEMPTS = 4


def _source_type(graph, bid, port):
    """Type of the block wired into bid.port, or None."""
    types = {b["id"]: b["type"] for b in graph.get("blocks") or []}
    for w in graph.get("wires") or []:
        if w.get("to_block") == bid and w.get("to_port") == port:
            return types.get(w.get("from_block"))
    return None


def _untriggered(graph):
    """Blocks with a 'run' input that no Button/Timer/Start chain ever reaches."""
    blocks = {b["id"]: b for b in graph.get("blocks") or []}
    wires = graph.get("wires") or []
    fired = {i for i, b in blocks.items() if b["type"] in TRIGGERS}
    stack = list(fired)
    while stack:
        cur = stack.pop()
        outs = LIB[blocks[cur]["type"]]["out"]
        for w in wires:
            if w.get("from_block") == cur and outs.get(w.get("from_port")) == "flow":
                t = w.get("to_block")
                if t in blocks and t not in fired:
                    fired.add(t)
                    stack.append(t)
    return [b for b in blocks.values() if "run" in LIB[b["type"]]["in"] and b["id"] not in fired]


def test_graph(graph, timeout=20):
    """Build the app in test mode and run it headless with network/files/popups simulated.

    Returns {ran, ok, failures, notes, labels, stubbed, skipped}. Only 'failures' should be sent
    back to the model; 'notes' are things the simulated data can't prove either way.
    """
    res = {"ran": False, "ok": True, "failures": [], "notes": [], "labels": {},
           "stubbed": [], "skipped": None}

    def fail(msg):
        res["failures"].append(msg)
        res["ok"] = False

    blocks = {b["id"]: b for b in graph.get("blocks") or []}
    for b in _untriggered(graph):
        fail(f"{b['type']} block '{b['id']}' never runs: nothing triggers its 'run' input. Wire a "
             "Button 'click', Timer 'tick' or Start 'start' into it, or another block's 'done'.")

    try:
        code = generate(graph, test=True)
    except Exception as e:
        fail(f"Code generation failed: {e}")
        return res
    try:
        compile(code, "app_test.py", "exec")
    except SyntaxError as e:
        fail(f"The generated code has a syntax error on line {e.lineno}: {e.msg}.")
        return res

    py = _be_python()
    if not py:
        res["skipped"] = "Python wasn't found on PATH, so the app couldn't be test-run."
        return res
    cmd = [py]
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        xvfb = shutil.which("xvfb-run")
        if not xvfb:
            res["skipped"] = "No display is available, so the app couldn't be test-run."
            return res
        cmd = [xvfb, "-a", py]

    tmp = tempfile.mkdtemp(prefix="block_test_")
    path = os.path.join(tmp, "app_test.py")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            r = subprocess.run(cmd + [path], capture_output=True, text=True, timeout=timeout,
                               cwd=tmp, creationflags=flags)
        except subprocess.TimeoutExpired:
            fail(f"The app didn't finish its test run within {timeout} seconds. "
                 "Something in it may be stuck or waiting.")
            return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    res["ran"] = True
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith(TEST_MARKER)), None)
    if line is None:
        both = (r.stderr or "") + "\n" + (r.stdout or "")   # xvfb-run merges stderr into stdout
        last = next((ln for ln in reversed(both.splitlines()) if ln.strip()), "")
        fail("The app crashed before its test finished" + (f": {last.strip()}" if last else "."))
        return res
    try:
        data = json.loads(line[len(TEST_MARKER):])
    except ValueError:
        fail("The test report couldn't be read.")
        return res

    seen = set()
    for e in data.get("errors") or []:
        bid = e.get("block") or ""
        typ = blocks.get(bid, {}).get("type", "block")
        msg = f"{typ} block '{bid}' raised {e.get('error')}" if bid else f"The app raised {e.get('error')}"
        if msg not in seen:
            seen.add(msg)
            fail(msg)
    for k, v in (data.get("S") or {}).items():
        if not str(v).startswith("Error:"):
            continue
        bid = k.split(".", 1)[0]
        typ = blocks.get(bid, {}).get("type", "block")
        upstream = _source_type(graph, bid, "json") if typ == "JsonGet" else None
        if typ == "JsonGet" and upstream in ("HttpGet", "ReadFile"):
            res["notes"].append(f"JsonGet '{bid}' said '{v}', but the test used sample data, "
                                "so this can't be checked without the real response.")
        else:
            fail(f"{typ} block '{bid}' produced: {v}")
    res["labels"] = data.get("labels") or {}
    for lid, text in res["labels"].items():
        if text == "" and _source_type(graph, lid, "text"):
            res["notes"].append(f"Label '{lid}' is still empty after the test.")
    res["stubbed"] = sorted(set(data.get("stubbed") or []))
    res["popups"] = data.get("popups") or []
    return res


def _test_feedback(t):
    lines = ["I built your graph into an app and ran it (network, files and popups were simulated with "
             "sample data), and it had problems:"]
    lines += ["- " + f for f in t["failures"]]
    if t["labels"]:
        lines.append("After firing every trigger, the labels showed: " +
                     ", ".join(f"{k} = {v!r}" for k, v in t["labels"].items()) + ".")
    lines.append("Fix the wiring or settings and return the full corrected graph as JSON.")
    return "\n".join(lines)


def ask_model(prompt, cfg):
    schema = build_schema()
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt}]
    call = _be_model_call
    do_test = cfg.get("auto_test", True) is not False
    graph, errs, test = {"blocks": [], "wires": []}, ["No graph returned."], None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        text = call(messages, schema, cfg)
        test = None
        try:
            graph = repair(json.loads(text))
            errs = validate_graph(graph)
        except (ValueError, TypeError, AttributeError) as e:
            errs = [f"The output wasn't a valid graph ({e})."]
        if not errs and do_test:
            test = test_graph(graph)
            if test["failures"]:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": _test_feedback(test)})
                if attempt < MAX_ATTEMPTS:
                    continue
                return {"graph": graph, "attempts": attempt, "errors": [], "test": test}
        if not errs:
            return {"graph": graph, "attempts": attempt, "errors": [], "test": test}
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
                         "That graph has errors:\n- " + "\n- ".join(errs) +
                         "\nReturn the full corrected graph as JSON."})
    return {"graph": graph, "attempts": MAX_ATTEMPTS, "errors": errs, "test": test}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Block Editor web server (Flask, imported lazily so the chat app still runs without it)
# ---------------------------------------------------------------------------

class BlockEditorServer:
    def __init__(self, app, host=BLOCK_EDITOR_HOST, port=BLOCK_EDITOR_PORT):
        self.app = app
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    @property
    def running(self):
        return self._server is not None

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    def _build_flask(self):
        from flask import Flask, Response, jsonify, request      # ImportError -> caller reports it
        flask_app = Flask("block_editor")

        @flask_app.get("/")
        def index():
            return Response(BLOCK_EDITOR_PAGE, mimetype="text/html")

        @flask_app.get("/api/library")
        def api_library():
            return jsonify({"lib": LIB, "demo": DEMO})

        @flask_app.get("/api/status")
        def api_status():
            d = self.app.http_bridge.status()
            return jsonify({"reachable": True, "loaded": bool(d.get("loaded")),
                            "model": d.get("model"), "reason": d.get("reason")})

        @flask_app.post("/api/generate")
        def api_generate():
            graph = (request.get_json(silent=True) or {}).get("graph") or {}
            errors = validate_graph(graph)
            try:
                code = generate(graph)
            except Exception as e:
                code = f"# Fix the problems above to generate code.\n# ({e})\n"
            return jsonify({"code": code, "errors": errors})

        @flask_app.post("/api/test")
        def api_test():
            graph = (request.get_json(silent=True) or {}).get("graph") or {}
            errors = validate_graph(graph)
            if errors:
                return jsonify({"error": "Fix the graph problems first: " + errors[0]}), 400
            try:
                return jsonify(test_graph(graph))
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @flask_app.post("/api/ask")
        def api_ask():
            data = request.get_json(silent=True) or {}
            prompt = (data.get("prompt") or "").strip()
            cfg = data.get("cfg") or {}
            if not prompt:
                return jsonify({"error": "Describe the app you want first."}), 400
            try:
                return jsonify(ask_model(prompt, cfg))
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @flask_app.post("/api/run")
        def api_run():
            code = (request.get_json(silent=True) or {}).get("code") or ""
            path = os.path.join(tempfile.gettempdir(), f"block_app_{int(time.time())}.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            py = _be_python()
            if not py:
                return jsonify({"ok": False, "message": "Python wasn't found on PATH, so the app can't run. "
                                                         "Save the .py and run it yourself."})
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen([py, path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, creationflags=flags)
            try:
                out, _ = proc.communicate(timeout=1.5)
                return jsonify({"ok": proc.returncode == 0, "message": "The app exited.", "output": out or ""})
            except subprocess.TimeoutExpired:
                return jsonify({"ok": True, "message": "App is running in its own window.", "output": ""})

        return flask_app

    def start(self):
        """Start serving. Raises ImportError (Flask missing) or OSError (port taken)."""
        global _BE_APP
        if self._server is not None:
            return
        from werkzeug.serving import make_server
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)     # keep the console quiet
        flask_app = self._build_flask()
        self._server = make_server(self.host, self.port, flask_app, threaded=True)
        _BE_APP = self.app
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()


def _be_python():
    """Interpreter used to test-run / run generated apps."""
    if getattr(sys, "frozen", False):
        return shutil.which("python") or shutil.which("py")
    return sys.executable



BLOCK_EDITOR_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Block Editor</title>
<style>
:root{--bg:#14161a;--panel:#1b1e24;--panel2:#23272f;--line:#2d323b;--text:#e6e8ec;--dim:#9aa3b2;
--flow:#e9edf3;--txt:#38bdf8;--ui:#8b93a7;--accent:#38bdf8;--ok:#34d399;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 "Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--text);
height:100vh;display:flex;flex-direction:column}
button,input,select{font:inherit;color:var(--text)}
button{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer}
button:hover{border-color:#4a5262}
button.primary{background:var(--accent);border-color:var(--accent);color:#06222e;font-weight:600}
input[type=text],input[type=number],select{background:#12141a;border:1px solid var(--line);border-radius:6px;
padding:7px 10px;width:100%}
input:focus,select:focus,button:focus-visible{outline:2px solid var(--accent);outline-offset:0}
header{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:600;margin-right:auto}
.pill{font-size:12px;color:var(--dim);display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;background:#555}
.dot.on{background:var(--ok)}
#askRow{display:flex;gap:8px;padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line)}
#status{padding:0 14px;font-size:13px;display:none}
#status.show{display:block;margin:8px 14px 0;padding:8px 12px;border-radius:6px}
#status.ok{background:#12352b;color:var(--ok)}
#status.warn{background:#3b3010;color:var(--warn)}
#status.bad{background:#3d1a1a;color:var(--bad)}
#status.busy{background:var(--panel2);color:var(--dim)}
#status ul{margin:4px 0 0;padding-left:18px}
#app{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 360px;min-height:0;margin-top:8px}
#left{display:flex;flex-direction:column;min-height:0;min-width:0}
#palette,#bar{display:flex;flex-wrap:wrap;gap:6px;padding:6px 14px}
.chip{font-size:12px;padding:3px 10px;border-radius:999px;border:1px solid transparent}
.chip.ui{background:#312e5a;color:#c9c4ff}.chip.io{background:#0f3a35;color:#7ee8d6}.chip.logic{background:#43310d;color:#fcd47a}
#bar button{font-size:12px;padding:4px 10px}
#canvas{flex:1;position:relative;overflow:hidden;min-height:320px;touch-action:none;cursor:grab;
background-image:radial-gradient(#2a2f38 1px,transparent 1px);background-size:24px 24px;border-top:1px solid var(--line)}
#canvas.panning{cursor:grabbing}
#world{position:absolute;left:0;top:0;transform-origin:0 0}
#wires{position:absolute;left:0;top:0;width:1px;height:1px;overflow:visible;pointer-events:none}
.wire{fill:none;stroke-width:2.2;pointer-events:none}
.wire.t-flow{stroke:var(--flow)}.wire.t-text{stroke:var(--txt)}.wire.t-ui{stroke:#64748b;stroke-dasharray:5 4;stroke-width:1.6}
.wire.sel{stroke:var(--warn);stroke-width:3.2}
.hit{fill:none;stroke:transparent;stroke-width:16;pointer-events:stroke;cursor:pointer}
.node{position:absolute;background:var(--panel);border:1px solid var(--line);border-radius:8px;user-select:none}
.node.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.head{height:30px;display:flex;align-items:center;justify-content:space-between;padding:0 10px;border-radius:7px 7px 0 0;cursor:move;font-weight:600;font-size:13px}
.head .nid{font-weight:400;font-size:11px;opacity:.7}
.cat-ui .head{background:#312e5a;color:#d4d0ff}.cat-io .head{background:#0f3a35;color:#a7f0e3}.cat-logic .head{background:#43310d;color:#fde3a5}
.prow{height:26px;position:relative;display:flex;justify-content:space-between;align-items:center;padding:0 14px;font-size:12px;color:var(--text)}
.sub{height:26px;padding:0 10px;font-size:12px;color:var(--dim);display:flex;align-items:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-top:1px solid var(--line);margin-top:6px}
.port{position:absolute;top:6px;width:14px;height:14px;border-radius:50%;background:var(--panel);border:2px solid var(--c);cursor:crosshair}
.port::before{content:"";position:absolute;inset:-7px}
.port.in{left:-7px}.port.out{right:-7px}
.port.t-flow{--c:var(--flow)}.port.t-text{--c:var(--txt)}.port.t-ui{--c:#7b8499}
.port.out{background:var(--c)}
.port.ok{box-shadow:0 0 0 4px rgba(56,189,248,.35)}
#side{border-left:1px solid var(--line);background:var(--panel);display:flex;flex-direction:column;min-height:0;overflow:auto}
.sec{padding:14px;border-bottom:1px solid var(--line)}
.ptitle{font-weight:600;font-size:15px}.dim{color:var(--dim);font-weight:400;font-size:12px}
.pdesc{color:var(--dim);font-size:13px;margin:4px 0 12px}
.field{display:block;margin-bottom:10px}.field span{display:block;font-size:12px;color:var(--dim);margin-bottom:3px}
.ports{font-size:12px;color:var(--dim);margin:6px 0 12px}
button.danger{color:var(--bad)}
#codeHead{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
#codeHead b{margin-right:auto;font-size:13px}
#codeHead button{font-size:12px;padding:4px 10px}
#codeErrors{display:none;margin:10px 0 0;padding:8px 8px 8px 24px;background:#3b3010;color:var(--warn);border-radius:6px;font-size:12px}
#code{margin:10px 0 0;padding:10px;background:#101216;border:1px solid var(--line);border-radius:6px;font:12px/1.55 Consolas,"Cascadia Mono",monospace;overflow:auto;white-space:pre;max-height:380px}
#runOut{margin:8px 0 0;font-size:12px;color:var(--dim);white-space:pre-wrap;font-family:Consolas,monospace}
dialog{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:18px;width:min(420px,92vw)}
dialog::backdrop{background:rgba(0,0,0,.6)}
dialog h2{margin:0 0 12px;font-size:16px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.chk{display:flex;align-items:center;gap:8px;margin:6px 0 12px;font-size:13px}
.chk input{width:auto}
@media(max-width:900px){body{height:auto;min-height:100vh}#app{grid-template-columns:1fr}#canvas{height:62vh;flex:none}#side{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header>
  <h1>Block Editor</h1>
  <span class="pill"><span class="dot" id="dot"></span><span id="modelTxt">Checking GGUFllama…</span></span>
  <button id="settingsBtn">Settings</button>
</header>
<div id="askRow">
  <input type="text" id="prompt" placeholder="A window with a button that fetches a URL and shows the result" aria-label="Describe the app">
  <button class="primary" id="askBtn">Ask</button>
</div>
<div id="status" role="status"></div>
<div id="app">
  <div id="left">
    <div id="palette"></div>
    <div id="bar">
      <button id="layoutBtn">Auto-layout</button>
      <button id="fitBtn">Fit</button>
      <button id="zoomIn" aria-label="Zoom in">+</button>
      <button id="zoomOut" aria-label="Zoom out">&minus;</button>
      <button id="saveBtn">Save graph</button>
      <button id="loadBtn">Load graph</button>
      <button id="clearBtn">Clear</button>
      <input type="file" id="fileIn" accept=".json" hidden>
    </div>
    <div id="canvas"><div id="world"><svg id="wires"></svg></div></div>
  </div>
  <div id="side">
    <div class="sec" id="panel"></div>
    <div class="sec">
      <div id="codeHead"><b>Generated app.py</b>
        <button id="copyBtn">Copy</button><button id="savePyBtn">Save .py</button><button id="testBtn" title="Run the app headless with sample data and report problems">Test</button><button class="primary" id="runBtn">Run</button></div>
      <ul id="codeErrors"></ul>
      <pre id="code"></pre>
      <div id="runOut"></div>
    </div>
  </div>
</div>

<dialog id="dlg">
  <h2>Settings</h2>
  <label class="field"><span>Temperature</span><input type="number" id="cTemp" min="0" max="1" step="0.1"></label>
  <label style="display:flex;gap:8px;align-items:center;font-size:13px;margin:0 0 12px"><input type="checkbox" id="cTest"> Test the app after building and let the model fix problems</label>
  <p style="font-size:12px;color:var(--dim);margin:0 0 12px">Models are loaded in GGUFllama's Models window (Local backend). The Block Editor uses the loaded model directly.</p>
  <div style="display:flex;justify-content:flex-end;gap:8px"><button id="dlgCancel">Cancel</button><button class="primary" id="dlgSave">Save settings</button></div>
</dialog>

<script>
const NODE_W=176,HEAD=30,ROW=26,SUB=26;
let LIB={},graph={blocks:[],wires:[]},sel=null,view={x:40,y:30,z:1},drag=null;
const $=s=>document.querySelector(s);
const canvas=$('#canvas'),world=$('#world'),svg=$('#wires');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const byId=id=>graph.blocks.find(b=>b.id===id);

function dims(b){const t=LIB[b.type];const rows=Math.max(Object.keys(t.in).length,Object.keys(t.out).length);
  return{w:NODE_W,h:HEAD+rows*ROW+(t.settings.length?SUB+6:0)+4};}
function portPos(b,side,name){const t=LIB[b.type];const keys=Object.keys(side==='in'?t.in:t.out);const i=keys.indexOf(name);
  return{x:b.x+(side==='in'?0:NODE_W),y:b.y+HEAD+i*ROW+ROW/2};}
function toWorld(cx,cy){const r=canvas.getBoundingClientRect();return{x:(cx-r.left-view.x)/view.z,y:(cy-r.top-view.y)/view.z};}
function applyView(){world.style.transform=`translate(${view.x}px,${view.y}px) scale(${view.z})`;
  canvas.style.backgroundPosition=`${view.x}px ${view.y}px`;canvas.style.backgroundSize=`${24*view.z}px ${24*view.z}px`;}
function curve(a,b){const dx=Math.max(40,Math.abs(b.x-a.x)/2);return`M${a.x} ${a.y} C${a.x+dx} ${a.y},${b.x-dx} ${b.y},${b.x} ${b.y}`;}

function render(){
  world.querySelectorAll('.node').forEach(n=>n.remove());
  graph.blocks.forEach(b=>{
    const t=LIB[b.type],d=dims(b),ins=Object.entries(t.in),outs=Object.entries(t.out);
    const n=document.createElement('div');n.className=`node cat-${t.cat}`+(sel&&sel.kind==='block'&&sel.id===b.id?' sel':'');
    n.style.cssText=`left:${b.x}px;top:${b.y}px;width:${d.w}px;height:${d.h}px`;n.dataset.id=b.id;
    n.innerHTML=`<div class="head"><span>${esc(b.type)}</span><span class="nid">${esc(b.id)}</span></div>`;
    const rows=Math.max(ins.length,outs.length);
    for(let i=0;i<rows;i++){
      const r=document.createElement('div');r.className='prow';
      const l=document.createElement('span'),rt=document.createElement('span');
      if(ins[i]){l.textContent=ins[i][0];const p=document.createElement('div');p.className=`port in t-${ins[i][1]}`;p.dataset.id=b.id;p.dataset.port=ins[i][0];r.appendChild(p);}
      if(outs[i]){rt.textContent=outs[i][0];const p=document.createElement('div');p.className=`port out t-${outs[i][1]}`;p.dataset.id=b.id;p.dataset.port=outs[i][0];
        p.addEventListener('pointerdown',e=>startWire(e,b,outs[i][0],outs[i][1]));r.appendChild(p);}
      r.prepend(l);r.appendChild(rt);n.appendChild(r);
    }
    if(t.settings.length){const s=t.settings[0];const v=(b.settings[s.key]??s.default)+'';
      const sub=document.createElement('div');sub.className='sub';sub.textContent=`${s.key}: ${v}`;n.appendChild(sub);}
    const head=n.querySelector('.head');
    head.addEventListener('pointerdown',e=>{
      if(e.button>0)return;e.stopPropagation();select({kind:'block',id:b.id});
      const st=toWorld(e.clientX,e.clientY),ox=b.x,oy=b.y;head.setPointerCapture(e.pointerId);
      const mv=ev=>{const p=toWorld(ev.clientX,ev.clientY);b.x=Math.round(ox+p.x-st.x);b.y=Math.round(oy+p.y-st.y);
        n.style.left=b.x+'px';n.style.top=b.y+'px';drawWires();};
      const up=()=>{head.removeEventListener('pointermove',mv);head.removeEventListener('pointerup',up);};
      head.addEventListener('pointermove',mv);head.addEventListener('pointerup',up);});
    n.addEventListener('pointerdown',e=>{e.stopPropagation();if(!e.target.closest('.head'))select({kind:'block',id:b.id});});
    world.appendChild(n);
  });
  drawWires();
}

function drawWires(){
  let h='';
  graph.wires.forEach((w,i)=>{
    const a=byId(w.from_block),b=byId(w.to_block);if(!a||!b)return;
    const t=LIB[a.type].out[w.from_port];const d=curve(portPos(a,'out',w.from_port),portPos(b,'in',w.to_port));
    const s=sel&&sel.kind==='wire'&&sel.i===i;
    h+=`<path class="wire t-${t}${s?' sel':''}" d="${d}"/><path class="hit" data-i="${i}" d="${d}"/>`;
  });
  if(drag&&drag.to)h+=`<path class="wire t-${drag.type}" style="opacity:.6" d="${curve(drag.from,drag.to)}"/>`;
  svg.innerHTML=h;
  svg.querySelectorAll('.hit').forEach(p=>p.addEventListener('pointerdown',e=>{e.stopPropagation();select({kind:'wire',i:+p.dataset.i});}));
}

function select(s){sel=s;world.querySelectorAll('.node').forEach(n=>n.classList.toggle('sel',!!s&&s.kind==='block'&&n.dataset.id===s.id));drawWires();renderPanel();}

function startWire(e,b,name,type){
  e.stopPropagation();e.preventDefault();const port=e.target;port.setPointerCapture(e.pointerId);
  drag={from:portPos(b,'out',name),type,to:null};
  world.querySelectorAll('.port.in').forEach(p=>{const tb=byId(p.dataset.id);
    if(tb&&tb.id!==b.id&&LIB[tb.type].in[p.dataset.port]===type)p.classList.add('ok');});
  const mv=ev=>{drag.to=toWorld(ev.clientX,ev.clientY);drawWires();};
  const up=ev=>{port.removeEventListener('pointermove',mv);port.removeEventListener('pointerup',up);
    const t=document.elementFromPoint(ev.clientX,ev.clientY);const pt=t&&t.closest?t.closest('.port.in'):null;
    if(pt)connect(b.id,name,pt.dataset.id,pt.dataset.port);
    drag=null;world.querySelectorAll('.port.ok').forEach(p=>p.classList.remove('ok'));drawWires();};
  port.addEventListener('pointermove',mv);port.addEventListener('pointerup',up);
}

function connect(fb,fp,tb,tp){
  const a=byId(fb),b=byId(tb);if(!a||!b||fb===tb)return;
  const ft=LIB[a.type].out[fp],tt=LIB[b.type].in[tp];
  if(ft!==tt){setStatus('warn',`Can't wire ${ft} to ${tt}. Ports must be the same type.`);return;}
  if(tt!=='flow')graph.wires=graph.wires.filter(w=>!(w.to_block===tb&&w.to_port===tp));
  if(graph.wires.some(w=>w.from_block===fb&&w.from_port===fp&&w.to_block===tb&&w.to_port===tp))return;
  graph.wires.push({from_block:fb,from_port:fp,to_block:tb,to_port:tp});setStatus();render();scheduleGen();
}

function renderPanel(){
  const p=$('#panel');p.innerHTML='';
  const add=(tag,cls,html)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(html!==undefined)e.innerHTML=html;p.appendChild(e);return e;};
  if(sel&&sel.kind==='block'&&byId(sel.id)){
    const b=byId(sel.id),t=LIB[b.type];
    add('div','ptitle',`${esc(b.type)} <span class="dim">${esc(b.id)}</span>`);add('p','pdesc',esc(t.desc));
    t.settings.forEach(s=>{
      const wrap=add('label','field');const sp=document.createElement('span');sp.textContent=s.label;wrap.appendChild(sp);
      let inp;const cur=b.settings[s.key]??s.default;
      if(s.options){inp=document.createElement('select');s.options.forEach(o=>{const op=document.createElement('option');op.value=o;op.textContent=o;inp.appendChild(op);});}
      else{inp=document.createElement('input');inp.type='text';}
      inp.value=cur;inp.addEventListener('input',()=>{b.settings[s.key]=inp.value;render();scheduleGen();});wrap.appendChild(inp);
    });
    const io=[...Object.entries(t.in).map(([k,v])=>`${k} (${v})`),...Object.entries(t.out).map(([k,v])=>`${k} (${v})`)];
    add('div','ports',`Inputs: ${Object.entries(t.in).map(([k,v])=>k+' ('+v+')').join(', ')||'none'}<br>Outputs: ${Object.entries(t.out).map(([k,v])=>k+' ('+v+')').join(', ')||'none'}`);
    const del=add('button','danger','Delete block');del.addEventListener('click',()=>{
      graph.blocks=graph.blocks.filter(x=>x.id!==b.id);graph.wires=graph.wires.filter(w=>w.from_block!==b.id&&w.to_block!==b.id);
      sel=null;render();renderPanel();scheduleGen();});
  }else if(sel&&sel.kind==='wire'&&graph.wires[sel.i]){
    const w=graph.wires[sel.i];
    add('div','ptitle','Wire');add('p','pdesc',`${esc(w.from_block)}.${esc(w.from_port)} &rarr; ${esc(w.to_block)}.${esc(w.to_port)}`);
    const del=add('button','danger','Delete wire');del.addEventListener('click',()=>{graph.wires.splice(sel.i,1);sel=null;render();renderPanel();scheduleGen();});
  }else{
    add('div','ptitle','Nothing selected');
    add('p','pdesc','Select a block to edit its settings. Drag from a filled dot on the right of a block to a highlighted dot on another block to wire them. Tap a wire to select it.');
    add('div','ports','<span style="color:var(--flow)">&#9679;</span> flow triggers &nbsp; <span style="color:var(--txt)">&#9679;</span> text data &nbsp; <span style="color:#7b8499">&#9679;</span> ui parent');
  }
}

function setStatus(kind,msg,list){
  const s=$('#status');if(!kind){s.className='';s.textContent='';return;}
  s.className='show '+kind;s.textContent=msg;
  if(list&&list.length){const ul=document.createElement('ul');list.forEach(x=>{const li=document.createElement('li');li.textContent=x;ul.appendChild(li);});s.appendChild(ul);}
}

function autoLayout(){
  const B=graph.blocks,depth={};B.forEach(b=>depth[b.id]=0);
  for(let i=0;i<B.length;i++){let ch=false;
    graph.wires.forEach(w=>{if(depth[w.from_block]===undefined||depth[w.to_block]===undefined)return;
      if(depth[w.to_block]<depth[w.from_block]+1){depth[w.to_block]=depth[w.from_block]+1;ch=true;}});
    if(!ch)break;}
  const cols={};B.forEach(b=>{(cols[depth[b.id]]=cols[depth[b.id]]||[]).push(b);});
  Object.keys(cols).forEach(c=>{let y=40;cols[c].forEach(b=>{b.x=40+c*(NODE_W+90);b.y=y;y+=dims(b).h+28;});});
}
function fit(){
  if(!graph.blocks.length){view={x:40,y:30,z:1};applyView();return;}
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  graph.blocks.forEach(b=>{const d=dims(b);x0=Math.min(x0,b.x);y0=Math.min(y0,b.y);x1=Math.max(x1,b.x+d.w);y1=Math.max(y1,b.y+d.h);});
  const cw=canvas.clientWidth,ch=canvas.clientHeight,bw=x1-x0,bh=y1-y0;
  const z=Math.max(.3,Math.min(1,(cw-60)/bw,(ch-60)/bh));
  view.z=z;view.x=(cw-bw*z)/2-x0*z;view.y=Math.max(20,(ch-bh*z)/2)-y0*z;applyView();
}
function zoomAt(cx,cy,f){const r=canvas.getBoundingClientRect(),px=cx-r.left,py=cy-r.top,nz=Math.max(.3,Math.min(1.6,view.z*f));
  view.x=px-(px-view.x)*(nz/view.z);view.y=py-(py-view.y)*(nz/view.z);view.z=nz;applyView();}

canvas.addEventListener('pointerdown',e=>{
  if(e.target.closest('.node')||e.target.closest('.hit'))return;
  select(null);canvas.setPointerCapture(e.pointerId);canvas.classList.add('panning');
  const sx=e.clientX,sy=e.clientY,ox=view.x,oy=view.y;
  const mv=ev=>{view.x=ox+ev.clientX-sx;view.y=oy+ev.clientY-sy;applyView();};
  const up=()=>{canvas.classList.remove('panning');canvas.removeEventListener('pointermove',mv);canvas.removeEventListener('pointerup',up);};
  canvas.addEventListener('pointermove',mv);canvas.addEventListener('pointerup',up);
});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoomAt(e.clientX,e.clientY,e.deltaY<0?1.1:.9);},{passive:false});
document.addEventListener('keydown',e=>{
  if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName))return;
  if((e.key==='Delete'||e.key==='Backspace')&&sel){
    if(sel.kind==='block'){const id=sel.id;graph.blocks=graph.blocks.filter(b=>b.id!==id);graph.wires=graph.wires.filter(w=>w.from_block!==id&&w.to_block!==id);}
    else graph.wires.splice(sel.i,1);
    sel=null;render();renderPanel();scheduleGen();}
});

function addBlock(type){
  const base=type.toLowerCase().slice(0,4);let n=1;while(byId(base+n))n++;
  const c=toWorld(canvas.getBoundingClientRect().left+canvas.clientWidth/2,canvas.getBoundingClientRect().top+canvas.clientHeight/2);
  const b={id:base+n,type,settings:{},x:Math.round(c.x-NODE_W/2+(Math.random()*40-20)),y:Math.round(c.y-30+(Math.random()*40-20))};
  LIB[type].settings.forEach(s=>b.settings[s.key]=s.default);graph.blocks.push(b);render();select({kind:'block',id:b.id});scheduleGen();
}

let genT;function scheduleGen(){clearTimeout(genT);genT=setTimeout(gen,250);}
async function gen(){
  try{const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})});
    const d=await r.json();$('#code').textContent=d.code;const ul=$('#codeErrors');ul.innerHTML='';
    d.errors.forEach(e=>{const li=document.createElement('li');li.textContent=e;ul.appendChild(li);});ul.style.display=d.errors.length?'block':'none';}catch(e){}
}

const CFG_KEY='blockEditorCfg';
function getCfg(){const c=JSON.parse(localStorage.getItem(CFG_KEY)||'{}');
  return Object.assign({temperature:0.2,auto_test:true},c);}
function openSettings(){const c=getCfg();$('#cTemp').value=c.temperature;$('#cTest').checked=c.auto_test!==false;$('#dlg').showModal();}
function saveSettings(){localStorage.setItem(CFG_KEY,JSON.stringify({
  temperature:+$('#cTemp').value,auto_test:$('#cTest').checked}));$('#dlg').close();}

async function refreshModel(){
  try{const d=await(await fetch('/api/status')).json();
    $('#dot').className='dot'+(d.loaded?' on':'');
    $('#modelTxt').textContent=d.loaded?('GGUFllama: '+d.model):(d.reachable?'GGUFllama: no model loaded':'GGUFllama not reachable');
    $('#modelTxt').title=d.reason||'';}catch(e){}
}

async function ask(){
  const prompt=$('#prompt').value.trim();
  if(!prompt){setStatus('warn','Describe the app you want first.');return;}
  const cfg=getCfg();$('#askBtn').disabled=true;
  setStatus('busy',cfg.auto_test===false?'Asking GGUFllama and drawing the graph…':'Asking GGUFllama, drawing the graph and test-running the app…');
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,cfg})});
    const d=await r.json();if(!r.ok||d.error)throw new Error(d.error||'Request failed');
    graph=d.graph;autoLayout();sel=null;render();renderPanel();fit();scheduleGen();
    const n=`${graph.blocks.length} blocks, ${graph.wires.length} wires.`;
    const tries=d.attempts>1?` (took ${d.attempts} tries)`:'';
    if(d.errors.length)setStatus('warn',`${n} Some problems remain after ${d.attempts} tries. Fix them by hand:`,d.errors);
    else if(d.test)showTest(d.test,n,tries);
    else setStatus('ok',`${n} All ports valid${tries}.`);
  }catch(err){setStatus('bad',err.message);}
  finally{$('#askBtn').disabled=false;refreshModel();}
}

function showTest(t,pre,tries){
  pre=pre?pre+' ':'';tries=tries||'';
  if(t.failures&&t.failures.length)setStatus('warn',`${pre}It still fails its test${tries}. Fix these by hand or ask again:`,t.failures);
  else if(t.ran)setStatus('ok',`${pre}Built, ran headless and tested clean${tries}.${t.notes&&t.notes.length?' Notes:':''}`,t.notes);
  else setStatus('warn',`${pre}All ports valid, but the test was skipped: ${t.skipped||'unknown reason'}`);
}
$('#testBtn').onclick=async()=>{
  setStatus('busy','Running the app headless to test it…');
  try{const r=await fetch('/api/test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})});
    const t=await r.json();if(!r.ok||t.error)throw new Error(t.error||'Test failed');
    const sim=(t.stubbed||[]).length?' Simulated: '+t.stubbed.join(', ')+'.':'';
    showTest(t,'');if(sim)$('#runOut').textContent='Network and files were simulated during the test.'+sim;
  }catch(err){setStatus('bad',err.message);}
};
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}

$('#askBtn').onclick=ask;$('#prompt').addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
$('#layoutBtn').onclick=()=>{autoLayout();render();fit();};$('#fitBtn').onclick=fit;
$('#zoomIn').onclick=()=>zoomAt(canvas.getBoundingClientRect().left+canvas.clientWidth/2,canvas.getBoundingClientRect().top+canvas.clientHeight/2,1.2);
$('#zoomOut').onclick=()=>zoomAt(canvas.getBoundingClientRect().left+canvas.clientWidth/2,canvas.getBoundingClientRect().top+canvas.clientHeight/2,.83);
$('#saveBtn').onclick=()=>download('graph.json',JSON.stringify(graph,null,2),'application/json');
$('#loadBtn').onclick=()=>$('#fileIn').click();
$('#fileIn').onchange=e=>{const f=e.target.files[0];if(!f)return;const rd=new FileReader();rd.onload=()=>{
  try{const g=JSON.parse(rd.result);if(!Array.isArray(g.blocks)||!Array.isArray(g.wires))throw 0;
    g.blocks.forEach(b=>{b.settings=b.settings||{};});graph=g;if(graph.blocks.some(b=>b.x===undefined))autoLayout();
    sel=null;render();renderPanel();fit();scheduleGen();setStatus('ok','Graph loaded.');}
  catch(x){setStatus('bad','That file isn\'t a block graph.');}};rd.readAsText(f);e.target.value='';};
$('#clearBtn').onclick=()=>{if(!confirm('Clear the whole graph?'))return;graph={blocks:[],wires:[]};sel=null;render();renderPanel();scheduleGen();setStatus();};
$('#copyBtn').onclick=async()=>{try{await navigator.clipboard.writeText($('#code').textContent);setStatus('ok','Code copied.');}catch(e){setStatus('warn','Copy failed. Select the code and copy it by hand.');}};
$('#savePyBtn').onclick=()=>download('app.py',$('#code').textContent,'text/x-python');
$('#runBtn').onclick=async()=>{const out=$('#runOut');out.textContent='Starting…';
  const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:$('#code').textContent})});
  const d=await r.json();out.textContent=d.message+(d.output?'\n'+d.output:'');};
$('#settingsBtn').onclick=openSettings;$('#dlgSave').onclick=()=>{saveSettings();refreshModel();};$('#dlgCancel').onclick=()=>$('#dlg').close();

(async function init(){
  const d=await(await fetch('/api/library')).json();LIB=d.lib;
  const pal=$('#palette');Object.keys(LIB).forEach(t=>{const c=document.createElement('button');c.className='chip '+LIB[t].cat;c.textContent='+ '+t;c.title=LIB[t].desc;c.onclick=()=>addBlock(t);pal.appendChild(c);});
  graph=JSON.parse(JSON.stringify(d.demo));autoLayout();render();renderPanel();applyView();requestAnimationFrame(fit);scheduleGen();refreshModel();setInterval(refreshModel,5000);
  window.addEventListener('resize',()=>{});
})();
</script>
</body>
</html>
"""


# ==============================================================================================
#  DRAWING (built in) - watch the loaded model draw. The model writes an SVG; SvgLiveRenderer
#  parses it while it streams and paints every shape onto a canvas the moment it is complete,
#  so the picture builds up stroke by stroke. Follow-up prompts ("make the sky darker") send the
#  current drawing back to the model, which rewrites it. Title-bar right-click menu ->
#  "Drawing". Uses the Local model that is already loaded; nothing extra to install.
# ==============================================================================================

DRAWING_SYSTEM_PROMPT = (
    "You are an SVG illustrator. Reply with ONE complete SVG and nothing else - no explanation "
    "and no markdown fences. Use viewBox=\"0 0 800 600\". Paint from back to front: the "
    "background first, then large shapes, then details. Put every element on its own line. "
    "Allowed elements: rect, circle, ellipse, line, polyline, polygon, path, text, g (with "
    "transform), and linearGradient with stop elements inside defs. Give every shape explicit "
    "fill and stroke attributes with plain colors like fill=\"#3a7bd5\" (fill=\"none\" for outlines "
    "only). Do not use CSS classes, <style>, filters, masks, clip paths, patterns or <use>. Keep "
    "the drawing bold, simple and recognisable, with sensible proportions."
)

_SVG_ATTR_RE = re.compile(r'([\w:-]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')')
_SVG_NUM_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
_SVG_PATH_RE = re.compile(r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def _svg_num(value, default=0.0):
    m = _SVG_NUM_RE.search(value or "")
    return float(m.group()) if m else default


def _mat_mul(m, n):
    a1, b1, c1, d1, e1, f1 = m
    a2, b2, c2, d2, e2, f2 = n
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def _svg_transform(text):
    """'translate(10 20) rotate(45)' -> affine matrix (a, b, c, d, e, f)."""
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, args in re.findall(r'(\w+)\s*\(([^)]*)\)', text or ""):
        v = [float(x) for x in _SVG_NUM_RE.findall(args)]
        name = name.lower()
        t = None
        if name == "translate" and v:
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale" and v:
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate" and v:
            r = math.radians(v[0])
            cs, sn = math.cos(r), math.sin(r)
            t = (cs, sn, -sn, cs, 0, 0)
            if len(v) >= 3:
                cx, cy = v[1], v[2]
                t = _mat_mul(_mat_mul((1, 0, 0, 1, cx, cy), t), (1, 0, 0, 1, -cx, -cy))
        elif name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "skewx" and v:
            t = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif name == "skewy" and v:
            t = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        if t:
            m = _mat_mul(m, t)
    return m


def _apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def _flatten_path(d):
    """SVG path data -> [(points, closed)]. Curves are flattened; arcs become straight lines to
    their end point (good enough for a live preview)."""
    tokens = _SVG_PATH_RE.findall(d or "")
    subs, cur = [], []
    x = y = sx = sy = 0.0
    last_c = last_q = None
    cmd, i = None, 0

    def num():
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    def more():
        return i < len(tokens) and not tokens[i].isalpha()

    def cubic(p0, p1, p2, p3):
        for k in range(1, 13):
            t = k / 12.0
            u = 1 - t
            cur.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                        u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))

    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
        elif cmd is None:
            break
        rel = cmd.islower()
        c = cmd.upper()
        try:
            if c == "Z":
                if cur:
                    subs.append((cur, True))
                    cur = []
                x, y = sx, sy
                last_c = last_q = None
                continue
            if not more():
                if c != "Z":
                    continue
            if c == "M":
                nx, ny = num(), num()
                if rel:
                    nx, ny = x + nx, y + ny
                if cur:
                    subs.append((cur, False))
                x, y = sx, sy = nx, ny
                cur = [(x, y)]
                cmd = "l" if rel else "L"
                last_c = last_q = None
            elif c == "L":
                nx, ny = num(), num()
                x, y = (x + nx, y + ny) if rel else (nx, ny)
                if not cur:
                    cur = [(sx, sy)]
                cur.append((x, y))
                last_c = last_q = None
            elif c == "H":
                nx = num()
                x = x + nx if rel else nx
                if not cur:
                    cur = [(sx, sy)]
                cur.append((x, y))
                last_c = last_q = None
            elif c == "V":
                ny = num()
                y = y + ny if rel else ny
                if not cur:
                    cur = [(sx, sy)]
                cur.append((x, y))
                last_c = last_q = None
            elif c in ("C", "S"):
                if c == "C":
                    x1, y1 = num(), num()
                    if rel:
                        x1, y1 = x + x1, y + y1
                else:
                    x1, y1 = (2 * x - last_c[0], 2 * y - last_c[1]) if last_c else (x, y)
                x2, y2, nx, ny = num(), num(), num(), num()
                if rel:
                    x2, y2, nx, ny = x + x2, y + y2, x + nx, y + ny
                if not cur:
                    cur = [(x, y)]
                cubic((x, y), (x1, y1), (x2, y2), (nx, ny))
                last_c, last_q = (x2, y2), None
                x, y = nx, ny
            elif c in ("Q", "T"):
                if c == "Q":
                    x1, y1 = num(), num()
                    if rel:
                        x1, y1 = x + x1, y + y1
                else:
                    x1, y1 = (2 * x - last_q[0], 2 * y - last_q[1]) if last_q else (x, y)
                nx, ny = num(), num()
                if rel:
                    nx, ny = x + nx, y + ny
                if not cur:
                    cur = [(x, y)]
                p0 = (x, y)
                cubic(p0, (p0[0] + 2 / 3 * (x1 - p0[0]), p0[1] + 2 / 3 * (y1 - p0[1])),
                      (nx + 2 / 3 * (x1 - nx), ny + 2 / 3 * (y1 - ny)), (nx, ny))
                last_q, last_c = (x1, y1), None
                x, y = nx, ny
            elif c == "A":
                num(); num(); num(); num(); num()
                nx, ny = num(), num()
                x, y = (x + nx, y + ny) if rel else (nx, ny)
                if not cur:
                    cur = [(sx, sy)]
                cur.append((x, y))
                last_c = last_q = None
            else:
                i += 1
        except (IndexError, ValueError):
            break
    if cur:
        subs.append((cur, False))
    return subs


class SvgLiveRenderer:
    """Incremental SVG -> Tk canvas. feed() takes streamed text; every element is painted as soon
    as its closing '>' has arrived. Elements are kept as records, so the canvas can be repainted
    (window resize) or replayed. Supports rect, circle, ellipse, line, polyline, polygon, path,
    text, g/transform, solid colors, gradient fallback (uses a mid stop) and opacity (blended)."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.reset()
        self._replay_job = None

    # ---- state -------------------------------------------------------------------------------
    def reset(self):
        self.buf = ""
        self.pos = 0
        self.records = []
        self.vb = (0.0, 0.0, 800.0, 600.0)
        self.gradients = {}
        self._grad_id = None
        self.stack = [self._base_style()]
        self.canvas.delete("all")
        self._replay_stop()
        self._draw_frame()

    @staticmethod
    def _base_style():
        return {"fill": "#000000", "stroke": "", "stroke-width": 1.0, "opacity": 1.0,
                "fill-opacity": 1.0, "stroke-opacity": 1.0, "font-size": 16.0,
                "font-weight": "normal", "text-anchor": "start", "m": (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)}

    def _viewport(self):
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        minx, miny, w, h = self.vb
        s = min(cw / w, ch / h)
        return s, (cw - w * s) / 2 - minx * s, (ch - h * s) / 2 - miny * s

    def _draw_frame(self):
        s, ox, oy = self._viewport()
        minx, miny, w, h = self.vb
        self.canvas.delete("frame")
        self.canvas.create_rectangle(ox + minx * s, oy + miny * s, ox + (minx + w) * s, oy + (miny + h) * s,
                                     fill="#ffffff", outline="#d1d5db", tags="frame")
        self.canvas.tag_lower("frame")

    def redraw(self, upto=None):
        self._replay_stop()
        self.canvas.delete("all")
        self._draw_frame()
        for rec in (self.records if upto is None else self.records[:upto]):
            self._paint(rec)

    # ---- replay ------------------------------------------------------------------------------
    def replay(self, step_ms=35, on_done=None):
        self._replay_stop()
        self.canvas.delete("all")
        self._draw_frame()
        n = len(self.records)

        def step(k):
            if k >= n:
                self._replay_job = None
                if on_done:
                    on_done()
                return
            self._paint(self.records[k])
            self._replay_job = self.canvas.after(step_ms, step, k + 1)
        step(0)

    def _replay_stop(self):
        if getattr(self, "_replay_job", None) is not None:
            try:
                self.canvas.after_cancel(self._replay_job)
            except Exception:                       # noqa: BLE001
                pass
            self._replay_job = None

    # ---- colors ------------------------------------------------------------------------------
    def _rgb(self, color):
        try:
            r, g, b = self.canvas.winfo_rgb(color)
            return r // 257, g // 257, b // 257
        except tk.TclError:
            return None

    def _color(self, value, opacity=1.0):
        v = (value or "").strip()
        low = v.lower()
        if low in ("", "none", "transparent"):
            return ""
        if low == "currentcolor":
            v = "#000000"
        m = re.match(r'url\(\s*#([^)\s]+)\s*\)', v)
        if m:
            v = self.gradients.get(m.group(1), "#888888")
        m = re.match(r'rgba?\(([^)]*)\)', v, re.I)
        if m:
            parts = [p.strip() for p in m.group(1).split(",")]
            try:
                vals = [int(float(p[:-1]) * 2.55) if p.endswith("%") else int(float(p)) for p in parts[:3]]
                v = "#%02x%02x%02x" % tuple(max(0, min(255, x)) for x in vals)
                if len(parts) > 3:
                    opacity *= float(parts[3])
            except ValueError:
                v = "#000000"
        elif re.fullmatch(r'#[0-9a-fA-F]{3}', v):
            v = "#" + "".join(ch * 2 for ch in v[1:])
        elif re.fullmatch(r'#[0-9a-fA-F]{8}', v):
            opacity *= int(v[7:9], 16) / 255.0
            v = v[:7]
        rgb = self._rgb(v)
        if rgb is None:
            rgb = (0, 0, 0)
        if opacity < 1.0:                            # Tk has no alpha: blend toward white
            rgb = tuple(int(c * opacity + 255 * (1 - opacity)) for c in rgb)
        return "#%02x%02x%02x" % rgb

    # ---- streaming parse ---------------------------------------------------------------------
    def feed(self, text):
        if not text:
            return 0
        self.buf += text
        before = len(self.records)
        while True:
            i = self.buf.find("<", self.pos)
            if i == -1:
                break
            if self.buf.startswith("<!--", i):
                j = self.buf.find("-->", i)
                if j == -1:
                    break
                self.pos = j + 3
                continue
            j = self.buf.find(">", i)
            if j == -1:
                break
            tag = self.buf[i:j + 1]
            end = j + 1
            if re.match(r'<text\b', tag, re.I) and not tag.endswith("/>"):
                k = self.buf.lower().find("</text>", j)
                if k == -1:
                    break
                inner = re.sub(r'<[^>]+>', '', self.buf[j + 1:k])
                end = k + len("</text>")
                self._element("text", tag, inner)
            else:
                self._tag(tag)
            self.pos = end
        return len(self.records) - before

    def _tag(self, tag):
        if tag.startswith("</"):
            name = tag[2:-1].strip().lower()
            if name == "g" and len(self.stack) > 1:
                self.stack.pop()
            elif name == "lineargradient" or name == "radialgradient":
                self._grad_id = None
            return
        m = re.match(r'<\s*([\w:-]+)', tag)
        if not m:
            return
        name = m.group(1).lower()
        selfclose = tag.endswith("/>")
        if name == "g":
            if not selfclose:
                self.stack.append(self._style(tag))
            return
        if name == "svg":
            self._root(tag)
            return
        if name in ("lineargradient", "radialgradient"):
            attrs = self._attrs(tag)
            self._grad_id = attrs.get("id")
            self._grad_stops = []
            if selfclose:
                self._grad_id = None
            return
        if name == "stop" and self._grad_id:
            attrs = self._attrs(tag)
            style = attrs.get("style", "")
            col = attrs.get("stop-color") or (re.search(r'stop-color\s*:\s*([^;]+)', style) or [None, None])[1]
            if col:
                self._grad_stops = getattr(self, "_grad_stops", []) + [col.strip()]
                stops = self._grad_stops
                self.gradients[self._grad_id] = stops[len(stops) // 2] if len(stops) > 1 else stops[0]
            return
        if name in ("rect", "circle", "ellipse", "line", "polyline", "polygon", "path"):
            self._element(name, tag, None)

    @staticmethod
    def _attrs(tag):
        return {k.lower(): (a if a is not None and a != "" else b or "")
                for k, a, b in _SVG_ATTR_RE.findall(tag)}

    def _root(self, tag):
        a = self._attrs(tag)
        vb = [float(x) for x in _SVG_NUM_RE.findall(a.get("viewbox", ""))]
        if len(vb) == 4 and vb[2] > 0 and vb[3] > 0:
            self.vb = tuple(vb)
        elif a.get("width") and a.get("height") and _svg_num(a["width"]) > 0 and _svg_num(a["height"]) > 0:
            self.vb = (0.0, 0.0, _svg_num(a["width"]), _svg_num(a["height"]))
        self._draw_frame()
        if a.get("fill"):
            self.stack[0]["fill"] = a["fill"]

    def _style(self, tag):
        a = self._attrs(tag)
        st = dict(self.stack[-1])
        props = dict(a)
        for part in a.get("style", "").split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                props[k.strip().lower()] = v.strip()
        for k in ("fill", "stroke", "text-anchor", "font-weight"):
            if k in props:
                st[k] = props[k]
        for k in ("stroke-width", "font-size"):
            if k in props:
                st[k] = _svg_num(props[k], st[k])
        for k in ("opacity", "fill-opacity", "stroke-opacity"):
            if k in props:
                st[k] = max(0.0, min(1.0, _svg_num(props[k], 1.0)))
        if "transform" in props:
            st["m"] = _mat_mul(st["m"], _svg_transform(props["transform"]))
        return st

    def _element(self, name, tag, inner):
        a = self._attrs(tag)
        st = self._style(tag)
        m = st["m"]
        f = lambda k, d=0.0: _svg_num(a.get(k), d)
        polys = []                                   # [(points, closed)]
        if name == "rect":
            x, y, w, h = f("x"), f("y"), f("width"), f("height")
            r = min(f("rx", f("ry")), w / 2, h / 2)
            if r > 0:
                pts = []
                for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0),
                                   (x + r, y + h - r, 90), (x + r, y + r, 180)):
                    for k in range(7):
                        t = math.radians(a0 + 90 * k / 6)
                        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
            else:
                pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            polys.append((pts, True))
        elif name in ("circle", "ellipse"):
            cx, cy = f("cx"), f("cy")
            rx = f("r") if name == "circle" else f("rx")
            ry = rx if name == "circle" else f("ry")
            polys.append(([(cx + rx * math.cos(2 * math.pi * k / 48), cy + ry * math.sin(2 * math.pi * k / 48))
                           for k in range(48)], True))
        elif name == "line":
            polys.append(([(f("x1"), f("y1")), (f("x2"), f("y2"))], False))
        elif name in ("polyline", "polygon"):
            v = [float(x) for x in _SVG_NUM_RE.findall(a.get("points", ""))]
            pts = list(zip(v[0::2], v[1::2]))
            if len(pts) >= 2:
                polys.append((pts, name == "polygon"))
        elif name == "path":
            polys.extend(_flatten_path(a.get("d", "")))
        elif name == "text":
            txt = (inner or "").strip()
            if txt:
                px, py = _apply(m, f("x"), f("y"))
                scale = math.sqrt(abs(m[0] * m[3] - m[1] * m[2])) or 1.0
                self._add({"type": "text", "x": px, "y": py, "text": txt, "size": st["font-size"] * scale,
                           "bold": str(st["font-weight"]).lower() in ("bold", "700", "800", "900"),
                           "anchor": st["text-anchor"],
                           "fill": self._color(st["fill"], st["opacity"] * st["fill-opacity"])})
            return
        scale = math.sqrt(abs(m[0] * m[3] - m[1] * m[2])) or 1.0
        fill_ok = name != "line"
        fill = self._color(st["fill"], st["opacity"] * st["fill-opacity"]) if fill_ok else ""
        stroke = self._color(st["stroke"], st["opacity"] * st["stroke-opacity"])
        for pts, closed in polys:
            if len(pts) < 2:
                continue
            self._add({"type": "poly", "pts": [_apply(m, px, py) for px, py in pts], "closed": closed,
                       "fill": fill, "stroke": stroke, "sw": st["stroke-width"] * scale})

    def _add(self, rec):
        self.records.append(rec)
        self._paint(rec)

    # ---- painting ----------------------------------------------------------------------------
    def _paint(self, rec):
        s, ox, oy = self._viewport()
        c = self.canvas
        if rec["type"] == "text":
            anchor = {"middle": "s", "end": "se"}.get(rec["anchor"], "sw")
            size = max(6, int(rec["size"] * s))
            c.create_text(ox + rec["x"] * s, oy + rec["y"] * s, text=rec["text"], anchor=anchor,
                          fill=rec["fill"] or "#000000",
                          font=("Segoe UI", -size, "bold" if rec["bold"] else "normal"))
            return
        flat = []
        for px, py in rec["pts"]:
            flat += [ox + px * s, oy + py * s]
        width = max(1, rec["sw"] * s) if rec["stroke"] else 1
        if len(rec["pts"]) >= 3 and rec["fill"]:
            c.create_polygon(*flat, fill=rec["fill"], outline="")
        if rec["stroke"]:
            if rec["closed"] and len(rec["pts"]) >= 3:
                c.create_line(*(flat + flat[:2]), fill=rec["stroke"], width=width, joinstyle=tk.ROUND)
            else:
                c.create_line(*flat, fill=rec["stroke"], width=width, joinstyle=tk.ROUND, capstyle=tk.ROUND)


def extract_svg(raw):
    """Best-effort clean SVG document out of the model's raw reply ('' if there is none)."""
    low = raw.lower()
    i = low.find("<svg")
    if i == -1:
        shapes = re.findall(r'<(?:rect|circle|ellipse|line|polyline|polygon|path|text)\b', low)
        if not shapes:
            return ""
        first = min(low.find(t) for t in ("<rect", "<circle", "<ellipse", "<line", "<polyline",
                                          "<polygon", "<path", "<text") if low.find(t) != -1)
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">\n'
                + raw[first:].replace("```", "").strip() + "\n</svg>\n")
    j = low.rfind("</svg>")
    svg = raw[i:j + 6] if j != -1 else raw[i:].replace("```", "").rstrip() + "\n</svg>"
    head = svg[:svg.find(">") + 1]
    if "xmlns" not in head.lower():
        svg = head[:-1].rstrip("/ ") + ' xmlns="http://www.w3.org/2000/svg">' + svg[len(head):]
    return svg.strip() + "\n"


class DrawingWindow:
    def __init__(self, root, theme, app):
        self.app, self.t = app, theme
        t = theme
        self.win = tk.Toplevel(root)
        self.win.title("Drawing")
        self.win.geometry("900x720")
        self.win.configure(bg=t["bg"])
        font = (t["family"], t["size"])
        small = (t["family"], t["small"])

        top = tk.Frame(self.win, bg=t["bg"])
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        self.entry = tk.Entry(top, font=font, relief=tk.FLAT, bg=t["input"], fg=t["text"],
                              insertbackground=t["text"], highlightthickness=1,
                              highlightbackground=t["neutral"], highlightcolor=t["accent"])
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.entry.bind("<Return>", lambda _e: self._start())
        self.draw_btn = self._btn(top, "Draw", self._start, accent=True)
        self.draw_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.stop_btn = self._btn(top, "Stop", self._stop)
        self.stop_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.stop_btn.config(state=tk.DISABLED)
        self.think_var = tk.BooleanVar(value=bool(getattr(app, "show_thinking", True)))
        tk.Checkbutton(top, text="Show thinking", variable=self.think_var, command=self._toggle_thinking,
                       bg=t["bg"], fg=t["muted"], activebackground=t["bg"], selectcolor=t["input"],
                       font=small, cursor="hand2", bd=0, highlightthickness=0).pack(side=tk.LEFT, padx=(10, 0))

        self.canvas = tk.Canvas(self.win, bg=t["bg"], highlightthickness=0)
        self.renderer = SvgLiveRenderer(self.canvas)
        self.canvas.bind("<Configure>", self._on_resize)
        self._resize_job = None

        bottom = tk.Frame(self.win, bg=t["bg"])
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        self.think_frame = tk.Frame(self.win, bg=t["bg"])
        tk.Label(self.think_frame, text="Thinking", bg=t["bg"], fg=t["muted"], font=small,
                 anchor="w").pack(fill=tk.X)
        box = tk.Frame(self.think_frame, bg=t["card"], highlightthickness=1, highlightbackground=t["neutral"])
        box.pack(fill=tk.X)
        self.think_text = tk.Text(box, height=6, wrap=tk.WORD, bg=t["card"], fg=t["muted"], relief=tk.FLAT,
                                  bd=0, padx=8, pady=6, font=(t["family"], t["small"], "italic"),
                                  state=tk.DISABLED)
        think_sb = tk.Scrollbar(box, command=self.think_text.yview)
        self.think_text.config(yscrollcommand=think_sb.set)
        think_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.think_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if self.think_var.get():
            self.think_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6))
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.status = tk.Label(bottom, text="Describe something to draw.", bg=t["bg"], fg=t["muted"],
                               font=small, anchor="w")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        for label, cmd in (("New drawing", self._new), ("Replay", self._replay),
                           ("Copy SVG", self._copy), ("Save SVG\u2026", self._save)):
            self._btn(bottom, label, cmd).pack(side=tk.RIGHT, padx=(6, 0))

        self.messages = []          # follow-up history: user / assistant (SVG) turns
        self.svg_text = ""
        self.raw = ""
        self.busy = False
        self._stop_flag = threading.Event()
        self._thinking = 0
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.entry.focus_set()

    def _btn(self, parent, text, cmd, accent=False):
        t = self.t
        bg, fg = (t["accent"], t["on_accent"]) if accent else (t["neutral"], t["text"])
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, activebackground=bg,
                         activeforeground=fg, relief=tk.FLAT, bd=0, padx=10, pady=3, cursor="hand2",
                         font=(t["family"], t["small"]))

    def _toggle_thinking(self):
        if self.think_var.get():
            self.think_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6), before=self.canvas)
        else:
            self.think_frame.pack_forget()

    def _think_clear(self):
        self.think_text.config(state=tk.NORMAL)
        self.think_text.delete("1.0", tk.END)
        self.think_text.config(state=tk.DISABLED)

    def _think_add(self, text):
        self.think_text.config(state=tk.NORMAL)
        self.think_text.insert(tk.END, text)
        self.think_text.see(tk.END)
        self.think_text.config(state=tk.DISABLED)

    def _on_close(self):
        self._stop_flag.set()
        self.renderer._replay_stop()
        self.win.destroy()

    def _on_resize(self, _e=None):
        if self._resize_job:
            self.canvas.after_cancel(self._resize_job)
        self._resize_job = self.canvas.after(120, self._do_resize)

    def _do_resize(self):
        self._resize_job = None
        if not self.busy:
            self.renderer.redraw()

    def _set_status(self, text, error=False):
        try:
            self.status.config(text=text, fg="#dc2626" if error else self.t["muted"])
        except tk.TclError:
            pass

    def _ui(self, fn, *args):
        try:
            self.win.after(0, fn, *args)
        except Exception:                            # noqa: BLE001 - window closed
            pass

    # ---- actions -----------------------------------------------------------------------------
    def _new(self):
        if self.busy:
            return
        self.messages, self.svg_text, self.raw = [], "", ""
        self.renderer.reset()
        self._set_status("Describe something to draw.")

    def _replay(self):
        if self.busy or not self.renderer.records:
            return
        self._set_status("Replaying\u2026")
        self.renderer.replay(on_done=lambda: self._set_status(f"{len(self.renderer.records)} shapes."))

    def _copy(self):
        if self.svg_text:
            self.win.clipboard_clear()
            self.win.clipboard_append(self.svg_text)
            self._set_status("SVG copied.")

    def _save(self):
        if not self.svg_text:
            self._set_status("Nothing to save yet.")
            return
        self.app.save_text_content(self.svg_text, default_filename="drawing.svg", dialog_title="Save Drawing")

    def _stop(self):
        self._stop_flag.set()
        self._set_status("Stopping\u2026")

    def _start(self):
        if self.busy:
            return
        prompt = self.entry.get().strip()
        if not prompt:
            return
        try:
            h = self.app.http_bridge._handle()
        except BridgeError as exc:
            self._set_status(str(exc), error=True)
            return
        editing = bool(self.messages)
        user_msg = (f"Change the drawing: {prompt}\nReturn the complete updated SVG." if editing
                    else f"Draw: {prompt}")
        history = self.messages[-4:]                 # keep the last two exchanges
        messages = [{"role": "system", "content": DRAWING_SYSTEM_PROMPT}] + history + \
                   [{"role": "user", "content": user_msg}]
        self.busy = True
        self._stop_flag.clear()
        self.raw = ""
        self._thinking = 0
        self._think_clear()
        self.renderer.reset()
        self.draw_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("Waiting for the model\u2026" if h.gen_lock.locked() else "Starting\u2026")
        threading.Thread(target=self._worker, args=(h, messages, user_msg), daemon=True).start()

    # ---- generation --------------------------------------------------------------------------
    def _worker(self, h, messages, user_msg):
        err = None
        try:
            with h.gen_lock:
                if h.cancel.is_set() or h.llm is None:
                    raise BridgeError(409, "The model was unloaded.")
                try:
                    tpl = h.llm.metadata.get("tokenizer.chat_template", "") or ""
                except Exception:                    # noqa: BLE001
                    tpl = ""
                open_think = bool(re.search(r"add_generation_prompt.*?'<think>\\n'", tpl, re.DOTALL))
                stream = h.llm.create_chat_completion(messages=messages, temperature=0.7,
                                                      max_tokens=4096, stream=True)
                harmony = HarmonyStreamFilter()
                parser = {"mode": "think" if open_think else "pre", "buffer": ""}

                def route(kind, text):
                    if kind == "reasoning":
                        self._ui(self._on_piece, "reasoning", text)
                    else:
                        for k2, t2 in feed_think_state(parser, text):
                            self._ui(self._on_piece, k2, t2)

                for chunk in stream:
                    if self._stop_flag.is_set() or h.cancel.is_set():
                        stream.close()
                        if h.cancel.is_set():
                            raise BridgeError(409, "Stopped \u2014 the model was unloaded.")
                        break
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    r_piece = delta.get("reasoning_content") or delta.get("reasoning")
                    if r_piece:
                        self._ui(self._on_piece, "reasoning", r_piece)
                    piece = delta.get("content")
                    if piece:
                        for kind, text in harmony.feed(piece):
                            route(kind, text)
                for kind, text in harmony.flush():
                    route(kind, text)
                if parser["buffer"]:
                    self._ui(self._on_piece, "reasoning" if parser["mode"] == "think" else "answer",
                             parser["buffer"])
        except Exception as exc:                     # noqa: BLE001
            err = str(exc) or type(exc).__name__
        self._ui(self._on_done, err, user_msg)

    def _on_piece(self, kind, text):
        if kind == "reasoning":
            self._thinking += len(text)
            self._think_add(text)
            if not self.renderer.records:
                self._set_status(f"Thinking\u2026 ({self._thinking} chars)")
            return
        self.raw += text
        self.renderer.feed(text)
        n = len(self.renderer.records)
        if n:
            self._set_status(f"Drawing\u2026 {n} shapes")

    def _on_done(self, err, user_msg):
        self.busy = False
        try:
            self.draw_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
        except tk.TclError:
            return
        stopped = self._stop_flag.is_set()
        svg = extract_svg(self.raw)
        n = len(self.renderer.records)
        if err:
            self._set_status(f"Failed \u2014 {err}", error=True)
        elif not n:
            self._set_status("The model didn't return any drawable shapes. Try again, or a bigger model.",
                             error=True)
        else:
            self._set_status(("Stopped. " if stopped else "Done. ") + f"{n} shapes \u2014 type a change to edit it.")
        if svg and n:
            self.svg_text = svg
            self.messages += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": svg}]
            self.entry.delete(0, tk.END)




# ==============================================================================================
#  MUSIC (built in) - the loaded model composes; you watch the notes appear and press Play.
#  The model writes a small JSON score (tempo + tracks of notes). parse_music() reads it while it
#  streams, the piano roll fills in note by note, and a tiny pure-Python synthesizer (sine /
#  square / triangle / saw + noise drums) renders it to a WAV that plays through winsound on
#  Windows (afplay / aplay / paplay elsewhere). Also saves .wav and .mid. Nothing to install.
#  Title-bar right-click menu -> "Music". Follow-ups ("slower, add a bass line") rewrite the score.
# ==============================================================================================

MUSIC_SYSTEM_PROMPT = (
    "You are a composer. Reply with ONE JSON object and nothing else - no explanation and no "
    "markdown fences. Format: {\"tempo\": 110, \"tracks\": [{\"name\": \"melody\", \"wave\": \"square\", "
    "\"notes\": [{\"p\": \"C4\", \"s\": 0, \"d\": 1, \"v\": 0.8}, {\"p\": \"E4\", \"s\": 1, \"d\": 0.5, "
    "\"v\": 0.8}]}]}. p is a pitch name (C4 is middle C; use # or b for sharps and flats). For a "
    "drums track, p is kick, snare, hat or tom. s is the start time in beats from 0, d is the "
    "length in beats, v is the volume from 0 to 1. wave is one of sine, square, triangle, saw or "
    "drums. Write 2 to 4 tracks (melody, bass, optional chords and drums), written one track "
    "after another, in 4/4 time, 8 to 16 bars, staying in one key with a clear repeating "
    "pattern. Put each note object on its own line. Use at most 160 notes in total."
)

_MUSIC_WAVES = ("sine", "square", "triangle", "saw")
_MUSIC_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d")
_DRUM_MIDI = {"kick": 36, "snare": 38, "hat": 42, "tom": 45}
_DRUM_ROWS = ("hat", "snare", "kick", "tom")
_NOTE_BASE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_NOTES_MARK_RE = re.compile(r'"notes"\s*:\s*\[')


def pitch_to_midi(p):
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        return int(round(p))
    s = str(p).strip()
    m = re.fullmatch(r'([A-Ga-g])([#b\u266f\u266d]?)(-?\d+)', s)
    if not m:
        return int(s) if re.fullmatch(r'\d+', s) else None
    n = _NOTE_BASE[m.group(1).lower()]
    if m.group(2) in ("#", "\u266f"):
        n += 1
    elif m.group(2) in ("b", "\u266d"):
        n -= 1
    return (int(m.group(3)) + 1) * 12 + n


def _drum_kind(p):
    s = str(p).lower()
    for key, kind in (("kick", "kick"), ("bd", "kick"), ("bass", "kick"), ("snare", "snare"),
                      ("sd", "snare"), ("clap", "snare"), ("hat", "hat"), ("hh", "hat"),
                      ("cym", "hat"), ("ride", "hat"), ("crash", "hat"), ("tom", "tom")):
        if key in s:
            return kind
    midi = pitch_to_midi(p)
    if midi is not None:
        return "kick" if midi <= 37 else "snare" if midi <= 40 else "tom" if midi <= 50 else "hat"
    return "hat"


def _first(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _parse_note_obj(txt):
    try:
        d = json.loads(txt)
        if not isinstance(d, dict):
            return None
    except ValueError:
        d = {}
        for k, v in re.findall(r'"?(\w+)"?\s*:\s*("[^"]*"|\'[^\']*\'|[-\w.#]+)', txt):
            d[k.lower()] = v.strip("\"'")
    return d


def parse_music(raw):
    """Tolerant, incremental-friendly parse of the model's JSON score: only notes whose closing
    brace has arrived are included, so calling this on a growing string is safe."""
    tempo = 110.0
    m = re.search(r'"tempo"\s*:\s*(\d+(?:\.\d+)?)', raw)
    if m:
        tempo = max(40.0, min(240.0, float(m.group(1))))
    marks = list(_NOTES_MARK_RE.finditer(raw))[:8]
    tracks = []
    for k, mm in enumerate(marks):
        head_from = marks[k - 1].end() if k else 0
        head = raw[head_from:mm.start()]
        names = re.findall(r'"name"\s*:\s*"([^"]*)"', head)
        waves = re.findall(r'"(?:wave|waveform|instrument)"\s*:\s*"([^"]*)"', head)
        name = names[-1] if names else f"track {k + 1}"
        wave = (waves[-1] if waves else "").lower()
        drum = wave == "drums" or bool(re.search(r'drum|perc', name, re.I)) or wave in ("drum", "percussion")
        if not drum and wave not in _MUSIC_WAVES:
            wave = _MUSIC_WAVES[(k + 1) % len(_MUSIC_WAVES)] if k else "square"
        end = marks[k + 1].start() if k + 1 < len(marks) else len(raw)
        notes = []
        for om in re.finditer(r'\{[^{}]*\}', raw[mm.end():end]):
            d = _parse_note_obj(om.group())
            if not d:
                continue
            p = _first({kk.lower(): vv for kk, vv in d.items()}, "p", "pitch", "note")
            dl = {kk.lower(): vv for kk, vv in d.items()}
            try:
                s = float(_first(dl, "s", "start", "t", "time", default=None))
                dur = float(_first(dl, "d", "dur", "duration", "len", "length", default=1))
                vol = float(_first(dl, "v", "vel", "velocity", "volume", default=0.8))
            except (TypeError, ValueError):
                continue
            if p is None:
                continue
            if vol > 1.0:
                vol = vol / 127.0
            note = {"p": p if isinstance(p, str) else str(p), "s": max(0.0, min(512.0, s)),
                    "d": max(0.05, min(16.0, dur)), "v": max(0.05, min(1.0, vol))}
            if drum:
                note["drum"] = _drum_kind(p)
            else:
                midi = pitch_to_midi(p)
                if midi is None:
                    continue
                note["midi"] = max(21, min(108, midi))
            notes.append(note)
        tracks.append({"name": name, "wave": "drums" if drum else wave, "drum": drum, "notes": notes})
    return {"tempo": tempo, "tracks": tracks}


def score_note_count(score):
    return sum(len(t["notes"]) for t in score["tracks"])


def score_to_json(score):
    return json.dumps({"tempo": score["tempo"], "tracks": [
        {"name": t["name"], "wave": t["wave"],
         "notes": [{"p": n["p"], "s": n["s"], "d": n["d"], "v": n["v"]} for n in t["notes"]]}
        for t in score["tracks"]]}, separators=(",", ":"))


def score_end_beat(score):
    return max((n["s"] + n["d"] for t in score["tracks"] for n in t["notes"]), default=0.0)


# ---- synthesizer -----------------------------------------------------------------------------

def _tone(freq, dur, wave, sr):
    release = 0.08
    total = int((dur + release) * sr)
    inc = freq / sr
    attack = max(1, int(0.006 * sr))
    hold = int(dur * sr)
    out = []
    ph = 0.0
    for i in range(total):
        ph += inc
        if ph >= 1.0:
            ph -= 1.0
        if wave == "sine":
            x = math.sin(6.283185307 * ph)
        elif wave == "square":
            x = 0.5 if ph < 0.5 else -0.5
        elif wave == "saw":
            x = 0.6 * (2.0 * ph - 1.0)
        else:                                        # triangle
            x = 4.0 * abs(ph - 0.5) - 1.0
        t = i / sr
        env = 0.65 + 0.35 * math.exp(-3.0 * t)
        if i < attack:
            env *= i / attack
        if i >= hold:
            env *= max(0.0, 1.0 - (i - hold) / (release * sr))
        out.append(x * env)
    return out


def _drum(kind, sr):
    rnd = random.random
    if kind == "kick":
        n, ph, out = int(0.28 * sr), 0.0, []
        for i in range(n):
            t = i / sr
            ph += (45.0 + 90.0 * math.exp(-t * 28.0)) / sr
            out.append(math.sin(6.283185307 * ph) * math.exp(-t * 13.0) * 1.1)
        return out
    if kind == "snare":
        n = int(0.2 * sr)
        return [((rnd() * 2 - 1) * 0.7 * math.exp(-(i / sr) * 22.0)
                 + math.sin(6.283185307 * 190.0 * i / sr) * 0.4 * math.exp(-(i / sr) * 30.0)) for i in range(n)]
    if kind == "tom":
        n, ph, out = int(0.3 * sr), 0.0, []
        for i in range(n):
            t = i / sr
            ph += (100.0 + 60.0 * math.exp(-t * 16.0)) / sr
            out.append(math.sin(6.283185307 * ph) * math.exp(-t * 9.0))
        return out
    n, prev, out = int(0.07 * sr), 0.0, []           # hat: high-passed noise burst
    for i in range(n):
        r = rnd() * 2 - 1
        out.append((r - prev) * 0.35 * math.exp(-(i / sr) * 60.0))
        prev = r
    return out


def render_wav(score, sr=22050):
    """Score -> 16-bit mono WAV bytes."""
    spb = 60.0 / score["tempo"]
    total = int((score_end_beat(score) * spb + 1.0) * sr)
    buf = [0.0] * total
    drum_cache = {}
    for t in score["tracks"]:
        for n in t["notes"]:
            start = int(n["s"] * spb * sr)
            if t["drum"]:
                kind = n["drum"]
                if kind not in drum_cache:
                    drum_cache[kind] = _drum(kind, sr)
                samples = drum_cache[kind]
            else:
                samples = _tone(440.0 * 2 ** ((n["midi"] - 69) / 12.0), n["d"] * spb, t["wave"], sr)
            v = n["v"] * 0.45
            end = min(total, start + len(samples))
            for i in range(start, end):
                buf[i] += samples[i - start] * v
    peak = max((abs(x) for x in buf), default=0.0)
    scale = 0.92 / peak if peak > 0.92 else 1.0
    pcm = array.array("h", (int(max(-1.0, min(1.0, x * scale)) * 32767) for x in buf))
    if sys.byteorder == "big":
        pcm.byteswap()
    bio = io.BytesIO()
    with _wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return bio.getvalue()


def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def render_midi(score):
    """Score -> Standard MIDI File bytes (format 1, one track per score track, drums on ch 10)."""
    ppq = 480
    programs = {"sine": 11, "square": 80, "triangle": 73, "saw": 81}

    def chunk(tag, data):
        return tag + struct.pack(">I", len(data)) + data
    tempo_us = int(60000000 / score["tempo"])
    chunks = [chunk(b"MTrk", b"\x00\xff\x51\x03" + tempo_us.to_bytes(3, "big") + b"\x00\xff\x2f\x00")]
    ch = 0
    for t in score["tracks"]:
        if t["drum"]:
            c = 9
        else:
            if ch == 9:
                ch += 1
            c = min(ch, 15)
            ch += 1
        events = []
        if not t["drum"]:
            events.append((0, 0, bytes([0xC0 | c, programs.get(t["wave"], 80)])))
        for n in t["notes"]:
            key = _DRUM_MIDI[n["drum"]] if t["drum"] else n["midi"]
            on = int(n["s"] * ppq)
            off = on + max(1, int(n["d"] * ppq))
            vel = max(1, min(127, int(n["v"] * 127)))
            events.append((on, 1, bytes([0x90 | c, key, vel])))
            events.append((off, 0, bytes([0x80 | c, key, 0])))
        events.sort(key=lambda e: (e[0], e[1]))
        name = t["name"].encode("ascii", "replace")[:40]
        data = b"\x00\xff\x03" + bytes([len(name)]) + name
        last = 0
        for tick, _o, msg in events:
            data += _vlq(tick - last) + msg
            last = tick
        data += b"\x00\xff\x2f\x00"
        chunks.append(chunk(b"MTrk", data))
    return chunk(b"MThd", struct.pack(">HHH", 1, len(chunks), ppq)) + b"".join(chunks)


class MusicWindow:
    def __init__(self, root, theme, app):
        self.app, self.t = app, theme
        t = theme
        self.win = tk.Toplevel(root)
        self.win.title("Music")
        self.win.geometry("900x720")
        self.win.configure(bg=t["bg"])
        font = (t["family"], t["size"])
        small = (t["family"], t["small"])

        top = tk.Frame(self.win, bg=t["bg"])
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        self.entry = tk.Entry(top, font=font, relief=tk.FLAT, bg=t["input"], fg=t["text"],
                              insertbackground=t["text"], highlightthickness=1,
                              highlightbackground=t["neutral"], highlightcolor=t["accent"])
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.entry.bind("<Return>", lambda _e: self._start())
        self.go_btn = self._btn(top, "Compose", self._start, accent=True)
        self.go_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.stop_btn = self._btn(top, "Stop", self._stop)
        self.stop_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.stop_btn.config(state=tk.DISABLED)
        self.think_var = tk.BooleanVar(value=bool(getattr(app, "show_thinking", True)))
        tk.Checkbutton(top, text="Show thinking", variable=self.think_var, command=self._toggle_thinking,
                       bg=t["bg"], fg=t["muted"], activebackground=t["bg"], selectcolor=t["input"],
                       font=small, cursor="hand2", bd=0, highlightthickness=0).pack(side=tk.LEFT, padx=(10, 0))

        self.canvas = tk.Canvas(self.win, bg="#ffffff", highlightthickness=1, highlightbackground=t["neutral"])
        bottom = tk.Frame(self.win, bg=t["bg"])
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        self.think_frame = tk.Frame(self.win, bg=t["bg"])
        tk.Label(self.think_frame, text="Thinking", bg=t["bg"], fg=t["muted"], font=small,
                 anchor="w").pack(fill=tk.X)
        box = tk.Frame(self.think_frame, bg=t["card"], highlightthickness=1, highlightbackground=t["neutral"])
        box.pack(fill=tk.X)
        self.think_text = tk.Text(box, height=5, wrap=tk.WORD, bg=t["card"], fg=t["muted"], relief=tk.FLAT,
                                  bd=0, padx=8, pady=6, font=(t["family"], t["small"], "italic"),
                                  state=tk.DISABLED)
        sb = tk.Scrollbar(box, command=self.think_text.yview)
        self.think_text.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.think_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if self.think_var.get():
            self.think_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6))
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.canvas.bind("<Configure>", lambda _e: self._schedule_redraw())

        self.status = tk.Label(bottom, text="Describe the music you want.", bg=t["bg"], fg=t["muted"],
                               font=small, anchor="w")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        for label, cmd in (("New", self._new), ("Save MIDI\u2026", self._save_midi),
                           ("Save WAV\u2026", self._save_wav)):
            self._btn(bottom, label, cmd).pack(side=tk.RIGHT, padx=(6, 0))
        self.play_btn = self._btn(bottom, "\u25B6 Play", self._toggle_play, accent=True)
        self.play_btn.pack(side=tk.RIGHT, padx=(6, 0))

        self.messages = []
        self.score = {"tempo": 110.0, "tracks": []}
        self.raw = ""
        self.busy = False
        self._stop_flag = threading.Event()
        self._thinking = 0
        self._wav = None
        self._redraw_job = None
        self._playing = False
        self._play_start = 0.0
        self._play_len = 0.0
        self._proc = None
        self._tick_job = None
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.entry.focus_set()
        self._redraw()

    # ---- small UI helpers --------------------------------------------------------------------
    def _btn(self, parent, text, cmd, accent=False):
        t = self.t
        bg, fg = (t["accent"], t["on_accent"]) if accent else (t["neutral"], t["text"])
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, activebackground=bg,
                         activeforeground=fg, relief=tk.FLAT, bd=0, padx=10, pady=3, cursor="hand2",
                         font=(t["family"], t["small"]))

    def _set_status(self, text, error=False):
        try:
            self.status.config(text=text, fg="#dc2626" if error else self.t["muted"])
        except tk.TclError:
            pass

    def _ui(self, fn, *args):
        try:
            self.win.after(0, fn, *args)
        except Exception:                            # noqa: BLE001 - window closed
            pass

    def _toggle_thinking(self):
        if self.think_var.get():
            self.think_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6), before=self.canvas)
        else:
            self.think_frame.pack_forget()

    def _think_clear(self):
        self.think_text.config(state=tk.NORMAL)
        self.think_text.delete("1.0", tk.END)
        self.think_text.config(state=tk.DISABLED)

    def _think_add(self, text):
        self.think_text.config(state=tk.NORMAL)
        self.think_text.insert(tk.END, text)
        self.think_text.see(tk.END)
        self.think_text.config(state=tk.DISABLED)

    def _on_close(self):
        self._stop_flag.set()
        self._stop_audio()
        self.win.destroy()

    # ---- piano roll --------------------------------------------------------------------------
    def _schedule_redraw(self):
        if self._redraw_job is None:
            self._redraw_job = self.canvas.after(60, self._redraw)

    def _layout(self):
        c = self.canvas
        W, H = max(c.winfo_width(), 100), max(c.winfo_height(), 100)
        score = self.score
        pitched = [n["midi"] for t in score["tracks"] if not t["drum"] for n in t["notes"]]
        has_drums = any(t["drum"] and t["notes"] for t in score["tracks"])
        lo, hi = (min(pitched) - 1, max(pitched) + 1) if pitched else (55, 79)
        if hi - lo < 12:
            mid = (hi + lo) // 2
            lo, hi = mid - 6, mid + 6
        drum_h = 22 * len(_DRUM_ROWS) if has_drums else 0
        top_pad, left, right = 8, 34, 10
        roll_h = H - drum_h - top_pad - 8 - (8 if has_drums else 0)
        end = max(16.0, math.ceil(score_end_beat(score) / 4.0) * 4.0)
        return dict(W=W, H=H, lo=lo, hi=hi, drum_h=drum_h, top=top_pad, left=left, right=right,
                    roll_h=max(roll_h, 40), end=end, has_drums=has_drums)

    def _redraw(self):
        self._redraw_job = None
        c = self.canvas
        try:
            c.delete("all")
        except tk.TclError:
            return
        L = self._layout()
        span = L["hi"] - L["lo"] + 1
        row = L["roll_h"] / span
        gw = L["W"] - L["left"] - L["right"]
        xb = gw / L["end"]
        # grid
        for b in range(int(L["end"]) + 1):
            x = L["left"] + b * xb
            c.create_line(x, L["top"], x, L["H"] - 4, fill="#cbd5e1" if b % 4 == 0 else "#eef2f7")
        for m in range(L["lo"], L["hi"] + 1):
            y = L["top"] + (L["hi"] - m) * row
            if m % 12 == 0:
                c.create_line(L["left"], y + row, L["W"] - L["right"], y + row, fill="#cbd5e1")
                c.create_text(4, y + row / 2, text=f"C{m // 12 - 1}", anchor="w", fill="#94a3b8",
                              font=("Segoe UI", 7))
            elif (m % 12) in (1, 3, 6, 8, 10):
                c.create_rectangle(L["left"], y, L["W"] - L["right"], y + row, fill="#f8fafc", outline="")
        c.tag_lower("all")
        drum_top = L["top"] + L["roll_h"] + 8
        if L["has_drums"]:
            for k, name in enumerate(_DRUM_ROWS):
                y = drum_top + k * 22
                c.create_line(L["left"], y, L["W"] - L["right"], y, fill="#e2e8f0")
                c.create_text(4, y + 11, text=name, anchor="w", fill="#94a3b8", font=("Segoe UI", 7))
        for ti, t in enumerate(self.score["tracks"]):
            color = _MUSIC_COLORS[ti % len(_MUSIC_COLORS)]
            for n in t["notes"]:
                x0 = L["left"] + n["s"] * xb
                x1 = max(x0 + 2, L["left"] + (n["s"] + n["d"]) * xb - 1)
                if t["drum"]:
                    k = _DRUM_ROWS.index(n["drum"])
                    y = drum_top + k * 22 + 3
                    c.create_rectangle(x0, y, max(x0 + 4, x1 - (x1 - x0) * 0.5), y + 16, fill=color, outline="")
                else:
                    y = L["top"] + (L["hi"] - n["midi"]) * row
                    c.create_rectangle(x0, y + 1, x1, y + max(row - 1, 3), fill=color, outline="")
        # legend
        lx = L["W"] - L["right"] - 4
        for ti, t in reversed(list(enumerate(self.score["tracks"]))):
            color = _MUSIC_COLORS[ti % len(_MUSIC_COLORS)]
            label = f"{t['name']} ({t['wave']})"
            c.create_text(lx, 10 + 0, text=label, anchor="ne", fill=color, font=("Segoe UI", 8, "bold"))
            lx -= 8 * len(label) + 14
        if self._playing:
            self._draw_playhead()

    # ---- playback ----------------------------------------------------------------------------
    def _draw_playhead(self):
        c = self.canvas
        c.delete("playhead")
        if not self._playing or self._play_len <= 0:
            return
        L = self._layout()
        frac = min(1.0, (time.monotonic() - self._play_start) * (self.score["tempo"] / 60.0) / L["end"])
        x = L["left"] + frac * (L["W"] - L["left"] - L["right"])
        c.create_line(x, 0, x, L["H"], fill="#111827", width=2, tags="playhead")

    def _tick(self):
        self._tick_job = None
        if not self._playing:
            return
        if time.monotonic() - self._play_start >= self._play_len:
            self._stop_audio()
            return
        self._draw_playhead()
        self._tick_job = self.win.after(30, self._tick)

    def _toggle_play(self):
        if self._playing:
            self._stop_audio()
            return
        if self.busy or not score_note_count(self.score):
            return
        if self._wav is None:
            self._set_status("Rendering audio\u2026")
            self.play_btn.config(state=tk.DISABLED)
            score = self.score
            threading.Thread(target=self._render_worker, args=(score,), daemon=True).start()
        else:
            self._play_now()

    def _render_worker(self, score):
        try:
            wav = render_wav(score)
        except Exception as exc:                     # noqa: BLE001
            self._ui(self._render_done, None, str(exc), score)
            return
        self._ui(self._render_done, wav, None, score)

    def _render_done(self, wav, err, score):
        try:
            self.play_btn.config(state=tk.NORMAL)
        except tk.TclError:
            return
        if err:
            self._set_status(f"Couldn't render the audio \u2014 {err}", error=True)
            return
        if score is self.score:
            self._wav = wav
            self._play_now()

    def _play_now(self):
        wav = self._wav
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(wav, winsound.SND_MEMORY | winsound.SND_ASYNC)
            else:
                player = next((shutil.which(p) for p in ("afplay", "paplay", "aplay") if shutil.which(p)), None)
                if not player:
                    self._set_status("No audio player found. Use Save WAV\u2026 instead.", error=True)
                    return
                fd, path = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(fd, "wb") as f:
                    f.write(wav)
                self._proc = subprocess.Popen([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:                     # noqa: BLE001
            self._set_status(f"Couldn't play audio \u2014 {exc}", error=True)
            return
        self._playing = True
        self._play_start = time.monotonic()
        self._play_len = score_end_beat(self.score) * 60.0 / self.score["tempo"] + 0.3
        self.play_btn.config(text="\u25A0 Stop")
        self._set_status("Playing\u2026")
        self._tick()

    def _stop_audio(self):
        was = self._playing
        self._playing = False
        if self._tick_job:
            try:
                self.win.after_cancel(self._tick_job)
            except Exception:                        # noqa: BLE001
                pass
            self._tick_job = None
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            elif self._proc is not None:
                self._proc.terminate()
        except Exception:                            # noqa: BLE001
            pass
        self._proc = None
        try:
            self.canvas.delete("playhead")
            self.play_btn.config(text="\u25B6 Play")
            if was:
                self._set_status("Done playing.")
        except tk.TclError:
            pass

    # ---- saving ------------------------------------------------------------------------------
    def _save_bytes(self, data, default_filename, title, kind, ext):
        if not data:
            self._set_status("Nothing to save yet.")
            return
        app = self.app
        if getattr(app, "save_folder", None):
            name = app._prompt_filename(default_filename, title)
            if not name:
                return
            path = os.path.join(app.save_folder, name)
        else:
            path = filedialog.asksaveasfilename(title=title, initialfile=default_filename, defaultextension=ext,
                                                filetypes=[(kind, "*" + ext), ("All files", "*.*")])
            if not path:
                return
        try:
            with open(path, "wb") as f:
                f.write(data)
            self._set_status(f"Saved {os.path.basename(path)}.")
        except Exception as exc:                     # noqa: BLE001
            messagebox.showerror(title, f"Couldn't save file:\n{exc}")

    def _save_midi(self):
        if score_note_count(self.score):
            self._save_bytes(render_midi(self.score), "music.mid", "Save MIDI", "MIDI files", ".mid")
        else:
            self._set_status("Nothing to save yet.")

    def _save_wav(self):
        if not score_note_count(self.score):
            self._set_status("Nothing to save yet.")
            return
        if self._wav is None:
            self._set_status("Rendering audio\u2026")
            self.win.update_idletasks()
            self._wav = render_wav(self.score)
        self._save_bytes(self._wav, "music.wav", "Save WAV", "WAV files", ".wav")

    # ---- composing ---------------------------------------------------------------------------
    def _new(self):
        if self.busy:
            return
        self._stop_audio()
        self.messages, self.raw, self._wav = [], "", None
        self.score = {"tempo": 110.0, "tracks": []}
        self._redraw()
        self._set_status("Describe the music you want.")

    def _stop(self):
        self._stop_flag.set()
        self._set_status("Stopping\u2026")

    def _start(self):
        if self.busy:
            return
        prompt = self.entry.get().strip()
        if not prompt:
            return
        try:
            h = self.app.http_bridge._handle()
        except BridgeError as exc:
            self._set_status(str(exc), error=True)
            return
        self._stop_audio()
        editing = bool(self.messages)
        user_msg = (f"Change the music: {prompt}\nReturn the complete updated JSON." if editing
                    else f"Compose: {prompt}")
        messages = [{"role": "system", "content": MUSIC_SYSTEM_PROMPT}] + self.messages[-4:] + \
                   [{"role": "user", "content": user_msg}]
        self.busy = True
        self._stop_flag.clear()
        self.raw = ""
        self._wav = None
        self._thinking = 0
        self._think_clear()
        self.score = {"tempo": 110.0, "tracks": []}
        self._redraw()
        self.go_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("Waiting for the model\u2026" if h.gen_lock.locked() else "Starting\u2026")
        threading.Thread(target=self._worker, args=(h, messages, user_msg), daemon=True).start()

    def _worker(self, h, messages, user_msg):
        err = None
        try:
            with h.gen_lock:
                if h.cancel.is_set() or h.llm is None:
                    raise BridgeError(409, "The model was unloaded.")
                try:
                    tpl = h.llm.metadata.get("tokenizer.chat_template", "") or ""
                except Exception:                    # noqa: BLE001
                    tpl = ""
                open_think = bool(re.search(r"add_generation_prompt.*?'<think>\\n'", tpl, re.DOTALL))
                stream = h.llm.create_chat_completion(messages=messages, temperature=0.8,
                                                      max_tokens=4096, stream=True)
                harmony = HarmonyStreamFilter()
                parser = {"mode": "think" if open_think else "pre", "buffer": ""}

                def route(kind, text):
                    if kind == "reasoning":
                        self._ui(self._on_piece, "reasoning", text)
                    else:
                        for k2, t2 in feed_think_state(parser, text):
                            self._ui(self._on_piece, k2, t2)

                for chunk in stream:
                    if self._stop_flag.is_set() or h.cancel.is_set():
                        stream.close()
                        if h.cancel.is_set():
                            raise BridgeError(409, "Stopped \u2014 the model was unloaded.")
                        break
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    r_piece = delta.get("reasoning_content") or delta.get("reasoning")
                    if r_piece:
                        self._ui(self._on_piece, "reasoning", r_piece)
                    piece = delta.get("content")
                    if piece:
                        for kind, text in harmony.feed(piece):
                            route(kind, text)
                for kind, text in harmony.flush():
                    route(kind, text)
                if parser["buffer"]:
                    self._ui(self._on_piece, "reasoning" if parser["mode"] == "think" else "answer",
                             parser["buffer"])
        except Exception as exc:                     # noqa: BLE001
            err = str(exc) or type(exc).__name__
        self._ui(self._on_done, err, user_msg)

    def _on_piece(self, kind, text):
        if kind == "reasoning":
            self._thinking += len(text)
            self._think_add(text)
            if not score_note_count(self.score):
                self._set_status(f"Thinking\u2026 ({self._thinking} chars)")
            return
        self.raw += text
        self.score = parse_music(self.raw)
        n = score_note_count(self.score)
        if n:
            self._set_status(f"Composing\u2026 {n} notes, {len(self.score['tracks'])} track(s)")
            self._schedule_redraw()

    def _on_done(self, err, user_msg):
        self.busy = False
        try:
            self.go_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.play_btn.config(state=tk.NORMAL)
        except tk.TclError:
            return
        self.score = parse_music(self.raw)
        self._redraw()
        n = score_note_count(self.score)
        stopped = self._stop_flag.is_set()
        if err:
            self._set_status(f"Failed \u2014 {err}", error=True)
        elif not n:
            self._set_status("The model didn't return any notes. Try again, or a bigger model.", error=True)
        else:
            beats = score_end_beat(self.score)
            secs = beats * 60.0 / self.score["tempo"]
            self._set_status(("Stopped. " if stopped else "Done. ") +
                             f"{n} notes, {secs:.0f}s at {self.score['tempo']:.0f} bpm \u2014 press Play, or type a change.")
        if n:
            self.messages += [{"role": "user", "content": user_msg},
                              {"role": "assistant", "content": score_to_json(self.score)}]
            self.entry.delete(0, tk.END)




# ==============================================================================================
#  MAIN WINDOW - text input, text output, Send button, a token counter, and buttons to open
#  the Server, Models, and Prompts windows. Send calls the Local backend for real.
# ==============================================================================================
# ==========================================================================================
#  IMAGE GENERATION (stable-diffusion.cpp, SD 1.5)
#  Runs the prebuilt sd-cli.exe / sd.exe as a subprocess - nothing is loaded into this
#  process, so it can't clash with llama.cpp's ggml, and closing/killing the process frees
#  the VRAM at once. Opened from the title-bar right-click menu -> "Images (SD 1.5)".
#  Paths/settings are session-only like the rest of the app.
# ==========================================================================================

SD_VRAM_NEEDED_MB = 4000          # rough SD 1.5 @ 512x512 incl. compute buffers
SD_SAMPLERS = ("euler_a", "euler", "dpm++2m", "heun", "lcm")
SD_SIZES = ("384", "448", "512", "576", "640", "768")


class ImageEngine:
    """Thin wrapper around the stable-diffusion.cpp command line."""

    def __init__(self, app):
        self.app = app
        self.proc = None
        self.cancel = threading.Event()

    @property
    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        self.cancel.set()
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def generate(self, exe, model, prompt, negative, width, height, steps, cfg, seed,
                 sampler, out_path, on_progress=None):
        """Blocking - call from a worker thread. Returns out_path, or raises RuntimeError."""
        cmd = [exe, "-m", model, "-p", prompt, "-W", str(width), "-H", str(height),
               "--steps", str(steps), "--cfg-scale", str(cfg), "-s", str(seed),
               "--sampling-method", sampler, "-o", out_path]
        if negative:
            cmd += ["-n", negative]
        self.cancel.clear()
        if os.path.exists(out_path):
            os.remove(out_path)
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                errors="replace", cwd=os.path.dirname(os.path.abspath(exe)),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            raise RuntimeError(f"Couldn't start {os.path.basename(exe)}: {exc}")
        tail = []
        for line in self.proc.stdout:          # text mode: \r progress updates arrive as lines
            line = line.strip()
            if not line:
                continue
            tail.append(line)
            del tail[:-25]
            if on_progress:
                on_progress(line)
        code = self.proc.wait()
        if self.cancel.is_set():
            raise RuntimeError("Stopped.")
        if code != 0 or not os.path.exists(out_path):
            hint = ""
            joined = "\n".join(tail).lower()
            if code in (3221225781, -1073741515) or "dll" in joined:
                hint = ("\n(A DLL is missing - install the Microsoft Visual C++ Redistributable, "
                        "and put the CUDA runtime DLLs next to the exe if the release has a "
                        "separate cudart zip.)")
            raise RuntimeError(f"{os.path.basename(exe)} exited with code {code}.\n"
                               + "\n".join(tail[-8:]) + hint)
        return out_path


class ImagesWindow:
    def __init__(self, parent, theme, app):
        self.app = app
        self.t = t = theme
        self.engine = app.image_engine
        self.photo = None
        self.last_path = None
        font = (t["family"], t["size"])
        small = (t["family"], t["small"])

        self.win = win = tk.Toplevel(parent)
        win.title("Images (SD 1.5)")
        win.configure(bg=t["bg"])
        win.geometry("620x860")
        win.protocol("WM_DELETE_WINDOW", self._on_close)

        def label(text):
            return tk.Label(win, text=text, bg=t["bg"], fg=t["muted"], font=small, anchor=tk.W)

        def entry(var=None, width=None):
            return tk.Entry(win if False else win, textvariable=var, font=font, bg=t["input"],
                            fg=t["text"], relief=tk.SOLID, bd=1, width=width or 20)

        def button(parent_, text, cmd, bg=None):
            return tk.Button(parent_, text=text, command=cmd, bg=bg or t["neutral"],
                             fg=t["text"] if bg is None else t["on_accent"],
                             activebackground=t["accent"], activeforeground=t["on_accent"],
                             relief=tk.FLAT, bd=0, padx=10, pady=4, cursor="hand2", font=small)

        # --- paths -------------------------------------------------------------------------
        self.exe_var = tk.StringVar(value=app.sd_exe)
        self.model_var = tk.StringVar(value=app.sd_model)
        for text, var, pick in (("stable-diffusion.cpp executable (sd-cli.exe / sd.exe)",
                                 self.exe_var, self._pick_exe),
                                ("SD 1.5 checkpoint (.safetensors / .ckpt / .gguf)",
                                 self.model_var, self._pick_model)):
            label(text).pack(fill=tk.X, padx=12, pady=(8, 0))
            row = tk.Frame(win, bg=t["bg"])
            row.pack(fill=tk.X, padx=12)
            tk.Entry(row, textvariable=var, font=font, bg=t["input"], fg=t["text"],
                     relief=tk.SOLID, bd=1).pack(side=tk.LEFT, fill=tk.X, expand=True)
            button(row, "Browse...", pick).pack(side=tk.LEFT, padx=(6, 0))

        # --- prompts -----------------------------------------------------------------------
        label("Prompt").pack(fill=tk.X, padx=12, pady=(10, 0))
        self.prompt_box = tk.Text(win, height=4, wrap=tk.WORD, font=font, bg=t["input"],
                                  fg=t["text"], relief=tk.SOLID, bd=1)
        self.prompt_box.pack(fill=tk.X, padx=12)
        label("Negative prompt").pack(fill=tk.X, padx=12, pady=(8, 0))
        self.neg_var = tk.StringVar(value="blurry, low quality, deformed, extra limbs, watermark")
        tk.Entry(win, textvariable=self.neg_var, font=font, bg=t["input"], fg=t["text"],
                 relief=tk.SOLID, bd=1).pack(fill=tk.X, padx=12)

        # --- parameters --------------------------------------------------------------------
        params = tk.Frame(win, bg=t["bg"])
        params.pack(fill=tk.X, padx=12, pady=(10, 0))
        self.w_var = tk.StringVar(value="512")
        self.h_var = tk.StringVar(value="512")
        self.steps_var = tk.StringVar(value="20")
        self.cfg_var = tk.StringVar(value="7.0")
        self.seed_var = tk.StringVar(value="-1")
        self.sampler_var = tk.StringVar(value=SD_SAMPLERS[0])
        col = 0
        for text, widget in (
                ("Width", ttk.Combobox(params, textvariable=self.w_var, values=SD_SIZES, width=5)),
                ("Height", ttk.Combobox(params, textvariable=self.h_var, values=SD_SIZES, width=5)),
                ("Steps", tk.Entry(params, textvariable=self.steps_var, width=5, font=font,
                                   bg=t["input"], fg=t["text"], relief=tk.SOLID, bd=1)),
                ("CFG", tk.Entry(params, textvariable=self.cfg_var, width=5, font=font,
                                 bg=t["input"], fg=t["text"], relief=tk.SOLID, bd=1)),
                ("Seed (-1 = random)", tk.Entry(params, textvariable=self.seed_var, width=10,
                                                font=font, bg=t["input"], fg=t["text"],
                                                relief=tk.SOLID, bd=1)),
                ("Sampler", ttk.Combobox(params, textvariable=self.sampler_var,
                                         values=SD_SAMPLERS, width=8, state="readonly"))):
            tk.Label(params, text=text, bg=t["bg"], fg=t["muted"], font=small).grid(
                row=0, column=col, sticky=tk.W, padx=(0, 8))
            widget.grid(row=1, column=col, sticky=tk.W, padx=(0, 8))
            col += 1

        # --- actions -----------------------------------------------------------------------
        bar = tk.Frame(win, bg=t["bg"])
        bar.pack(fill=tk.X, padx=12, pady=12)
        self.gen_btn = button(bar, "Generate", self._generate, bg=t["accent"])
        self.gen_btn.pack(side=tk.LEFT)
        self.stop_btn = button(bar, "Stop", self.engine.stop)
        self.stop_btn.pack(side=tk.LEFT, padx=6)
        self.stop_btn.config(state=tk.DISABLED)
        button(bar, "Save As...", self._save_as).pack(side=tk.LEFT)
        button(bar, "Open folder", self._open_folder).pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(win, text="Pick the exe and a checkpoint, then write a prompt.",
                               bg=t["bg"], fg=t["muted"], font=small, anchor=tk.W,
                               justify=tk.LEFT, wraplength=590)
        self.status.pack(fill=tk.X, padx=12)
        self.preview = tk.Label(win, bg=t["card"], relief=tk.SOLID, bd=1)
        self.preview.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    # ---- helpers -----------------------------------------------------------------------
    def _set_status(self, text):
        try:
            self.status.config(text=text)
        except tk.TclError:
            pass

    def _pick_exe(self):
        path = filedialog.askopenfilename(parent=self.win, title="stable-diffusion.cpp executable",
                                          filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self.exe_var.set(path)
            self.app.sd_exe = path

    def _pick_model(self):
        path = filedialog.askopenfilename(
            parent=self.win, title="SD 1.5 checkpoint",
            filetypes=[("Model", "*.safetensors *.ckpt *.gguf"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)
            self.app.sd_model = path

    def _out_dir(self):
        base = self.app.save_folder or tempfile.gettempdir()
        path = os.path.join(base, "GGUFllama_images")
        os.makedirs(path, exist_ok=True)
        return path

    def _vram_ok(self):
        """SD runs in its own process, so the LLM's VRAM is invisible to it. Warn if a Local
        model is loaded and the estimate says SD 1.5 won't fit next to it."""
        lb = self.app.local_backend
        if lb is None or not lb.slots:
            return True
        free = lb.TOTAL_VRAM_MB - LOCAL_VRAM_SAFETY_MARGIN_MB - lb._current_vram_mb()
        if free >= SD_VRAM_NEEDED_MB:
            return True
        return messagebox.askyesno(
            "Low VRAM",
            f"A Local model is loaded and only ~{max(free, 0)} MB of VRAM looks free; SD 1.5 "
            f"wants ~{SD_VRAM_NEEDED_MB} MB.\n\nUnload the model in the Models window first, "
            "or continue anyway (it may fail or be very slow).\n\nContinue anyway?",
            parent=self.win)

    # ---- generate ----------------------------------------------------------------------
    def _generate(self):
        if self.engine.busy:
            return
        exe, model = self.exe_var.get().strip(), self.model_var.get().strip()
        prompt = self.prompt_box.get("1.0", tk.END).strip()
        if not os.path.isfile(exe):
            self._set_status("Pick a valid stable-diffusion.cpp executable first.")
            return
        if not os.path.isfile(model):
            self._set_status("Pick a valid SD 1.5 checkpoint first.")
            return
        if not prompt:
            self._set_status("Write a prompt first.")
            return
        try:
            w, h = int(self.w_var.get()), int(self.h_var.get())
            steps, cfg = int(self.steps_var.get()), float(self.cfg_var.get())
            seed = int(self.seed_var.get())
        except ValueError:
            self._set_status("Width, height, steps, CFG and seed must be numbers.")
            return
        if w % 64 or h % 64:
            w, h = (w // 64) * 64 or 64, (h // 64) * 64 or 64   # SD wants multiples of 64
            self.w_var.set(str(w))
            self.h_var.set(str(h))
        if not self._vram_ok():
            return
        self.app.sd_exe, self.app.sd_model = exe, model
        out = os.path.join(self._out_dir(), f"sd_{time.strftime('%Y%m%d_%H%M%S')}.png")
        self.gen_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("Loading model and generating...")
        neg, sampler = self.neg_var.get().strip(), self.sampler_var.get()
        started = time.time()

        def progress(line):
            self.win.after(0, self._set_status, line[-200:])

        def work():
            try:
                path = self.engine.generate(exe, model, prompt, neg, w, h, steps, cfg, seed,
                                            sampler, out, on_progress=progress)
                self.win.after(0, self._done, path, time.time() - started)
            except Exception as exc:
                self.win.after(0, self._failed, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _done(self, path, secs):
        self.gen_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.last_path = path
        try:
            img = tk.PhotoImage(file=path)
            factor = max(1, math.ceil(max(img.width(), img.height()) / 512))
            self.photo = img.subsample(factor) if factor > 1 else img
            self.preview.config(image=self.photo)
            self._set_status(f"Done in {secs:.1f}s - saved to {path}")
        except tk.TclError as exc:
            self._set_status(f"Saved to {path} (couldn't preview: {exc})")

    def _failed(self, msg):
        self.gen_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._set_status(msg)

    def _save_as(self):
        if not self.last_path or not os.path.exists(self.last_path):
            self._set_status("Nothing to save yet.")
            return
        dest = filedialog.asksaveasfilename(parent=self.win, defaultextension=".png",
                                            initialfile=os.path.basename(self.last_path),
                                            filetypes=[("PNG image", "*.png")])
        if dest:
            shutil.copyfile(self.last_path, dest)
            self._set_status(f"Saved to {dest}")

    def _open_folder(self):
        try:
            os.startfile(self._out_dir())
        except (AttributeError, OSError):
            webbrowser.open(self._out_dir())

    def _on_close(self):
        self.engine.stop()
        self.win.destroy()


class SimpleChat:
    def __init__(self, root):
        self.root = root
        self.t = dict(PALETTE, family=FONT_UI, size=10, small=9, title=11)
        self._prompt_win = None
        self._models_win = None
        self._notes_win = None
        self.notes_text = ""       # kept in memory only - resets on next launch
        self.save_folder = None   # this session's save folder - asked at boot, resets on next launch
        self.last_response_text = ""   # most recently completed model reply (chat or discussion),
                                        # for the chat-log right-click "Export to..." actions

        self.server_connected = False
        self.server_model = None
        self.server_instance_id = None
        self.server_catalog = []   # last backend catalog, shared with Models window

        # Single backend: llama-cpp-python running models directly in-process on the GPU.
        # None if llama-cpp-python isn't installed (Connect then refuses with a message).
        self.local_models_root = DEFAULT_LOCAL_MODELS_ROOT
        self.local_backend = LocalLlamaBackend(self) if LLAMA_CPP_AVAILABLE else None
        self.backend = self.local_backend
        self.conversation = []     # [{"role": "user"/"assistant", "content": str}, ...]
        self.persona_enabled = False       # Terminator persona + lore grounding (see module above)
        self.lore_path = find_default_lore_file()
        self.lore_sections = []
        self.speaker = Speaker()
        self.speak_enabled = False
        self.total_tokens = 0
        self.awaiting_reply = False

        # Discussion mode: two models the Models window has tagged "Discussion A" / "B", kept
        # loaded simultaneously and protected from the normal single-slot unload bookkeeping.
        # slot -> {"key", "display_name", "instance_id"} or None if unassigned.
        self.discussion_models = {"A": None, "B": None}
        self.reasoning_enabled = True   # Local backend: let the model think (Qwen on/off; GPT-OSS low effort when off)
        self.reasoning_level = "Medium" # Local backend: Low / Medium / High (GPT-OSS; Qwen ignores it)
        self.show_thinking = True       # show the muted "Thinking" block in the chat log
        self.ram_spillover = False      # Local backend: allow layers to spill from VRAM into system RAM
        self.gpu_layers_override = None # Local backend: manual n_gpu_layers (None = auto)
        self.http_bridge = HttpBridge(self)   # optional 127.0.0.1 API for other local programs
        self.image_engine = ImageEngine(self)   # stable-diffusion.cpp subprocess wrapper
        self.sd_exe = ""                        # path to sd-cli.exe / sd.exe (session-only)
        self.sd_model = ""                      # path to the SD 1.5 checkpoint (session-only)
        self.block_editor = BlockEditorServer(self)   # built-in Block Editor (opened from the title-bar menu)
        self.auto_max_context = False   # mirrors the Models window's "Auto-max context" checkbox

        self.constraints_enabled = False        # mirrors the "Enabled" checkbox on the Constraints card
        self.constraint_engine = None            # lazy-loaded ConstraintEngine instance
        self._constraint_editor_win = None
        self._constraint_test_win = None
        self.constraints_collapsed = False       # whole-card collapse state, toggled above the box
        self.simple_mode = False                 # True when collapsed to just log + input box

        self.mode = "chat"                    # "chat" or "discussion"
        self.discussion_running = False
        self.discussion_stop_flag = False
        self.discussion_conv = {"A": [], "B": []}   # each model's own view of the exchange
        self.discussion_interject_queue = []          # your messages, drained between turns
        self.discussion_turn = None                    # which slot speaks next: "A" or "B"
        self.discussion_transcript = []   # ordered [{"speaker": str, "text": str}, ...] for the
                                           # whole run - display order

        # Live streaming state for whichever single request is currently in flight (chat or
        # discussion never overlap - only one call is ever active at a time). Reset by
        # _begin_stream_turn() right before each request goes out.
        self._stream_ctx = None
        self._stream_parse_state = {"mode": "pre", "buffer": ""}
        self._stream_reasoning_started = False
        self._stream_answer_started = False
        self._stream_full_reasoning = ""
        self._stream_full_answer = ""
        self._stream_usage = None

        self._build_ui()
        self.root.after(0, self._prompt_save_folder_at_boot)

    def _build_ui(self):
        t, win = self.t, self.root
        font = (t["family"], t["size"])
        font_chat = (t["family"], 12)
        win.title(APP_TITLE)
        win.geometry("760x640")
        win.minsize(560, 420)
        win.configure(bg=t["bg"])

        self.head = head = tk.Frame(win, bg=t["bg"])
        head.pack(fill=tk.X, padx=14, pady=(12, 0))
        title_lbl = tk.Label(head, text=APP_TITLE, bg=t["bg"], fg=t["text"],
                             font=(t["family"], t["title"], "bold"))
        title_lbl.pack(side=tk.LEFT)

        self.head_menu = tk.Menu(head, tearoff=0)
        head.bind("<Button-3>", self._on_head_right_click)
        title_lbl.bind("<Button-3>", self._on_head_right_click)

        self.head2 = head2 = tk.Frame(win, bg=t["bg"])
        head2.pack(fill=tk.X, padx=14, pady=(6, 0))
        tk.Button(head2, text="\U0001F5C2 Models", command=self.open_models_viewer,
                  bg=t["go"], fg=t["on_go"], activebackground=t["go"], activeforeground=t["on_go"],
                  relief=tk.FLAT, bd=0,
                  padx=10, pady=4, cursor="hand2", font=font).pack(side=tk.LEFT)
        tk.Button(head2, text="\U0001F4D6 Prompts", command=self.open_prompt_editor,
                  bg=t["go"], fg=t["on_go"], activebackground=t["go"], activeforeground=t["on_go"],
                  relief=tk.FLAT, bd=0,
                  padx=10, pady=4, cursor="hand2", font=font).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(head2, text="\U0001F4DD Notes", command=self.open_notes,
                  bg=t["go"], fg=t["on_go"], activebackground=t["go"], activeforeground=t["on_go"],
                  relief=tk.FLAT, bd=0,
                  padx=10, pady=4, cursor="hand2", font=font).pack(side=tk.LEFT, padx=(8, 0))
        self.new_chat_btn = tk.Button(head2, text="\U0001F195 New Chat", command=self.new_chat,
                                      bg=t["go"], fg=t["on_go"], activebackground=t["go"],
                                      activeforeground=t["on_go"],
                                      relief=tk.FLAT, bd=0, padx=10, pady=4, cursor="hand2", font=font)
        self.new_chat_btn.pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(head2, text="\U0001F9E9 Simple Mode", command=self.toggle_simple_mode,
                  bg=t["go"], fg=t["on_go"], activebackground=t["go"], activeforeground=t["on_go"],
                  relief=tk.FLAT, bd=0,
                  padx=10, pady=4, cursor="hand2", font=font).pack(side=tk.LEFT, padx=(8, 0))

        self.constraints_toggle_row = constraints_toggle_row = tk.Frame(win, bg=t["bg"])
        constraints_toggle_row.pack(fill=tk.X, padx=14, pady=(8, 0))
        self.constraints_toggle_btn = tk.Button(constraints_toggle_row, text="\u25BE Hide Constraints Engine",
                                                command=self._toggle_constraints_card, bg=t["bg"], fg=t["muted"],
                                                activebackground=t["bg"], relief=tk.FLAT, bd=0, padx=0, pady=2,
                                                cursor="hand2", font=(t["family"], t["small"]))
        self.constraints_toggle_btn.pack(side=tk.RIGHT)

        self.constraints_card = constraints_card = tk.Frame(win, bg=t["card"], relief=tk.SOLID, bd=1)
        constraints_card.pack(fill=tk.X, padx=14, pady=(4, 0))

        server_row = tk.Frame(constraints_card, bg=t["card"])
        c_title_row = tk.Frame(constraints_card, bg=t["card"])
        c_title_row.pack(fill=tk.X, padx=10, pady=(10, 4))
        tk.Label(c_title_row, text="Constraints Engine", bg=t["card"], fg=t["text"],
                 font=(t["family"], t["size"], "bold")).pack(side=tk.LEFT)
        self.constraints_status_lbl = tk.Label(c_title_row, text="Engine Disabled", bg=t["card"],
                                               fg=t["muted"], font=(t["family"], t["small"]))
        self.constraints_status_lbl.pack(side=tk.RIGHT)

        c_row = tk.Frame(constraints_card, bg=t["card"])
        c_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.constraints_enabled_var = tk.BooleanVar(value=self.constraints_enabled)
        tk.Checkbutton(c_row, text="Enabled", variable=self.constraints_enabled_var,
                      command=self._on_constraints_toggle, bg=t["card"], fg=t["text"],
                      activebackground=t["card"], selectcolor=t["bg"], font=(t["family"], t["small"]),
                      cursor="hand2", bd=0, highlightthickness=0).pack(side=tk.LEFT, padx=(0, 12))

        for label, cmd in (("Edit Engine", self.open_constraints_editor),
                           ("Test Engine", self.open_constraints_test)):
            tk.Button(c_row, text=label, command=cmd, bg=t["warn"], fg=t["on_warn"],
                     activebackground=t["warn"], activeforeground=t["on_warn"], relief=tk.FLAT,
                     bd=0, padx=10, pady=4, cursor="hand2", font=(t["family"], t["small"])
                     ).pack(side=tk.LEFT, padx=(0, 6))

        self._update_constraints_status()

        self.server_frame = tk.Frame(server_row, bg=t["neutral"])
        self.server_frame.pack(side=tk.LEFT)
        self.server_dot = tk.Canvas(self.server_frame, width=10, height=10, bg=t["neutral"],
                                    highlightthickness=0, bd=0)
        self._server_dot_id = self.server_dot.create_oval(1, 1, 9, 9, fill="#ef4444", outline="")
        self.server_dot.pack(side=tk.LEFT, padx=(10, 6), pady=4)
        self.server_label = tk.Label(self.server_frame, text="Disconnected", bg=t["neutral"], fg=t["text"], font=font)
        self.server_label.pack(side=tk.LEFT, padx=(0, 8), pady=4)
        self.connect_btn = tk.Button(self.server_frame, text="Authenticate", command=self.connect_server,
                                     bg=t["accent"], fg=t["on_accent"], activebackground=t["accent"],
                                     activeforeground=t["on_accent"], relief=tk.FLAT, bd=0, padx=8,
                                     pady=2, cursor="hand2", font=(t["family"], t["small"]))
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 10), pady=4)
        server_row.pack(fill=tk.X, padx=10, pady=(4, 10))

        # Auto-max context, RAM spillover and GPU layers now live in the model popup menu under
        # the input box (see _show_reasoning_menu), together with the other model settings.
        self.auto_max_ctx_var = tk.BooleanVar(value=self.auto_max_context)
        self.ram_spill_var = tk.BooleanVar(value=self.ram_spillover)
        self.gpu_layers_slider_var = tk.IntVar(value=0)
        self.gpu_layers_label = None        # only exists while the GPU layers dialog is open
        self._gpu_layers_win = None

        # Reasoning / Effort / Show thinking live in a popup menu under the input box
        # (see the reasoning button next to the hint line below).
        self.reasoning_var = tk.BooleanVar(value=self.reasoning_enabled)
        self.reasoning_level_var = tk.StringVar(value=self.reasoning_level)
        self.show_thinking_var = tk.BooleanVar(value=self.show_thinking)

        persona_row = tk.Frame(constraints_card, bg=t["card"])
        persona_row.pack(fill=tk.X, padx=10, pady=(0, 2))
        self.persona_var = tk.BooleanVar(value=self.persona_enabled)
        tk.Checkbutton(persona_row, text="Terminator persona", variable=self.persona_var,
                      command=self._on_persona_toggled, bg=t["card"], fg=t["muted"],
                      activebackground=t["card"], selectcolor=t["bg"], font=(t["family"], t["small"]),
                      cursor="hand2", bd=0, highlightthickness=0).pack(side=tk.LEFT)
        self.speak_var = tk.BooleanVar(value=self.speak_enabled)
        tk.Checkbutton(persona_row, text="Speak", variable=self.speak_var,
                      command=self._on_speak_toggled, bg=t["card"], fg=t["muted"],
                      activebackground=t["card"], selectcolor=t["bg"], font=(t["family"], t["small"]),
                      cursor="hand2", bd=0, highlightthickness=0).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(persona_row, text="Lore file...", command=self._choose_lore_file,
                 bg=t["neutral"], fg=t["text"], activebackground=t["neutral"], relief=tk.FLAT,
                 bd=0, padx=8, pady=2, cursor="hand2",
                 font=(t["family"], t["small"])).pack(side=tk.LEFT, padx=(10, 0))
        self.lore_status_lbl = tk.Label(persona_row, text="", bg=t["card"], fg=t["muted"],
                 font=(t["family"], t["small"]))
        self.lore_status_lbl.pack(side=tk.LEFT, padx=(8, 0))
        self._refresh_lore_status()

        auto_unload_row = tk.Frame(constraints_card, bg=t["card"])
        auto_unload_row.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.auto_unload_var = tk.BooleanVar(value=True)
        self.auto_unload_chk = tk.Checkbutton(auto_unload_row, text="Unload model after each reply",
                      variable=self.auto_unload_var, bg=t["card"], fg=t["muted"], activebackground=t["card"],
                      selectcolor=t["bg"], font=(t["family"], t["small"]), cursor="hand2", bd=0,
                      highlightthickness=0)
        self.auto_unload_chk.pack(side=tk.LEFT)

        self.mode_row = mode_row = tk.Frame(win, bg=t["bg"])
        mode_row.pack(fill=tk.X, padx=14, pady=(6, 0))
        self.chat_mode_btn = tk.Button(mode_row, text="\U0001F4AC Chat", command=lambda: self.set_mode("chat"),
                                       relief=tk.FLAT, bd=0, padx=10, pady=3, cursor="hand2",
                                       font=(t["family"], t["small"]))
        self.chat_mode_btn.pack(side=tk.LEFT)
        self.discussion_mode_btn = tk.Button(mode_row, text="\U0001F5E3 Discussion",
                                             command=lambda: self.set_mode("discussion"),
                                             relief=tk.FLAT, bd=0, padx=10, pady=3, cursor="hand2",
                                             font=(t["family"], t["small"]))
        self.discussion_mode_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.discussion_status_lbl = tk.Label(mode_row, text="", bg=t["bg"],
                                              font=(t["family"], t["small"]), anchor=tk.E, justify=tk.RIGHT)
        # packed/unpacked by _apply_mode_visuals - only shown while in Discussion mode

        self.sub_head = sub_head = tk.Frame(win, bg=t["bg"])
        sub_head.pack(fill=tk.X, padx=14, pady=(6, 6))
        self.token_lbl = tk.Label(sub_head, text="Tokens: 0", bg=t["bg"], fg=t["muted"],
                                  font=(t["family"], t["small"]), anchor=tk.W)
        self.token_lbl.pack(side=tk.LEFT)

        # Bar shown in place of everything above while Simple Mode is on - just a way back out.
        self.simple_mode_bar = tk.Frame(win, bg=t["bg"])
        tk.Button(self.simple_mode_bar, text="\u25C0 Exit Simple Mode", command=self.toggle_simple_mode,
                 bg=t["bg"], fg=t["muted"], activebackground=t["bg"], relief=tk.FLAT, bd=0, padx=0,
                 pady=2, cursor="hand2", font=(t["family"], t["small"])).pack(side=tk.LEFT)
        # not packed here - only shown while self.simple_mode is True (toggle_simple_mode)

        self.paned = paned = tk.PanedWindow(win, orient=tk.VERTICAL, sashwidth=6, sashrelief=tk.FLAT,
                               bg=t["bg"], bd=0, opaqueresize=True)
        paned.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))

        log_wrap = tk.Frame(paned, bg=t["bg"])
        paned.add(log_wrap, minsize=100, stretch="always")
        self.chat_text = tk.Text(log_wrap, wrap="word", bg=t["card"], fg=t["text"], relief=tk.SOLID,
                                 bd=1, font=font_chat, padx=8, pady=6, state=tk.DISABLED)
        sb = ttk.Scrollbar(log_wrap, orient=tk.VERTICAL, command=self.chat_text.yview)
        self.chat_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_text.tag_configure("who_user", foreground=t["accent"], font=(t["family"], t["small"], "bold"))
        self.chat_text.tag_configure("who_assistant", foreground=t["text"], font=(t["family"], t["small"], "bold"))
        self.chat_text.tag_configure("system_msg", foreground="#d64545", font=(t["family"], t["small"]))
        self.chat_text.tag_configure("who_discussion_a", foreground=t["discussion_a"],
                                     font=(t["family"], t["small"], "bold"))
        self.chat_text.tag_configure("who_discussion_b", foreground=t["discussion_b"],
                                     font=(t["family"], t["small"], "bold"))
        self.chat_text.tag_configure("who_reasoning", foreground=t["muted"],
                                     font=(t["family"], t["small"], "bold italic"))
        self.chat_text.tag_configure("reasoning_text", foreground=t["muted"],
                                     font=(t["family"], t["small"], "italic"),
                                     lmargin1=14, lmargin2=14, spacing3=2)

        self.chat_menu = tk.Menu(self.chat_text, tearoff=0)
        self.chat_text.bind("<Button-3>", self._on_chat_right_click)

        entry_row = tk.Frame(paned, bg=t["bg"])
        paned.add(entry_row, minsize=60, stretch="never")
        self.chat_input = tk.Text(entry_row, height=4, wrap="word", bg=t["input"], fg=t["text"], relief=tk.SOLID,
                                  bd=1, font=font_chat, padx=6, pady=4)
        self.chat_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_input.bind("<Return>", self._on_enter)
        self.chat_input.focus_set()

        self.send_btn = tk.Button(entry_row, text="Send", command=self.send, bg=t["accent"], fg=t["on_accent"],
                                  activebackground=t["accent"], activeforeground=t["on_accent"], relief=tk.FLAT,
                                  bd=0, padx=16, pady=6, cursor="hand2", font=font)
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0), fill=tk.Y)

        self.stop_btn = tk.Button(entry_row, text="Stop", command=self.stop_discussion, bg="#d64545",
                                  fg=t["on_accent"], activebackground="#d64545", activeforeground=t["on_accent"],
                                  relief=tk.FLAT, bd=0, padx=12, pady=6, cursor="hand2", font=font)
        # not packed here - only shown while a discussion is actually running (_apply_mode_visuals)

        tk.Label(win, text="Enter sends, Shift+Enter adds a new line.",
                 bg=t["bg"], fg=t["muted"], font=(t["family"], t["small"]),
                 anchor=tk.W).pack(fill=tk.X, padx=14, pady=(0, 10))

        # Claude-style model/effort chip, sitting inside the input box's bottom-right corner: click
        # for the model picker, Effort, Reasoning on/off and Show thinking (menu built in
        # _show_reasoning_menu).
        self.reasoning_btn = tk.Button(self.chat_input, text="", command=self._show_reasoning_menu,
                                       bg=t["input"], fg=t["muted"], activebackground=t["neutral"],
                                       activeforeground=t["text"], relief=tk.FLAT, bd=0, padx=6,
                                       pady=1, cursor="hand2", font=(t["family"], t["small"]))
        self.reasoning_btn.place(relx=1.0, rely=1.0, x=-6, y=-4, anchor="se")
        self._refresh_reasoning_controls()

        self._apply_mode_visuals()

    def new_chat(self):
        """Clears the Chat-mode conversation: wipes the local message history sent to the
        model, resets the token counter, and clears the visible log. Doesn't touch whatever
        model is currently loaded in the active backend - same 'leave models alone' pattern as
        Discussion's Stop button. Only meaningful in Chat mode; blocked mid-reply or mid-
        discussion so it can't race a request that's already in flight."""
        if self.awaiting_reply or self.discussion_running:
            return
        self.conversation = []
        self.speaker.stop()
        self.total_tokens = 0
        self.token_lbl.config(text="Tokens: 0")
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def _on_enter(self, event):
        if event.state & 0x0001:                              # Shift+Enter -> normal new line
            return None
        self.send()
        return "break"

    def _append(self, text, tag=None):
        w = self.chat_text
        w.config(state=tk.NORMAL)
        w.insert(tk.END, text, (tag,) if tag else ())
        w.see(tk.END)
        w.config(state=tk.DISABLED)

    # ---- Terminator persona / lore ----------------------------------------------------------
    def _refresh_lore_status(self):
        if self.lore_path and self.lore_sections:
            txt = f"Lore: {os.path.basename(self.lore_path)} ({len(self.lore_sections)} sections)"
        elif self.lore_path:
            txt = f"Lore: {os.path.basename(self.lore_path)} (loads when persona is on)"
        else:
            txt = "No lore file found - persona only"
        self.lore_status_lbl.config(text=txt)

    def _load_lore(self):
        self.lore_sections = []
        if self.lore_path:
            try:
                self.lore_sections = load_lore_sections(self.lore_path)
            except Exception as exc:                                    # noqa: BLE001
                messagebox.showerror("Lore file", f"Couldn't read lore file:\n{exc}")
        self._refresh_lore_status()

    def _choose_lore_file(self):
        path = filedialog.askopenfilename(
            title="Choose lore bible", filetypes=[("Markdown / text", "*.md *.txt"), ("All files", "*.*")])
        if path:
            self.lore_path = path
            self._load_lore()

    def _on_speak_toggled(self):
        self.speak_enabled = bool(self.speak_var.get())
        if not self.speak_enabled:
            self.speaker.stop()
        elif not self.speaker.available:
            self._append("Speech needs Windows (uses the built-in System.Speech voices).\n\n", "system_msg")
            self.speak_var.set(False)
            self.speak_enabled = False
        else:
            self.speaker.speak("Voice online.")

    def _on_persona_toggled(self):
        self.persona_enabled = bool(self.persona_var.get())
        if self.persona_enabled:
            self._load_lore()
            self._append("Terminator persona ON. Replies are grounded in the lore file; anything "
                         "it does not cover should come back as UNKNOWN.\n\n", "system_msg")
        else:
            self._append("Terminator persona OFF.\n\n", "system_msg")

    def _messages_for_model(self):
        """The Chat-mode history, with the persona + lore instructions folded into the LAST user
        message when persona mode is on. Folding (rather than a separate system message) works
        with every chat template - Mistral-7B-Instruct's, for one, has no system role - and small
        models follow instructions placed next to the question far more reliably. History is
        trimmed to the last few messages because the lore is re-injected every turn. Nothing
        here is stored: self.conversation stays clean."""
        msgs = [dict(m) for m in self.conversation]
        if not self.persona_enabled:
            return msgs
        msgs = msgs[-PERSONA_HISTORY_MESSAGES:]
        while msgs and msgs[0].get("role") != "user":          # keep strict user/assistant alternation
            msgs.pop(0)
        users = [m["content"] for m in self.conversation if m.get("role") == "user"]
        query = " ".join(users[-2:])
        instructions = build_persona_system_message(query, self.lore_sections)
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["content"] = (instructions + "\n\n---\nUser message (answer as the Terminator, "
                                "using only the LORE REFERENCE for lore):\n" + m["content"])
                break
        return msgs

    def send(self):
        if self.mode == "discussion":
            self._handle_discussion_send()
            return
        if self.awaiting_reply:
            return
        text = self.chat_input.get("1.0", "end-1c").strip()
        if not text:
            return
        self.chat_input.delete("1.0", tk.END)
        self._append("You\n", "who_user")
        self._append(text + "\n\n")

        if not self.server_connected or not self.server_model:
            self._append(f"Not connected. Click \u201cConnect\u201d to start {self.backend.display_name} "
                         f"and pick a model.\n\n", "system_msg")
            return

        self.conversation.append({"role": "user", "content": text})
        self.awaiting_reply = True
        self.send_btn.config(state=tk.DISABLED, text="\u2026")
        self.new_chat_btn.config(state=tk.DISABLED)
        self._begin_stream_turn({"kind": "chat"})
        threading.Thread(target=self._stream_worker,
                         args=(self._messages_for_model(), self.server_model),
                         daemon=True).start()

    # ==========================================================================================
    #  STREAMING - shared by both Chat and Discussion. One request is ever in flight at a time,
    #  so the in-progress turn's state (which model/slot, how much reasoning/answer text has
    #  streamed in, the <think>-tag parser position) lives on self between chunks.
    # ==========================================================================================

    def _begin_stream_turn(self, ctx):
        self._stream_ctx = ctx
        self._stream_parse_state = {"mode": "pre", "buffer": ""}
        self._stream_reasoning_started = False
        self._stream_answer_started = False
        self._stream_full_reasoning = ""
        self._stream_full_answer = ""
        self._stream_usage = None
        self._stream_live_tokens = 0

    def _stream_worker(self, messages, model):
        try:
            for delta, _finish_reason, usage in self.backend.stream_chat(model, messages):
                if usage:
                    self._stream_usage = usage
                r_delta = delta.get("reasoning_content") or delta.get("reasoning")
                if r_delta:
                    self.root.after(0, self._stream_append, "reasoning", r_delta)
                c_delta = delta.get("content")
                if c_delta:
                    for kind, text in feed_think_state(self._stream_parse_state, c_delta):
                        self.root.after(0, self._stream_append, kind, text)
        except Exception as exc:                                        # noqa: BLE001
            self.root.after(0, self._stream_error, exc)
            return
        self.root.after(0, self._stream_finish)

    def _stream_header_for(self, kind):
        ctx = self._stream_ctx
        if ctx["kind"] == "discussion":
            if kind == "reasoning":
                return f"{ctx['name']} thinking (Discussion {ctx['slot']})\n", "who_reasoning"
            return f"{ctx['name']} (Discussion {ctx['slot']})\n", f"who_discussion_{ctx['slot'].lower()}"
        if kind == "reasoning":
            return "Thinking\n", "who_reasoning"
        return "Assistant\n", "who_assistant"

    def _stream_append(self, kind, text):
        if not text:
            return
        if kind == "reasoning":
            self._stream_live_tokens += max(1, len(text) // 4)
            self.token_lbl.config(
                text=f"Tokens: {self.total_tokens} | ~{self._stream_live_tokens} thinking…")
            if not self.show_thinking:              # still generated, just not displayed
                return
            if not self._stream_reasoning_started:
                header, tag = self._stream_header_for("reasoning")
                self._append(header, tag)
                self._stream_reasoning_started = True
            self._append(text, "reasoning_text")
            self._stream_full_reasoning += text
        else:
            self._stream_live_tokens += max(1, len(text) // 4)
            self.token_lbl.config(
                text=f"Tokens: {self.total_tokens} | ~{self._stream_live_tokens} gen…")
            if not self._stream_answer_started:
                if self._stream_reasoning_started:
                    self._append("\n\n")
                header, tag = self._stream_header_for("answer")
                self._append(header, tag)
                self._stream_answer_started = True
            self._append(text)
            self._stream_full_answer += text

    def _stream_finish(self):
        self._append("\n\n")
        self.last_response_text = self._stream_full_answer
        if self._stream_usage:
            total = self._stream_usage.get("total_tokens")
            if total is None:
                total = self._stream_usage.get("prompt_tokens", 0) + self._stream_usage.get(
                    "completion_tokens", 0)
            self.total_tokens += total
        else:
            self.total_tokens += self._stream_live_tokens
        self.token_lbl.config(text=f"Tokens: {self.total_tokens}")

        ctx = self._stream_ctx
        self.awaiting_reply = False
        if ctx["kind"] == "chat":
            self.send_btn.config(state=tk.NORMAL, text="Send")
            self.new_chat_btn.config(state=tk.NORMAL)
            self.conversation.append({"role": "assistant", "content": self._stream_full_answer})
            if self.speak_enabled and self._stream_full_answer.strip():
                self.speaker.speak(self._stream_full_answer)
            self._maybe_auto_unload()
        else:
            if not self.discussion_running:                    # ended/disconnected mid-stream
                return
            slot, other = ctx["slot"], ctx["other"]
            self.discussion_conv[slot].append({"role": "assistant", "content": self._stream_full_answer})
            self.discussion_conv[other].append({"role": "user", "content": self._stream_full_answer})
            self.discussion_transcript.append({"speaker": f"{ctx['name']} (Discussion {slot})",
                                               "text": self._stream_full_answer})
            while self.discussion_interject_queue:
                text = self.discussion_interject_queue.pop(0)
                self.discussion_conv["A"].append({"role": "user", "content": text})
                self.discussion_conv["B"].append({"role": "user", "content": text})
                self.discussion_transcript.append({"speaker": "You", "text": text})
                self._append("You\n", "who_user")
                self._append(text + "\n\n")
            self.discussion_turn = other
            if self.discussion_stop_flag:
                self._end_discussion()
            else:
                self._run_discussion_turn()

    def _stream_error(self, exc):
        ctx = self._stream_ctx
        self.awaiting_reply = False
        if ctx["kind"] == "chat":
            self.send_btn.config(state=tk.NORMAL, text="Send")
            self.new_chat_btn.config(state=tk.NORMAL)
            if self.conversation and self.conversation[-1]["role"] == "user":
                self.conversation.pop()                        # don't poison history with a failed turn
            self._append(f"Request failed: {exc}\n\n", "system_msg")
        else:
            if not self.discussion_running:
                return
            self._append(f"Discussion {ctx['slot']} request failed: {exc}\n\n", "system_msg")
            self._end_discussion()

    def _maybe_auto_unload(self):
        if not self.auto_unload_var.get():
            return
        if not self.server_connected or not self.server_model:
            return
        instance_id = self.server_instance_id or self.server_model
        threading.Thread(target=self._auto_unload_worker, args=(instance_id,), daemon=True).start()

    def _auto_unload_worker(self, instance_id):
        try:
            self.backend.unload_model(instance_id)
        except Exception as exc:                                # noqa: BLE001
            self.root.after(0, self._auto_unload_failed, exc)
            return
        self.root.after(0, self._auto_unload_done)

    def _auto_unload_done(self):
        key = self.server_model
        self.server_model = None
        self.server_instance_id = None
        self.update_server_button()
        self._append("Model unloaded.\n\n", "system_msg")
        if key is not None and self._models_win is not None:
            try:
                if self._models_win.win.winfo_exists():
                    self._models_win.mark_unloaded(key)
            except tk.TclError:
                pass

    def _auto_unload_failed(self, exc):
        self._append(f"Auto-unload failed \u2014 {exc}\n\n", "system_msg")

    # ==========================================================================================
    #  AUTO-MAX CONTEXT - the main window's toggle for loading every model at its max supported
    #  context. Turning it on bulk-reloads whatever's already loaded (using the last-fetched
    #  server catalog, so this works even if the Models window is closed); from then on the
    #  Models window's own loads (_ctx_for_load) pick it up automatically too.
    # ==========================================================================================

    def _on_auto_max_ctx_toggle(self):
        self.auto_max_context = self.auto_max_ctx_var.get()
        if not self.auto_max_context:
            return
        if not self.server_connected:
            self._append("Auto-max context enabled \u2014 will apply the next time you load a model.\n\n",
                         "system_msg")
            return
        targets = []
        for entry in self.server_catalog:
            key = entry.get("key")
            max_ctx = entry.get("max_context_length")
            instances = entry.get("loaded_instances") or []
            if not key or not max_ctx or not instances:
                continue
            inst = instances[0]
            cur_ctx = (inst.get("config") or {}).get("context_length")
            if cur_ctx == max_ctx:
                continue
            targets.append((key, entry.get("display_name", key), inst.get("id") or key, max_ctx))
        if not targets:
            self._append("Auto-max context enabled. New loads will use each model's max context "
                         "automatically.\n\n", "system_msg")
            return
        self._append(f"Auto-max context enabled \u2014 reloading {len(targets)} loaded model(s) at "
                     f"their max context\u2026\n\n", "system_msg")
        for key, name, instance_id, max_ctx in targets:
            threading.Thread(target=self._auto_max_reload_worker, args=(key, name, instance_id, max_ctx),
                             daemon=True).start()

    def _auto_max_reload_worker(self, key, name, old_instance_id, new_ctx):
        try:
            self.backend.unload_model(old_instance_id)
        except Exception as exc:                                        # noqa: BLE001
            self.root.after(0, self._auto_max_reload_failed, key, name, exc)
            return
        try:
            data = self.backend.load_model(key, new_ctx)
        except Exception as exc:                                        # noqa: BLE001
            self.root.after(0, self._auto_max_reload_failed, key, name, exc)
            return
        self.root.after(0, self._auto_max_reload_done, key, name, data, new_ctx)

    def _auto_max_reload_done(self, key, name, data, new_ctx):
        new_instance_id = data.get("instance_id") or key
        if self.server_model == key:
            self.server_instance_id = new_instance_id
        for info in self.discussion_models.values():
            if info and info.get("key") == key:
                info["instance_id"] = new_instance_id
        self._append(f"Reloaded {name} at {new_ctx} tokens (auto-max).\n\n", "system_msg")
        if self._models_win is not None:
            try:
                if self._models_win.win.winfo_exists():
                    self._models_win.mark_reloaded(key, new_instance_id, new_ctx)
            except tk.TclError:
                pass

    def _auto_max_reload_failed(self, key, name, exc):
        if self.server_model == key:
            self.server_model = None
            self.server_instance_id = None
            self.update_server_button()
        for slot, info in list(self.discussion_models.items()):
            if info and info.get("key") == key:
                self.discussion_models[slot] = None
                self.on_discussion_models_changed()
        self._append(f"Auto-max reload failed for {name} \u2014 {exc}. It may now be unloaded.\n\n",
                     "system_msg")
        if self._models_win is not None:
            try:
                if self._models_win.win.winfo_exists():
                    self._models_win.mark_unloaded(key)
            except tk.TclError:
                pass

    # ==========================================================================================
    #  MODE SWITCHING - toggles between plain Chat and Discussion (two models talking to each
    #  other, kept loaded together via the Models window's Discussion A/B tags).
    # ==========================================================================================

    def set_mode(self, mode):
        if mode == self.mode:
            return
        if self.discussion_running:
            self._append("Stop the discussion before switching back to Chat.\n\n", "system_msg")
            return
        if self.awaiting_reply:
            return
        self.mode = mode
        self._apply_mode_visuals()

    def _apply_mode_visuals(self):
        t = self.t
        active = dict(bg=t["accent"], fg=t["on_accent"], activebackground=t["accent"],
                     activeforeground=t["on_accent"])
        inactive = dict(bg=t["neutral"], fg=t["text"], activebackground=t["neutral"], activeforeground=t["text"])
        self.chat_mode_btn.config(**(active if self.mode == "chat" else inactive))
        self.discussion_mode_btn.config(**(active if self.mode == "discussion" else inactive))
        self.chat_mode_btn.config(state=tk.DISABLED if self.discussion_running else tk.NORMAL)
        self.new_chat_btn.config(state=tk.DISABLED if self.discussion_running else tk.NORMAL)

        if self.mode == "discussion":
            self.discussion_status_lbl.pack(side=tk.RIGHT)
            self._refresh_discussion_status()
            self.send_btn.config(text="Send" if self.discussion_running else "Start")
        else:
            self.discussion_status_lbl.pack_forget()
            self.send_btn.config(text="Send")

        self._show_stop_button(self.mode == "discussion" and self.discussion_running)

    def _refresh_discussion_status(self):
        a = self.discussion_models.get("A")
        b = self.discussion_models.get("B")
        if a and b:
            self.discussion_status_lbl.config(text=f"A: {a['display_name']}    B: {b['display_name']}",
                                              fg=self.t["muted"])
        else:
            missing = [s for s, v in (("A", a), ("B", b)) if not v]
            self.discussion_status_lbl.config(
                text=f"Assign Discussion {' and '.join(missing)} in the Models window", fg="#d64545")

    def _show_stop_button(self, show):
        if show:
            self.stop_btn.pack(side=tk.LEFT, padx=(8, 0), fill=tk.Y)
        else:
            self.stop_btn.pack_forget()

    def toggle_simple_mode(self):
        """Collapses the window down to just the chat log and input box, hiding the title,
        button row, Constraints Engine card, mode row, and token counter behind a single
        small 'Exit Simple Mode' bar. Purely a visual collapse - whatever's loaded or running
        underneath (server connection, model, discussion pairing, engine state, an in-flight
        discussion) is untouched and keeps working; Send/Stop stay live in the entry row,
        which is never hidden by this."""
        self.simple_mode = not self.simple_mode
        if self.simple_mode:
            self.head.pack_forget()
            self.head2.pack_forget()
            self.constraints_toggle_row.pack_forget()
            self.constraints_card.pack_forget()
            self.mode_row.pack_forget()
            self.sub_head.pack_forget()
            self.simple_mode_bar.pack(fill=tk.X, padx=14, pady=(12, 0), before=self.paned)
        else:
            self.simple_mode_bar.pack_forget()
            self.head.pack(fill=tk.X, padx=14, pady=(12, 0), before=self.paned)
            self.head2.pack(fill=tk.X, padx=14, pady=(6, 0), before=self.paned)
            self.constraints_toggle_row.pack(fill=tk.X, padx=14, pady=(8, 0), before=self.paned)
            if not self.constraints_collapsed:
                self.constraints_card.pack(fill=tk.X, padx=14, pady=(4, 0), before=self.paned)
            self.mode_row.pack(fill=tk.X, padx=14, pady=(6, 0), before=self.paned)
            self.sub_head.pack(fill=tk.X, padx=14, pady=(6, 6), before=self.paned)
            self._apply_mode_visuals()

    # ==========================================================================================
    #  CHAT LOG RIGHT-CLICK MENU - Copy, Highlighted Text Count, Export to Notes. Bound once
    #  on self.chat_text, so it works the same in Simple Mode and normal mode (the log/input
    #  are never hidden by Simple Mode) and in both Chat and Discussion.
    # ==========================================================================================

    def _on_chat_right_click(self, event):
        has_selection = bool(self.chat_text.tag_ranges(tk.SEL))
        menu = self.chat_menu
        menu.delete(0, tk.END)
        menu.add_command(label="Copy", command=self._chat_copy_selection,
                         state=tk.NORMAL if has_selection else tk.DISABLED)
        menu.add_command(label="Highlighted Text Count", command=self._chat_show_selection_count,
                         state=tk.NORMAL if has_selection else tk.DISABLED)
        menu.add_separator()
        menu.add_command(label="Export to Notes", command=self._export_last_response_to_notes)
        menu.tk_popup(event.x_root, event.y_root)

    def _chat_copy_selection(self):
        if not self.chat_text.tag_ranges(tk.SEL):
            return
        selected = self.chat_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        self.chat_text.clipboard_clear()
        self.chat_text.clipboard_append(selected)

    def _chat_show_selection_count(self):
        if not self.chat_text.tag_ranges(tk.SEL):
            return
        selected = self.chat_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        messagebox.showinfo("Highlighted Text Count", f"{len(selected)} character(s) selected.")

    def _export_last_response_to_notes(self):
        text = self.last_response_text.strip()
        if not text:
            messagebox.showinfo("Export to Notes", "No response yet to export.")
            return
        if self._notes_win is not None:
            try:
                if self._notes_win.win.winfo_exists():
                    self._notes_win.append_text(text)
                    return
            except tk.TclError:
                self._notes_win = None
        self.notes_text = (self.notes_text + "\n\n" if self.notes_text.strip() else "") + text

    # ==========================================================================================
    #  DISCUSSION ENGINE - two independent conversation histories (each model sees itself as
    #  "assistant" and everything else - the other model, your interjections, the seed topic -
    #  as "user"), alternated by a background loop until Stop is pressed or a call fails.
    # ==========================================================================================

    def _handle_discussion_send(self):
        text = self.chat_input.get("1.0", "end-1c").strip()
        if not self.discussion_running:
            if self.awaiting_reply:
                return
            a = self.discussion_models.get("A")
            b = self.discussion_models.get("B")
            if not (a and b):
                self._append("Assign both Discussion A and B models in the Models window first.\n\n",
                             "system_msg")
                return
            if not self.server_connected:
                self._append("Not connected to a server. Click \u201cConnect\u201d first.\n\n", "system_msg")
                return
            if not text:
                self._append("Type a topic to start the discussion.\n\n", "system_msg")
                return
            self.chat_input.delete("1.0", tk.END)
            self._start_discussion(text)
        else:
            if not text:
                return
            self.chat_input.delete("1.0", tk.END)
            self._queue_interjection(text)

    def _start_discussion(self, topic):
        self.discussion_conv = {"A": [{"role": "user", "content": topic}],
                                "B": [{"role": "user", "content": topic}]}
        self.discussion_transcript = [{"speaker": "Topic", "text": topic}]
        self.discussion_interject_queue = []
        self.discussion_stop_flag = False
        self.discussion_running = True
        self.discussion_turn = "A"
        self._append("Topic\n", "who_user")
        self._append(topic + "\n\n")
        self._apply_mode_visuals()
        self._run_discussion_turn()

    def stop_discussion(self):
        if not self.discussion_running:
            return
        self.discussion_stop_flag = True
        self._append("Stopping after the current reply\u2026\n\n", "system_msg")

    def _queue_interjection(self, text):
        self.discussion_interject_queue.append(text)
        self._append("(queued \u2014 will drop in after the current reply)\n", "system_msg")

    def _run_discussion_turn(self):
        if not self.discussion_running or self.discussion_stop_flag:
            self._end_discussion()
            return
        slot = self.discussion_turn
        model_info = self.discussion_models.get(slot)
        if model_info is None:                    # unassigned mid-run, e.g. the server dropped it
            self._append(f"Discussion {slot} is no longer loaded \u2014 stopping.\n\n", "system_msg")
            self._end_discussion()
            return
        messages = list(self.discussion_conv[slot])
        self.awaiting_reply = True
        other = "B" if slot == "A" else "A"
        self._begin_stream_turn({"kind": "discussion", "slot": slot, "other": other,
                                 "name": model_info["display_name"]})
        threading.Thread(target=self._stream_worker,
                         args=(messages, model_info["key"]),
                         daemon=True).start()

    def _end_discussion(self):
        was_running = self.discussion_running
        self.discussion_running = False
        self.discussion_stop_flag = False
        self.awaiting_reply = False
        if was_running:
            self._append("Discussion stopped. Both models are still loaded \u2014 press Start to resume, "
                         "or switch back to Chat.\n\n", "system_msg")
        self._apply_mode_visuals()

    def _on_ram_spill_toggle(self):
        self.ram_spillover = self.ram_spill_var.get()

    def _on_gpu_layers_slider(self, _val=None):
        v = self.gpu_layers_slider_var.get()
        self.gpu_layers_override = None if v == 0 else v
        self._update_gpu_layers_label()

    def _gpu_layers_text(self):
        v = self.gpu_layers_slider_var.get()
        if v == 0:
            return "GPU layers: Auto"
        text = f"GPU layers: {v}"
        try:
            if self.backend is not None:
                slots = self.backend.slots
                if slots:
                    handle = next(iter(slots.values()))
                    size_bytes = os.path.getsize(handle.path)
                    meta = self.backend._get_metadata(handle.path, size_bytes)
                    n_layers = meta.get("n_layers", 0)
                    if n_layers > 0:
                        model_mb = size_bytes / (1024 * 1024)
                        kv_mb = handle.ctx * 0.06
                        per_layer = (model_mb + kv_mb) / n_layers
                        actual_v = min(v, n_layers)
                        gpu_mb = actual_v * per_layer + 250
                        ram_mb = max(0, (n_layers - actual_v)) * per_layer
                        text = (f"{v} layers  |  ~{gpu_mb/1024:.1f} GB GPU"
                                f" / ~{ram_mb/1024:.1f} GB RAM")
        except Exception:
            pass
        return text

    def _update_gpu_layers_label(self):
        lbl = self.gpu_layers_label
        if lbl is None:
            return
        try:
            lbl.config(text=self._gpu_layers_text())
        except tk.TclError:
            self.gpu_layers_label = None

    def _open_gpu_layers_dialog(self):
        """Small window with the GPU layers slider (a slider can't live inside a menu)."""
        win = self._gpu_layers_win
        try:
            if win is not None and win.winfo_exists():
                win.lift()
                return
        except tk.TclError:
            pass
        t = self.t
        win = tk.Toplevel(self.root)
        win.title("GPU layers")
        win.configure(bg=t["card"])
        win.resizable(False, False)
        win.transient(self.root)
        self._gpu_layers_win = win
        tk.Label(win, text="0 = Auto. Applies to the next model load.", bg=t["card"], fg=t["muted"],
                 font=(t["family"], t["small"])).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Scale(win, from_=0, to=60, orient=tk.HORIZONTAL, variable=self.gpu_layers_slider_var,
                 command=self._on_gpu_layers_slider, bg=t["card"], fg=t["muted"],
                 troughcolor=t["neutral"], highlightthickness=0, bd=0, showvalue=False,
                 length=260, sliderlength=14, width=10).pack(padx=12)
        self.gpu_layers_label = tk.Label(win, text=self._gpu_layers_text(), bg=t["card"], fg=t["muted"],
                                         font=(t["family"], t["small"]))
        self.gpu_layers_label.pack(anchor="w", padx=12, pady=(4, 4))
        tk.Button(win, text="Close", command=win.destroy, bg=t["neutral"], fg=t["text"],
                  activebackground=t["neutral"], relief=tk.FLAT, bd=0, padx=10, pady=2,
                  cursor="hand2", font=(t["family"], t["small"])).pack(pady=(0, 10))

    def _on_reasoning_changed(self):
        self.reasoning_enabled = self.reasoning_var.get()
        self.reasoning_level = self.reasoning_level_var.get() or "Medium"
        self._refresh_reasoning_controls()

    def _on_show_thinking_changed(self):
        self.show_thinking = self.show_thinking_var.get()

    def _active_model_name(self):
        """Display name of the model chat is using right now, or None."""
        key = self.server_model
        if not key:
            return None
        for entry in self.server_catalog:
            if entry.get("key") == key:
                return entry.get("display_name") or os.path.basename(str(key))
        return os.path.basename(str(key))

    def _refresh_reasoning_controls(self):
        """Keeps the chip in the input box in step with the settings, Claude-style:
        '<model>  <effort> \u25BE' - effort reads 'Off' while Reasoning is off."""
        if not hasattr(self, "reasoning_btn"):
            return
        name = self._active_model_name() or "No model"
        if len(name) > 28:
            name = name[:27] + "\u2026"
        effort = self.reasoning_level if self.reasoning_enabled else "Off"
        self.reasoning_btn.config(text=f"{name}  {effort}  \u25BE")

    def _model_is_loaded(self, key):
        be = self.backend
        return bool(be is not None and be.slots.get(be.key_to_instance.get(key)) is not None)

    def _show_reasoning_menu(self):
        """Pop the chip's menu open above it. Built fresh each time so the model list and
        checkmarks reflect what's loaded right now."""
        t = self.t
        mfont = (t["family"], t["size"])
        menu = tk.Menu(self.root, tearoff=0, font=mfont)

        name = self._active_model_name()
        if name and self.server_connected:
            menu.add_command(label=f"\u2713  {name}")
        else:
            menu.add_command(label="No model loaded" if self.server_connected else "Not connected",
                             state=tk.DISABLED)

        effort_menu = tk.Menu(menu, tearoff=0, font=mfont)
        for level in ("Low", "Medium"):
            effort_menu.add_radiobutton(label=level, value=level, variable=self.reasoning_level_var,
                                        command=self._on_reasoning_changed)
        menu.add_cascade(label="Effort", menu=effort_menu,
                         state=tk.NORMAL if self.reasoning_enabled else tk.DISABLED)

        more = tk.Menu(menu, tearoff=0, font=mfont)
        catalog = sorted(self.server_catalog, key=lambda e: e.get("size_bytes") or 0, reverse=True)
        if not self.server_connected:
            more.add_command(label="Connect first", state=tk.DISABLED)
        elif not catalog:
            more.add_command(label="No .gguf models found", state=tk.DISABLED)
        for entry in catalog:
            key = entry.get("key")
            label = entry.get("display_name") or os.path.basename(str(key))
            if self._model_is_loaded(key):
                label += "   \u2713"
            more.add_command(label=label, command=lambda k=key: self._pick_model(k))
        menu.add_cascade(label="More models", menu=more)

        menu.add_separator()
        menu.add_checkbutton(label="Reasoning", variable=self.reasoning_var,
                             command=self._on_reasoning_changed)
        menu.add_checkbutton(label="Show thinking", variable=self.show_thinking_var,
                             command=self._on_show_thinking_changed)
        menu.add_separator()
        menu.add_checkbutton(label="Auto-max context for all models", variable=self.auto_max_ctx_var,
                             command=self._on_auto_max_ctx_toggle)
        menu.add_checkbutton(label="Allow RAM spillover (Local)", variable=self.ram_spill_var,
                             command=self._on_ram_spill_toggle)
        v = self.gpu_layers_slider_var.get()
        menu.add_command(label=f"GPU layers: {'Auto' if v == 0 else v}\u2026",
                         command=self._open_gpu_layers_dialog,
                         state=tk.NORMAL if self.ram_spillover else tk.DISABLED)
        menu.add_separator()
        menu.add_command(label="Manage models\u2026", command=self.open_models_viewer)

        menu.update_idletasks()
        b = self.reasoning_btn
        x = b.winfo_rootx() + b.winfo_width() - menu.winfo_reqwidth()
        y = max(0, b.winfo_rooty() - menu.winfo_reqheight())
        try:
            menu.tk_popup(max(0, x), y)
        finally:
            menu.grab_release()

    def _pick_model(self, key):
        """Load (or, if already loaded, switch chat to) a model chosen from the chip menu. Uses
        the Models window's own loader, so eviction, context length, spillover and
        Discussion protection all behave exactly like double-clicking there."""
        if self.awaiting_reply or self.discussion_running:
            self._append("Can't change models mid-reply or mid-discussion \u2014 finish or stop "
                         "first.\n\n", "system_msg")
            return
        viewer = self._ensure_models_viewer()
        if not viewer.models or not any(e.get("key") == key for e in viewer.models):
            viewer.apply_catalog(self.server_catalog)
        index = next((i for i, e in enumerate(viewer.models) if e.get("key") == key), None)
        if index is None:
            return
        state = viewer.model_state[key]
        if state["loaded"]:
            self.server_model = key
            self.server_instance_id = state["instance_id"]
            self.update_server_button()
            return
        self._append(f"Loading {viewer.models[index].get('display_name', key)}\u2026\n\n", "system_msg")
        viewer._load(index)

    def _ensure_models_viewer(self):
        """The Models window object, created hidden if the user hasn't opened it - the chip
        menu drives loads through it."""
        if self._models_win is not None:
            try:
                if self._models_win.win.winfo_exists():
                    return self._models_win
            except tk.TclError:
                pass
            self._models_win = None
        self._models_win = ModelsViewer(self.root, self.t, self)
        self._models_win.win.withdraw()
        return self._models_win

    def connect_server(self):
        if self.backend is None:
            self._append("llama-cpp-python isn't installed in this environment \u2014 can't connect. "
                         "Install it with: pip install llama-cpp-python\n\n", "system_msg")
            return
        self.connect_btn.config(state=tk.DISABLED, text="Authenticating\u2026")
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        try:
            self.backend.connect()
        except Exception as exc:                                # noqa: BLE001
            self.root.after(0, self._connect_failed, exc)
            return
        self.root.after(0, self._connect_done)

    def _connect_done(self):
        self.connect_btn.config(state=tk.DISABLED, text="Authenticate")
        self.on_server_connected()
        self._append(f"Connected \u2014 using {self.backend.display_name}.\n\n", "system_msg")
        self.fetch_model_catalog()             # auto-populate the Models window's matches

    def _connect_failed(self, exc):
        self.connect_btn.config(state=tk.NORMAL, text="Authenticate")
        self._append(f"Could not connect ({self.backend.display_name}) \u2014 {exc}\n\n", "system_msg")

    def disconnect_server(self):
        if self.discussion_running:
            self._append("Disconnecting \u2014 discussion stopped.\n\n", "system_msg")
            self._end_discussion()
        try:
            self.backend.disconnect()           # frees GPU VRAM for the Local backend
        except Exception:                       # noqa: BLE001 - best-effort cleanup
            pass
        self.on_server_disconnected()
        self.connect_btn.config(state=tk.NORMAL, text="Authenticate")

    def fetch_model_catalog(self):
        """Pull the active backend's model catalog and, if the Models window is open, push the
        refreshed matches into it right away - this is the "automatic" half of Connect."""
        threading.Thread(target=self._fetch_catalog_worker, daemon=True).start()

    def _fetch_catalog_worker(self):
        try:
            models = self.backend.fetch_catalog()
        except Exception as exc:                                # noqa: BLE001
            self.root.after(0, self._fetch_catalog_failed, exc)
            return
        self.root.after(0, self._fetch_catalog_done, models)

    def _fetch_catalog_done(self, models):
        self.server_catalog = models
        if self._models_win is not None:
            try:
                if self._models_win.win.winfo_exists():
                    self._models_win.apply_catalog(models)
            except tk.TclError:
                pass

    def _fetch_catalog_failed(self, exc):
        self._append(f"Connected, but could not fetch the model catalog \u2014 {exc}\n\n", "system_msg")
        if self._models_win is not None:
            try:
                if self._models_win.win.winfo_exists():
                    self._models_win._set_status(f"Could not fetch the server's model catalog \u2014 {exc}")
            except tk.TclError:
                pass

    def on_server_connected(self):
        self.server_connected = True
        self.update_server_button()

    def on_server_disconnected(self):
        self.server_connected = False
        self.server_model = None
        self.server_instance_id = None
        self.server_catalog = []
        self.update_server_button()

    def update_server_button(self):
        """Dot + label give an unambiguous read at a glance: red/"Disconnected" when there's
        no server connection, green whenever there is - showing the loaded model's name in
        place of a plain "Connected" once one's active, same as before."""
        if self.server_connected and self.server_model:
            self.server_dot.itemconfig(self._server_dot_id, fill="#22c55e")
            self.server_label.config(text=self.server_model)
        elif self.server_connected:
            self.server_dot.itemconfig(self._server_dot_id, fill="#22c55e")
            self.server_label.config(text="Connected")
        else:
            self.server_dot.itemconfig(self._server_dot_id, fill="#ef4444")
            self.server_label.config(text="Disconnected")
        self._refresh_reasoning_controls()

    # ==========================================================================================
    #  CONSTRAINTS ENGINE - a simplified port of the lmengine "Constraints Engine" panel: an
    #  Enabled checkbox, a status line, and Edit Engine / Test Engine buttons.
    #  Uses the ConstraintEngine class embedded near the top of this file (formerly a separate
    #  constraint_engine.py that had to sit next to this script - now merged in directly).
    #  Wiring this into the actual reply pipeline (_stream_finish) is a separate step.
    # ==========================================================================================

    def _toggle_constraints_card(self):
        self.constraints_collapsed = not self.constraints_collapsed
        if self.constraints_collapsed:
            self.constraints_card.pack_forget()
            self.constraints_toggle_btn.config(text="\u25B8 Show Constraints Engine")
        else:
            self.constraints_card.pack(fill=tk.X, padx=14, pady=(4, 0), before=self.mode_row)
            self.constraints_toggle_btn.config(text="\u25BE Hide Constraints Engine")

    def _on_constraints_toggle(self):
        if self.constraints_enabled_var.get():
            if self.constraint_engine is None:
                engine = self._build_empty_constraint_engine()
                if engine is None:
                    self.constraints_enabled_var.set(False)
                    self.constraints_enabled = False
                    self._update_constraints_status()
                    return
                self.constraint_engine = engine
            self.constraints_enabled = True
        else:
            self.constraints_enabled = False
        self._update_constraints_status()

    def _build_empty_constraint_engine(self):
        """Builds a fresh engine with zero constraints active - the user turns individual
        constraints on/off afterwards via Edit Engine. Nothing here persists: this is called
        again from scratch every time "Enabled" is checked, including after a restart."""
        try:
            return ConstraintEngine()
        except Exception as exc:                                       # noqa: BLE001
            messagebox.showerror("Constraints Engine", f"Failed to start the engine:\n{exc}")
            return None

    def _update_constraints_status(self):
        if not self.constraints_enabled or self.constraint_engine is None:
            self.constraints_status_lbl.config(text="Engine Disabled", fg=self.t["muted"])
            return
        try:
            n_sep = len(self.constraint_engine.separate_constraints)
            n_col = len(self.constraint_engine.collective_constraints)
            self.constraints_status_lbl.config(text=f"Engine Running ({n_sep} separate, {n_col} collective)",
                                               fg="#16a34a")
        except AttributeError:
            self.constraints_status_lbl.config(text="Engine Running", fg="#16a34a")

    # Every SeparateConstraint / CollectiveConstraint subclass defined in the embedded
    # Constraint Engine section above, by exact class name - Edit Engine lists each of
    # these as one checkbox.
    _SEPARATE_CONSTRAINT_NAMES = (
        "IsNonEmpty", "NoDoubleSpaces", "NoTrailingWhitespace", "CapitalizeFirst", "MaxLength",
        "MinLength", "ValidJSON", "ContainsKeyword", "NoCharacters", "NoMarkdown", "IsEnglish",
        "NoURLs", "NoPlaceholderText", "EndsWithPunctuation", "NoRefusalLanguage",
    )
    _COLLECTIVE_CONSTRAINT_NAMES = (
        "NoContradictions", "JSONFieldsPopulated", "LogicalFlow", "NoRepetition",
        "FactualGrounding", "MinDistinctSentences", "StyleConsistency",
    )

    def open_constraints_editor(self):
        if self.constraint_engine is None:
            messagebox.showinfo("Constraints Engine", "Enable the engine first.")
            return
        if self._constraint_editor_win is not None:
            try:
                if self._constraint_editor_win.win.winfo_exists():
                    self._constraint_editor_win.win.lift()
                    self._constraint_editor_win.win.focus_force()
                    return
            except tk.TclError:
                pass
            self._constraint_editor_win = None
        separate_classes = [(n, globals()[n]) for n in self._SEPARATE_CONSTRAINT_NAMES]
        collective_classes = [(n, globals()[n]) for n in self._COLLECTIVE_CONSTRAINT_NAMES]
        self._constraint_editor_win = ConstraintCheckboxEditor(
            self.root, self.t, self.constraint_engine, separate_classes, collective_classes,
            on_change=self._update_constraints_status)

    def open_constraints_test(self):
        if self.constraint_engine is None:
            messagebox.showinfo("Constraints Engine", "Enable the engine first.")
            return
        if self._constraint_test_win is not None:
            try:
                if self._constraint_test_win.winfo_exists():
                    self._constraint_test_win.lift()
                    self._constraint_test_win.focus_force()
                    return
            except tk.TclError:
                pass
            self._constraint_test_win = None

        t = self.t
        win = tk.Toplevel(self.root)
        win.title("Test Engine")
        win.geometry("520x420")
        win.minsize(400, 300)
        win.configure(bg=t["bg"])
        self._constraint_test_win = win

        tk.Label(win, text="Paste sample text and run it through the engine:",
                bg=t["bg"], fg=t["text"], font=(t["family"], t["small"])).pack(
                anchor=tk.W, padx=12, pady=(12, 4))

        input_box = tk.Text(win, height=8, wrap="word", bg=t["input"], fg=t["text"],
                            relief=tk.SOLID, bd=1, font=(t["family"], t["size"]))
        input_box.pack(fill=tk.X, padx=12)

        report_box = tk.Text(win, wrap="word", bg=t["card"], fg=t["text"], relief=tk.SOLID,
                             bd=1, font=(t["family"], t["small"]), state=tk.DISABLED)
        report_sb = ttk.Scrollbar(win, orient=tk.VERTICAL, command=report_box.yview)
        report_box.configure(yscrollcommand=report_sb.set)

        def run_test():
            sample = input_box.get("1.0", "end-1c")
            try:
                result = self.constraint_engine.validate(sample)
            except AttributeError:
                report_text = "ConstraintEngine has no validate() method."
            except Exception as exc:                                    # noqa: BLE001
                report_text = f"Test failed:\n{exc}"
            else:
                report_text = str(result)
            report_box.config(state=tk.NORMAL)
            report_box.delete("1.0", tk.END)
            report_box.insert(tk.END, report_text)
            report_box.config(state=tk.DISABLED)

        tk.Button(win, text="Run", command=run_test, bg=t["accent"], fg=t["on_accent"],
                 activebackground=t["accent"], activeforeground=t["on_accent"], relief=tk.FLAT,
                 bd=0, padx=12, pady=4, cursor="hand2", font=(t["family"], t["small"])
                 ).pack(anchor=tk.W, padx=12, pady=8)

        report_sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12))
        report_box.pack(fill=tk.BOTH, expand=True, padx=(12, 0), pady=(0, 12))

    # ==========================================================================================
    #  SAVE LOCATIONS - right-click the title bar (blank area or the "Simple Chat" text) for
    #  "Save Locations", which shows/changes the folder every Save action in the app writes
    #  into. Asked once at boot (skippable); if never set (or skipped), every Save action
    #  falls back to a normal Save As browse dialog, exactly as before. Like everything else
    #  in this app, the chosen folder is session-only and is asked again fresh next launch.
    # ==========================================================================================

    def _on_head_right_click(self, event):
        menu = self.head_menu
        menu.delete(0, tk.END)
        menu.add_command(label="Save Locations", command=self._open_save_locations)
        menu.add_command(label="Local Models Folder", command=self._open_local_models_folder)
        menu.add_command(
            label=("Stop HTTP API" if self.http_bridge.running
                   else f"Start HTTP API ({BRIDGE_HOST}:{BRIDGE_PORT})"),
            command=self._toggle_http_bridge)
        menu.add_separator()
        menu.add_command(label="Drawing", command=self._open_drawing)
        menu.add_command(label="Music", command=self._open_music)
        menu.add_command(label="Images (SD 1.5)", command=self._open_images)
        menu.add_command(label="Open Block Editor", command=self._open_block_editor)
        if self.block_editor.running:
            menu.add_command(label="Stop Block Editor", command=self._stop_block_editor)
        menu.tk_popup(event.x_root, event.y_root)

    def _open_drawing(self):
        """Watch the loaded model draw (see the DRAWING section)."""
        win = getattr(self, "_drawing_win", None)
        try:
            if win is not None and win.win.winfo_exists():
                win.win.lift()
                win.win.focus_force()
                return
        except tk.TclError:
            pass
        self._drawing_win = DrawingWindow(self.root, self.t, self)

    def _open_music(self):
        """Have the loaded model compose music (see the MUSIC section)."""
        win = getattr(self, "_music_win", None)
        try:
            if win is not None and win.win.winfo_exists():
                win.win.lift()
                win.win.focus_force()
                return
        except tk.TclError:
            pass
        self._music_win = MusicWindow(self.root, self.t, self)

    def _open_images(self):
        """Generate images with stable-diffusion.cpp (see the IMAGE GENERATION section)."""
        win = getattr(self, "_images_win", None)
        try:
            if win is not None and win.win.winfo_exists():
                win.win.lift()
                win.win.focus_force()
                return
        except tk.TclError:
            pass
        self._images_win = ImagesWindow(self.root, self.t, self)

    def _open_block_editor(self):
        """Start the built-in Block Editor (if needed) and open it in the browser."""
        try:
            already = self.block_editor.running
            self.block_editor.start()
        except ImportError:
            self._append("The Block Editor needs Flask. Run:  pip install flask   then try again.\n\n", "system_msg")
            return
        except OSError as exc:
            self._append(f"Couldn't start the Block Editor: {exc}\n\n", "system_msg")
            return
        if not already:
            self._append(f"Block Editor running at {self.block_editor.url}\n", "system_msg")
            if not self.server_connected or self.local_backend is None:
                self._append("It uses the Local model loaded here: press Authenticate and load a model "
                             "in the Models window before pressing Build.\n", "system_msg")
            self._append("\n", "system_msg")
        webbrowser.open(self.block_editor.url)

    def _stop_block_editor(self):
        self.block_editor.stop()
        self._append("Block Editor stopped.\n\n", "system_msg")

    def _toggle_http_bridge(self):
        """Session-only on/off for the HTTP bridge (see HTTP BRIDGE section)."""
        try:
            if self.http_bridge.running:
                self.http_bridge.stop()
                self._append("HTTP API stopped.\n\n", "system_msg")
            else:
                self.http_bridge.start()
                self._append(f"HTTP API listening on http://{BRIDGE_HOST}:{BRIDGE_PORT} \u2014 other "
                             "programs can now use whichever Local model is loaded here.\n\n",
                             "system_msg")
        except OSError as exc:
            self._append(f"Couldn't start the HTTP API: {exc}\n\n", "system_msg")

    def _prompt_save_folder_at_boot(self):
        self.save_folder = self._prompt_save_folder(
            title="Save Location",
            intro="Choose a folder for this session's Save actions (Notes, and anything else "
                  "you save from this app). Skip to be asked to browse for a location "
                  "manually each time instead.")

    def _open_save_locations(self):
        chosen = self._prompt_save_folder(
            title="Save Locations",
            intro=(f"Current save folder:\n{self.save_folder}" if self.save_folder
                   else "No save folder is set - Save actions will ask you to browse each time."))
        if chosen:
            self.save_folder = chosen
        # Cancelling/leaving it blank here keeps whatever was already set, if anything.

    def _open_local_models_folder(self):
        """Where the llama.cpp backend looks for .gguf files. Takes effect the next time
        you Connect."""
        chosen = self._prompt_save_folder(
            title="Local Models Folder",
            intro=f"Folder the Local (llama.cpp) backend scans for .gguf files:\n"
                  f"{self.local_models_root}",
            initial=self.local_models_root)
        if chosen:
            self.local_models_root = chosen
            if self.local_backend is not None:
                self.local_backend.models_root = chosen

    def _prompt_save_folder(self, title="Save Location", intro=None, initial=None):
        """Modal folder picker: type a path directly, or Browse... for the OS folder picker.
        Returns the chosen folder, or None if cancelled / left blank."""
        t = self.t
        font = (t["family"], t["size"])
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.configure(bg=t["bg"])
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        if intro:
            tk.Label(dlg, text=intro, bg=t["bg"], fg=t["text"], font=font,
                    wraplength=360, justify=tk.LEFT).pack(padx=16, pady=(16, 8), anchor=tk.W)

        tk.Label(dlg, text="Folder:", bg=t["bg"], fg=t["muted"], font=font).pack(padx=16, anchor=tk.W)

        row = tk.Frame(dlg, bg=t["bg"])
        row.pack(padx=16, pady=(4, 4), fill=tk.X)
        entry_box = tk.Entry(row, font=font, bg=t["input"], fg=t["text"], relief=tk.SOLID, bd=1)
        entry_box.insert(0, (initial if initial is not None else self.save_folder) or "")
        entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry_box.select_range(0, tk.END)
        entry_box.focus_set()

        def browse():
            chosen = filedialog.askdirectory(title=title, initialdir=entry_box.get().strip() or os.getcwd())
            if chosen:
                entry_box.delete(0, tk.END)
                entry_box.insert(0, chosen)

        tk.Button(row, text="Browse...", command=browse, bg=t["neutral"], fg=t["text"],
                 activebackground=t["neutral"], relief=tk.FLAT, bd=0, padx=8, pady=4,
                 cursor="hand2", font=(t["family"], t["small"])).pack(side=tk.LEFT, padx=(6, 0))

        err_lbl = tk.Label(dlg, text="", bg=t["bg"], fg="#d64545", font=(t["family"], t["small"]))
        err_lbl.pack(padx=16, anchor=tk.W)

        result = {"path": None}

        def confirm():
            path = entry_box.get().strip()
            if not path:
                dlg.destroy()      # blank = same as Skip/Cancel
                return
            if not os.path.isdir(path):
                err_lbl.config(text="That folder doesn't exist.")
                return
            result["path"] = path
            dlg.destroy()

        def cancel():
            dlg.destroy()

        btn_row = tk.Frame(dlg, bg=t["bg"])
        btn_row.pack(padx=16, pady=(8, 16), fill=tk.X)
        cancel_label = "Cancel" if (initial if initial is not None else self.save_folder) else "Skip"
        tk.Button(btn_row, text=cancel_label, command=cancel, bg=t["neutral"], fg=t["text"],
                 activebackground=t["neutral"], relief=tk.FLAT, bd=0, padx=10, pady=5,
                 cursor="hand2", font=font).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(btn_row, text="OK", command=confirm, bg=t["accent"], fg=t["on_accent"],
                 activebackground=t["accent"], activeforeground=t["on_accent"], relief=tk.FLAT,
                 bd=0, padx=12, pady=5, cursor="hand2", font=font).pack(side=tk.RIGHT)

        dlg.bind("<Return>", lambda _e: confirm())
        dlg.bind("<Escape>", lambda _e: cancel())
        dlg.wait_window()
        return result["path"]

    def _prompt_filename(self, default_filename, dialog_title):
        """Small modal Entry dialog for just a file name, used when a save folder is already
        configured (so the OS Save As browser can be skipped entirely - only the name is
        still asked). Returns the name, or None if cancelled."""
        t = self.t
        font = (t["family"], t["size"])
        dlg = tk.Toplevel(self.root)
        dlg.title(dialog_title)
        dlg.configure(bg=t["bg"])
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text=f"Save to: {self.save_folder}", bg=t["bg"], fg=t["muted"],
                font=(t["family"], t["small"]), wraplength=360, justify=tk.LEFT).pack(
            padx=16, pady=(16, 4), anchor=tk.W)
        tk.Label(dlg, text="File name:", bg=t["bg"], fg=t["muted"], font=font).pack(padx=16, anchor=tk.W)

        entry_box = tk.Entry(dlg, font=font, bg=t["input"], fg=t["text"], relief=tk.SOLID, bd=1)
        entry_box.insert(0, default_filename)
        entry_box.pack(padx=16, pady=(4, 12), fill=tk.X)
        entry_box.select_range(0, tk.END)
        entry_box.focus_set()

        err_lbl = tk.Label(dlg, text="", bg=t["bg"], fg="#d64545", font=(t["family"], t["small"]))
        err_lbl.pack(padx=16, anchor=tk.W)

        result = {"name": None}

        def confirm():
            name = entry_box.get().strip()
            if not name:
                err_lbl.config(text="Enter a file name.")
                return
            result["name"] = name
            dlg.destroy()

        def cancel():
            dlg.destroy()

        btn_row = tk.Frame(dlg, bg=t["bg"])
        btn_row.pack(padx=16, pady=(0, 16), fill=tk.X)
        tk.Button(btn_row, text="Cancel", command=cancel, bg=t["neutral"], fg=t["text"],
                 activebackground=t["neutral"], relief=tk.FLAT, bd=0, padx=10, pady=5,
                 cursor="hand2", font=font).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(btn_row, text="Save", command=confirm, bg=t["accent"], fg=t["on_accent"],
                 activebackground=t["accent"], activeforeground=t["on_accent"], relief=tk.FLAT,
                 bd=0, padx=12, pady=5, cursor="hand2", font=font).pack(side=tk.RIGHT)

        dlg.bind("<Return>", lambda _e: confirm())
        dlg.bind("<Escape>", lambda _e: cancel())
        dlg.wait_window()
        return result["name"]

    def save_text_content(self, content, default_filename, dialog_title="Save"):
        """Shared save path for every 'Save' action in the app (Notes today; anything else
        that adds a Save action later should call this too). If a save folder is configured,
        writes straight into it after asking only for a file name - no OS browse dialog.
        Otherwise falls back to a normal Save As file-browser dialog, exactly as before."""
        if self.save_folder:
            filename = self._prompt_filename(default_filename, dialog_title)
            if not filename:
                return
            path = os.path.join(self.save_folder, filename)
        else:
            ext = os.path.splitext(default_filename)[1] or ".txt"
            path = filedialog.asksaveasfilename(
                title=dialog_title, initialfile=default_filename, defaultextension=ext,
                filetypes=([("SVG files", "*.svg")] if ext == ".svg" else [("Text files", "*.txt")])
                          + [("All files", "*.*")])
            if not path:
                return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:                                        # noqa: BLE001
            messagebox.showerror(dialog_title, f"Couldn't save file:\n{exc}")

    def open_prompt_editor(self):
        if self._prompt_win is not None:
            try:
                if self._prompt_win.win.winfo_exists():
                    self._prompt_win.win.lift()
                    self._prompt_win.win.focus_force()
                    return
            except tk.TclError:
                pass
            self._prompt_win = None
        self._prompt_win = HardcodedPromptsViewer(self.root, self.t)

    def open_notes(self):
        if self._notes_win is not None:
            try:
                if self._notes_win.win.winfo_exists():
                    self._notes_win.win.lift()
                    self._notes_win.win.focus_force()
                    return
            except tk.TclError:
                pass
            self._notes_win = None
        self._notes_win = NotepadWindow(self.root, self.t, "Notes", self.notes_text, self._set_notes_text,
                                        app=self)

    def _set_notes_text(self, text):
        self.notes_text = text

    def on_discussion_models_changed(self):
        """Called by the Models window whenever a Discussion A/B assignment changes (set,
        cleared, or invalidated because the server reports the model no longer loaded), so
        the status line and Start/Stop availability stay in sync even while it's hidden."""
        self._refresh_discussion_status()

    def open_models_viewer(self):
        if self._models_win is not None:
            try:
                if self._models_win.win.winfo_exists():
                    self._models_win.win.deiconify()
                    self._models_win.win.lift()
                    self._models_win.win.focus_force()
                    return
            except tk.TclError:
                pass
            self._models_win = None
        self._models_win = ModelsViewer(self.root, self.t, self)


def main():
    root = tk.Tk()
    SimpleChat(root)
    root.mainloop()


if __name__ == "__main__":
    main()
