"""
Sanity Checker for Inception-of-Context (IoC) Part 3: Generation, Application, and Validation Loop.
Enforces the 6 mandatory non-negotiable sanity rules from Subject VI.3.
These are hard refusals, not warnings. Any violation immediately rejects the patch
before touching the disk and provides explicit feedback for the retry loop.
"""

import ast
import builtins
import difflib
import io
import os
import re
import sys
import tokenize
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p1.indexer import safe_join, should_ignore_path  # noqa: E402
from p1.markers import ALL_MARKERS  # noqa: E402

# Derived from the single source of truth in p1/markers.py, which BOTH prompt
# builders format their section headers from. Hardcoding a list here is what let
# the rule drift until it no longer matched anything the patch prompt emits.
RETRIEVAL_MARKERS: List[str] = list(ALL_MARKERS)

VALID_OPERATIONS: Set[str] = {"create", "modify", "delete"}

# The file that holds the validation command (Subject VI.3). It is the judge of
# every attempt, so no patch may rewrite it.
VALIDATION_CONFIG_NAME = "ioc.config.yml"

# Subject VI.3 hard-refusal rules, plus rule 0 (the syntax gate) which must pass
# before rule 4 can be evaluated at all.
RULE_LABELS: Dict[str, str] = {
    "0": "Python Syntax Parse",
    "1": "Retrieval Markers Guard",
    "2": "No Overwrite on Create",
    "3": "Non-Empty Content",
    "4": "AST No-Stub Body",
    "5": "File Shrinkage <= 60%",
    "6": "Touches <= 3 Files",
}

# Maximum files a single patch may delete. Rule 5's shrinkage ceiling does not
# apply to deletions, so without this a patch could erase three whole files.
MAX_DELETIONS = 1

# Listed with a file's definitions: without this block `python3 main.py` runs
# nothing and exits 0, so a patch that drops it passes a validation command
# that no longer checks anything.
MAIN_GUARD = 'if __name__ == "__main__":'

# Intents that ask for code to disappear without naming every piece of it
# ("remove the unused helpers", "merge the two formatters").
_REMOVAL_RE = re.compile(
    r"\b(remov|delet|drop|get rid|clean\s*up|prun|eliminat|merg|consolidat|inlin|refactor|"
    r"simplif|rewrit|restructur|split|combin|dedup|replac|mov)\w*",
    re.IGNORECASE,
)
_MAIN_GUARD_RE = re.compile(r"__main__|entry[\s-]?point|main guard", re.IGNORECASE)

# "rename [the method] add[()] [method] to plus", dotted names allowed.
_KIND = r"(?:method|function|class|attribute|variable|constant|field)"
_RENAME_RE = re.compile(
    rf"\brename\s+(?:the\s+)?(?:{_KIND}\s+)?`?([A-Za-z_][\w.]*)(?:\(\))?`?\s+"
    rf"(?:{_KIND}\s+)?(?:to|as|into)\s+(?:the\s+)?(?:{_KIND}\s+)?`?([A-Za-z_][\w.]*)",
    re.IGNORECASE,
)
# Anything else the intent asks for besides the rename: then new uses of the
# new name are expected, and the rename check stays out of the way.
_OTHER_WORK_RE = re.compile(
    r"\b(add|create|implement|write|introduce|insert|append|remov|delet|fix|change|replac|"
    r"mov|extract|convert|make|refactor|document|log|validat|test|print|return|rais|handl)\w*"
    # "use plus in calculate_tax" is more work; "update every use" is the rename itself.
    r"|(?<!every )(?<!each )(?<!all )(?<!its )(?<!their )(?<!the )\b(?:use|call)\b",
    re.IGNORECASE,
)


@dataclass
class SanityCheckResult:
    """Represents the outcome of patch sanity validation."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    # Per-rule verdict so the dashboard can show which specific rules failed
    # instead of colouring every pill from one aggregate boolean.
    rule_status: Dict[str, bool] = field(
        default_factory=lambda: {key: True for key in RULE_LABELS}
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "rule_status": self.rule_status,
            "rule_labels": RULE_LABELS,
        }


def coerce_text(value: Any) -> str:
    """Normalize a JSON field the model may have emitted as null or a non-string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def describe_syntax_error(content: str, err: SyntaxError) -> str:
    """Render the offending source line so the retry prompt gets actionable feedback."""
    lines = content.splitlines()
    if err.lineno and 1 <= err.lineno <= len(lines):
        return repr(lines[err.lineno - 1].rstrip())
    return "<line unavailable>"


def body_without_docstring(node: Any) -> List[Any]:
    """The statements of a function or class body, its docstring left out."""
    statements = list(node.body)
    if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
        if isinstance(statements[0].value.value, str):
            statements = statements[1:]
    return statements


def is_stub_function(node: Any) -> bool:
    """
    Check if a Python AST function or async function body is only a stub.
    A stub is a body containing only 'pass', '...', 'return None', or docstrings followed by them.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False

    body = node.body
    if not body:
        return True

    # Ignore leading docstring if present
    statements = body_without_docstring(node)

    # Empty body after removing docstring
    if not statements:
        return True

    # If body has more than one functional statement, it's not a stub
    if len(statements) > 1:
        return False

    stmt = statements[0]

    # Check 1: 'pass'
    if isinstance(stmt, ast.Pass):
        return True

    # Check 2: '...' (Ellipsis)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        if stmt.value.value is ... or stmt.value.value == "...":
            return True

    # Check 3: 'return' or 'return None'
    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return True
        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            return True

    # Check 4: 'raise NotImplementedError' / 'raise NotImplementedError(...)' -
    # the other spelling of "not written yet" in Python.
    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True

    return False


def stub_functions(tree: ast.AST) -> List[Any]:
    """Every function or async function in the tree whose body is only a stub."""
    return [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_stub_function(node)
    ]


# What a comment or docstring says of code that is not written yet. A function
# that only returns a constant can be real code (`def supports_negative(self):
# return True`); marked like this it is "not implemented" in another spelling.
_PLACEHOLDER_RE = re.compile(
    r"\b(?:placeholder|stub(?:bed)?|todo|fixme|dummy|for now|not (?:yet )?implemented|"
    r"implement (?:me|this|later)|(?:actual|real) (?:logic|implementation))\b",
    re.IGNORECASE,
)


def returned_literal(node: Any) -> Tuple[bool, Any]:
    """(True, value) when a function's whole body, docstring aside, is `return <literal>`."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False, None
    statements = body_without_docstring(node)
    if len(statements) != 1 or not isinstance(statements[0], ast.Return) or statements[0].value is None:
        return False, None
    try:
        return True, ast.literal_eval(statements[0].value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def comments_by_row(content: str) -> Dict[int, str]:
    """{row: text} of every comment in the source."""
    found: Dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(content).readline):
            if tok.type == tokenize.COMMENT:
                found[tok.start[0]] = tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return found


def placeholder_note(node: Any, comments: Dict[int, str]) -> Optional[str]:
    """
    The note that marks a function returning only a constant as a
    placeholder, else None. Seen live, on attempt 3 of a run that then
    reported GREEN: `return True  # Placeholder for actual logic`.
    """
    if not returned_literal(node)[0]:
        return None
    rows = range(node.lineno, (node.end_lineno or node.lineno) + 1)
    notes = [comments[row] for row in rows if row in comments]
    notes.append(ast.get_docstring(node, clean=False) or "")
    return next((note.strip() for note in notes if _PLACEHOLDER_RE.search(note)), None)


def plain_text(text: str) -> str:
    """Feedback goes back into the prompt of a model that must not emit raw double quotes or backticks."""
    return text.replace('"', "'").replace("`", "'")


# ({module function: value}, {method or property: (value, is_property)}, class names)
_Consts = Tuple[Dict[str, Any], Dict[str, Any], Set[str]]
# (known, value, why)
_Static = Tuple[bool, Any, str]


def _constant_callables(tree: ast.AST) -> _Consts:
    """
    The functions whose every definition in the module returns the same
    literal, as _Consts. A name defined once with real logic (a subclass
    override) is not constant.
    """
    module_level = {id(n) for n in getattr(tree, "body", [])}
    values: Dict[Tuple[bool, str], List[Tuple[bool, Any, bool]]] = {}
    classes: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Calling an async function yields a coroutine, whatever it returns.
            literal, value = returned_literal(node) if isinstance(node, ast.FunctionDef) else (False, None)
            is_property = any(isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list)
            values.setdefault((id(node) in module_level, node.name), []).append((literal, value, is_property))
    functions: Dict[str, Any] = {}
    methods: Dict[str, Any] = {}
    for (top, name), found in values.items():
        first = found[0]
        if all(lit for lit, _, _ in found) and all((v, p) == first[1:] for _, v, p in found):
            if top and not first[2]:
                functions[name] = first[1]
            elif not top:
                methods[name] = (first[1], first[2])
    return functions, methods, classes


def _static_value(expr: Any, consts: _Consts) -> _Static:
    """
    (known, value, why) of a condition whose value the code itself fixes: a
    literal, or a call of a function of this module that only returns one
    (`self._precision_enabled()`), through not / and / or / == / is.
    Module-level names (`DEBUG = False`) are configuration and stay unknown.
    """
    unknown: Tuple[bool, Any, str] = (False, None, "")
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        known, value, why = _static_value(expr.operand, consts)
        return (True, not value, why) if known else unknown
    if isinstance(expr, ast.BoolOp):
        # Only the truth value is known: `x or True` is x or True.
        parts = [_static_value(v, consts) for v in expr.values]
        deciding = isinstance(expr.op, ast.Or)  # True decides an `or`, False an `and`
        for known, value, why in parts:
            if known and bool(value) == deciding:
                return True, deciding, why
        if all(known for known, _, _ in parts):
            return True, not deciding, next((why for _, _, why in parts if why), "")
        return unknown
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        left, right = _static_atom(expr.left, consts), _static_atom(expr.comparators[0], consts)
        op = expr.ops[0]
        if not (left[0] and right[0]):
            return unknown
        why = left[2] or right[2]
        if isinstance(op, (ast.Eq, ast.NotEq)):
            return True, (left[1] == right[1]) == isinstance(op, ast.Eq), why
        singletons = (None, True, False)
        if isinstance(op, (ast.Is, ast.IsNot)) and left[1] in singletons and right[1] in singletons:
            same = any(left[1] is s and right[1] is s for s in singletons)
            return True, same == isinstance(op, ast.Is), why
        return unknown
    return _static_atom(expr, consts)


def _static_atom(expr: Any, consts: _Consts) -> _Static:
    """(known, value, why) of a literal, or of a call of a function of this module that only returns one."""
    functions, methods, classes = consts
    unknown: Tuple[bool, Any, str] = (False, None, "")
    if isinstance(expr, ast.Constant):
        return True, expr.value, ""
    called = isinstance(expr, ast.Call)
    target = expr.func if called else expr
    if called and isinstance(target, ast.Name) and target.id in functions:
        return True, functions[target.id], f"{target.id}() always returns {functions[target.id]!r}"
    if (
        isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
        and (target.value.id in ("self", "cls") or target.value.id in classes)
        and target.attr in methods
    ):
        value, is_property = methods[target.attr]
        if called and not is_property:
            return True, value, f"{target.attr}() always returns {value!r}"
        if not called and is_property and target.value.id == "self":
            return True, value, f"the property {target.attr} always returns {value!r}"
    return unknown


def _exception_name(expr: Any) -> Optional[str]:
    target = expr.func if isinstance(expr, ast.Call) else expr
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _catches(handler: Any, raised: str) -> bool:
    """Whether this except clause catches an exception of this name."""
    if handler.type is None:
        return True
    raised_cls = getattr(builtins, raised, None)
    # SystemExit, KeyboardInterrupt, GeneratorExit: not caught by `except Exception`.
    beyond_exception = isinstance(raised_cls, type) and not issubclass(raised_cls, Exception)
    for caught in handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]:
        name = _exception_name(caught)
        if name is None:
            continue
        if name in (raised, "BaseException"):
            return True
        if name == "Exception" and not beyond_exception:
            return True
        caught_cls = getattr(builtins, name, None)
        if (
            isinstance(raised_cls, type) and isinstance(caught_cls, type)
            and issubclass(caught_cls, BaseException) and issubclass(raised_cls, caught_cls)
        ):
            return True
    return False


def dead_code(tree: ast.AST) -> List[Tuple[Any, str]]:
    """
    (key, description) of each piece of code that can never run, as the code
    itself decides: a branch behind a condition with a fixed value, a
    statement after return / raise / break / continue, a raise caught by an
    except of its own try that does not raise again. Validation cannot see
    any of them: asked to make add raise an error that main.py does not
    survive, the model put the raise behind `if not self._precision_enabled():`
    with a helper that returns True, and the run reported GREEN.
    """
    consts = _constant_callables(tree)
    found: List[Tuple[Any, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)):
            known, value, why = _static_value(node.test, consts)
            because = f", because {why}" if why else ""
            kind = "if" if isinstance(node, ast.If) else "while"
            if known and not value:
                found.append((
                    ("never", ast.dump(node), why),
                    f"line {node.body[0].lineno}: the {kind} body never runs, its condition on line "
                    f"{node.lineno} is always false{because}",
                ))
            elif known and isinstance(node, ast.If) and node.orelse:
                found.append((
                    ("always", ast.dump(node), why),
                    f"line {node.orelse[0].lineno}: the else branch never runs, the condition on line "
                    f"{node.lineno} is always true{because}",
                ))
        for name in ("body", "orelse", "finalbody"):
            block = getattr(node, name, None)
            if not isinstance(block, list):
                continue
            for stmt, after in zip(block, block[1:]):
                if not isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                    continue
                # `return` then `yield`: the idiom for an empty generator.
                if not (isinstance(after, ast.Expr) and isinstance(after.value, (ast.Yield, ast.YieldFrom))):
                    word = type(stmt).__name__.lower()
                    found.append((
                        ("unreachable", ast.dump(stmt), ast.dump(after)),
                        f"line {after.lineno} is never reached, it follows the {word} on line {stmt.lineno}",
                    ))
                break
        if isinstance(node, ast.Try):
            for stmt in node.body:
                raised = _exception_name(stmt.exc) if isinstance(stmt, ast.Raise) and stmt.exc else None
                handler = next((h for h in node.handlers if raised and _catches(h, raised)), None)
                if handler is not None and not any(isinstance(n, ast.Raise) for n in ast.walk(handler)):
                    found.append((
                        ("caught", ast.dump(node)),
                        f"line {stmt.lineno}: the {raised} never leaves the function, the except on "
                        f"line {handler.lineno} of the same try catches it",
                    ))
    return found


def added_dead_code(old_tree: Optional[ast.AST], new_tree: ast.AST) -> List[str]:
    """dead_code() of the patched file that the current file does not have already."""
    before = {key for key, _ in dead_code(old_tree)} if old_tree is not None else set()
    return [text for key, text in dead_code(new_tree) if key not in before]


def _is_main_guard(node: ast.AST) -> bool:
    """`if __name__ == "__main__":`, either operand order."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    sides = [test.left, test.comparators[0]]
    names = [s.id for s in sides if isinstance(s, ast.Name)]
    values = [s.value for s in sides if isinstance(s, ast.Constant)]
    return names == ["__name__"] and values == ["__main__"]


def definitions(tree: ast.AST) -> Dict[str, str]:
    """
    Module-level functions and classes, their methods, and the __main__
    block, as {qualified name: label for feedback}.
    """
    found: Dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = f"{node.name}()"
        elif isinstance(node, ast.ClassDef):
            found[node.name] = f"class {node.name}"
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"{node.name}.{child.name}"] = f"{node.name}.{child.name}()"
        elif _is_main_guard(node):
            # No raw double quote: this text goes back into the model's prompt.
            found[MAIN_GUARD] = "the if __name__ == '__main__' block"
    return found


def _named(name: str, intent: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", intent, re.IGNORECASE) is not None


def unrequested_removals(old_tree: ast.AST, new_tree: ast.AST, intent: str) -> List[str]:
    """
    Definitions the patch removes although the intent does not ask for it.

    A removal is requested when the intent names the definition (or, for a
    method, its removed class), or asks for removals in general - except
    for the __main__ block, which only goes when the intent says so.
    """
    before = definitions(old_tree)
    after = definitions(new_tree)
    removal_asked = _REMOVAL_RE.search(intent) is not None
    missing: List[str] = []
    for qualified, label in before.items():
        if qualified in after:
            continue
        if qualified == MAIN_GUARD:
            if not _MAIN_GUARD_RE.search(intent):
                missing.append(label)
            continue
        owner, _, member = qualified.rpartition(".")
        requested = _named(member, intent) or (
            bool(owner) and owner not in after and _named(owner, intent)
        )
        if not requested and not removal_asked:
            missing.append(label)
    return missing


def _functions(tree: ast.AST) -> Dict[str, Any]:
    """{qualified name: node} of every function and method, nested ones included."""
    found: Dict[str, Any] = {}

    def visit(body: List[Any], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not isinstance(node, ast.ClassDef):
                    found.setdefault(prefix + node.name, node)
                visit(node.body, f"{prefix}{node.name}.")

    visit(getattr(tree, "body", []), "")
    return found


def _raise_paths(body: List[Any], path: Tuple[Any, ...], found: Dict[Any, Tuple[int, str]]) -> None:
    """Each raise of a function body, keyed by the raise and the conditions that lead to it."""
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # a nested definition has checks of its own
        if isinstance(stmt, ast.Raise):
            found.setdefault((path, ast.dump(stmt)), (stmt.lineno, ast.unparse(stmt)))
        elif isinstance(stmt, (ast.If, ast.While)):
            test = ast.dump(stmt.test)
            _raise_paths(stmt.body, path + ((test, True),), found)
            _raise_paths(stmt.orelse, path + ((test, False),), found)
        else:  # for, with, try, match: the same conditions hold inside
            for name in ("body", "orelse", "finalbody"):
                block = getattr(stmt, name, None)
                if isinstance(block, list):
                    _raise_paths(block, path, found)
            for holder in list(getattr(stmt, "handlers", [])) + list(getattr(stmt, "cases", [])):
                _raise_paths(holder.body, path, found)


def dropped_checks(old_tree: ast.AST, new_tree: ast.AST, intent: str) -> List[str]:
    """
    Error checks - a raise and the conditions that lead to it - lost by a
    function the intent does not name, unless it asks for removals. Asked
    to make add raise, the model also replaced divide's `if b == 0: raise
    ValueError(...)` with a check of its own, and the run went GREEN:
    validation never divides by zero, so it cannot see a check disappear.
    """
    if _REMOVAL_RE.search(intent):
        return []
    after_functions = _functions(new_tree)
    lost: List[str] = []
    for name, old in _functions(old_tree).items():
        new = after_functions.get(name)
        if new is None or _named(name.rpartition(".")[2], intent):
            continue  # a removed definition is unrequested_removals' business
        before: Dict[Any, Tuple[int, str]] = {}
        after: Dict[Any, Tuple[int, str]] = {}
        _raise_paths(old.body, (), before)
        _raise_paths(new.body, (), after)
        lost.extend(
            f"{name}() line {line}: {source}" for key, (line, source) in before.items() if key not in after
        )
    return lost


def parse_rename(intent: str) -> Optional[Tuple[str, str]]:
    """
    (old, new) for an intent that asks for a rename and nothing else
    ("rename the method add to plus and update every caller"), else None.
    File renames are not symbol renames.
    """
    match = _RENAME_RE.search(intent or "")
    if not match:
        return None
    raw_old, raw_new = (g.rstrip(".") for g in match.groups())
    if raw_old.lower().endswith(".py") or raw_new.lower().endswith(".py"):
        return None
    old, new = raw_old.split(".")[-1], raw_new.split(".")[-1]
    if not old or not new or old == new:
        return None
    if _OTHER_WORK_RE.search(intent[: match.start()] + " " + intent[match.end():]):
        return None
    return old, new


def defines(content: str, name: str) -> bool:
    """`name` is a function, class or assigned name in this source - a real symbol, not a word."""
    word = re.escape(name)
    pattern = rf"(?m)\b(?:def|class)\s+{word}\b|^\s*{word}\s*(?::[^=\n]+)?=(?!=)"
    return re.search(pattern, content) is not None


def name_lines(content: str, name: str) -> List[int]:
    """Line of every use of `name` as a code identifier - not in strings or comments."""
    try:
        return [
            tok.start[0] for tok in tokenize.generate_tokens(io.StringIO(content).readline)
            if tok.type == tokenize.NAME and tok.string == name
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


def split_lines(content: str) -> List[str]:
    """Lines with their endings, split on newline only - the way tokenize counts rows."""
    return [line for line in re.split(r"(?<=\n)", content) if line]


def name_columns(content: str, name: str) -> Dict[int, List[int]]:
    """{row: columns} of every use of `name` as a code identifier, 1-based rows."""
    columns: Dict[int, List[int]] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(content).readline):
            if tok.type == tokenize.NAME and tok.string == name:
                columns.setdefault(tok.start[0], []).append(tok.start[1])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {}
    return columns


def rename_at(line: str, columns: Tuple[int, ...], old: str, new: str) -> str:
    """`line` with the identifier `old` at these columns renamed; every other byte kept."""
    for col in sorted(columns, reverse=True):
        line = line[:col] + new + line[col + len(old):]
    return line


def rename_strays(old_content: str, new_content: str, old: str, new: str) -> Optional[List[str]]:
    """
    For a pure rename of `old` to `new`: None when every new use of `new`
    replaces a use of `old`, else the lines where `new` appeared on its own.
    The model once turned `calc.multiply(...)` into `calc.plus(...)` while
    renaming add to plus; validation cannot see that, the count can.
    """
    gained = len(name_lines(new_content, new)) - len(name_lines(old_content, new))
    replaced = len(name_lines(old_content, old)) - len(name_lines(new_content, old))
    if gained <= replaced:
        return None
    old_lines = [line.strip() for line in old_content.splitlines()]
    new_lines = [line.strip() for line in new_content.splitlines()]
    # The original of each rewritten line, where the diff pairs them up: told
    # only what was wrong, the model repeated the same edit; the line to keep
    # is actionable.
    original: Dict[int, str] = {}
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                original[j1 + k] = old_lines[i1 + k]
    before = set(old_lines)
    strays = []
    for no in sorted(set(name_lines(new_content, new))):
        line = new_lines[no - 1]
        if re.sub(r"\b" + re.escape(new) + r"\b", old, line) in before or line in before:
            continue
        was = original.get(no - 1)
        strays.append(f"line {no} must stay: {was} (the patch has: {line})" if was else f"line {no}: {line}")
    return strays


class SanityChecker:
    """
    Validates candidate patches against all Subject VI.3 constraints before any disk write.
    """

    def __init__(self, target_dir: str) -> None:
        self.target_dir = os.path.abspath(target_dir)

    def check(self, patch: Dict[str, Any], intent: str = "") -> SanityCheckResult:
        """
        Execute all 6 mandatory sanity checks on the patch payload.
        Returns SanityCheckResult(passed=True) only if ALL checks pass.

        `intent` is what the patch was asked to do: definitions it does not
        name may not disappear, and a rename may not spread the new name.

        Never raises: a malformed patch is a hard refusal with actionable
        feedback for the retry loop, not an exception that aborts the loop.
        """
        errors: List[str] = []
        rule_status: Dict[str, bool] = {key: True for key in RULE_LABELS}

        def fail(rule: Optional[str], message: str) -> None:
            errors.append(message)
            if rule is not None:
                rule_status[rule] = False

        def result() -> SanityCheckResult:
            return SanityCheckResult(
                passed=len(errors) == 0,
                errors=errors,
                rule_status=rule_status,
            )

        if not isinstance(patch, dict):
            fail(None, "Patch must be a valid JSON object")
            return result()

        # 0. Basic Schema Validation
        files = patch.get("files")
        if not isinstance(files, list):
            fail(None, "Patch must contain a 'files' array")
            return result()

        if len(files) == 0:
            fail(None, "Patch contains an empty 'files' list; nothing to apply")
            return result()

        # ---------------------------------------------------------------------
        # RULE 6: Touches more than three files at once
        # ---------------------------------------------------------------------
        if len(files) > 3:
            fail(
                "6",
                f"Rule 6 Violation: Patch touches {len(files)} files at once. "
                f"Maximum allowed is 3 files.",
            )

        deletion_count = 0
        seen_paths: Set[str] = set()
        entries_checked = 0
        entries_unchanged = 0
        # (path, current content, patched content) of every modified .py file
        # that parses, for the checks that compare the two.
        modified_py: List[Tuple[str, str, str]] = []

        for idx, file_entry in enumerate(files, 1):
            if not isinstance(file_entry, dict):
                fail(None, f"File entry #{idx} is not a valid object")
                continue

            # Coerce defensively: small models emit `"path": null` and
            # `"op": null`, and calling .strip() on those raised AttributeError
            # straight out of the loop instead of refusing the patch.
            rel_path = coerce_text(file_entry.get("path")).strip()
            op = coerce_text(file_entry.get("op")).strip().lower()
            raw_content = file_entry.get("content", "")

            if not rel_path:
                fail(None, f"File entry #{idx} is missing a usable 'path' string")
                continue

            if op not in VALID_OPERATIONS:
                fail(
                    None,
                    f"File entry '{rel_path}' has invalid op '{op}'. "
                    f"Must be one of: {sorted(list(VALID_OPERATIONS))}",
                )
                continue

            # Content must be a real string for create/modify. A null or a
            # nested object previously sailed through every rule and only blew
            # up later inside the applier's write().
            if op in ("create", "modify") and not isinstance(raw_content, str):
                fail(
                    "3",
                    f"Rule 3 Violation: 'content' for '{rel_path}' must be a JSON string, "
                    f"got {type(raw_content).__name__}. Emit the complete file as one string.",
                )
                continue
            content = raw_content if isinstance(raw_content, str) else ""

            # Prevent directory traversal outside target_dir. commonpath compares
            # whole components, so the sibling '../demo_app_secrets/x.py' - which
            # a startswith() prefix test accepts - is correctly refused.
            full_path = safe_join(self.target_dir, rel_path)
            if full_path is None:
                fail(
                    None,
                    f"Path traversal detected: '{rel_path}' resolves outside the target directory",
                )
                continue
            canonical = os.path.relpath(full_path, self.target_dir).replace("\\", "/")

            # One entry per file. "./calc.py" and "calc.py" are the same file;
            # two entries for it make the outcome depend on their order.
            if canonical in seen_paths:
                fail(
                    None,
                    f"File '{canonical}' appears more than once in the patch. "
                    f"Emit exactly one entry per file.",
                )
                continue
            seen_paths.add(canonical)

            if canonical == "." or os.path.isdir(full_path):
                fail(None, f"Path '{rel_path}' is a directory, not a file.")
                continue

            # Only files the indexer itself would consider source may be
            # touched: never .git/, a virtualenv, a cache or a binary.
            if should_ignore_path(canonical):
                fail(
                    None,
                    f"Path '{canonical}' is hidden, vendored, a cache or a binary file; "
                    f"patches may only touch the project's source files.",
                )
                continue

            # The validation command is the judge of every attempt. A patch
            # that rewrites it could pass by replacing the test with `true`.
            if canonical == VALIDATION_CONFIG_NAME:
                fail(
                    None,
                    f"'{VALIDATION_CONFIG_NAME}' holds the validation command and may not be "
                    f"changed by a patch.",
                )
                continue

            file_exists = os.path.isfile(full_path)
            entries_checked += 1

            # Read the current file once; several rules compare against it.
            existing_content: Optional[str] = None
            if file_exists and op in ("modify", "delete"):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        existing_content = f.read()
                except Exception as e:
                    fail(None, f"Could not read existing file '{canonical}': {e}")
                    continue

            if op == "modify" and existing_content is not None and content == existing_content:
                entries_unchanged += 1

            # -----------------------------------------------------------------
            # RULE 1: Leaks retrieval markers into its content
            # -----------------------------------------------------------------
            if op in ("create", "modify"):
                for marker in RETRIEVAL_MARKERS:
                    if marker in content:
                        fail(
                            "1",
                            f"Rule 1 Violation: Prompt marker '{marker}' leaked into "
                            f"file content of '{canonical}'",
                        )
                        break

            # -----------------------------------------------------------------
            # RULE 2: Tries to create a file that already exists
            # -----------------------------------------------------------------
            if op == "create":
                if file_exists:
                    fail(
                        "2",
                        f"Rule 2 Violation: Tried to create file '{canonical}', "
                        f"but it already exists on disk. Use op='modify' instead.",
                    )

            # -----------------------------------------------------------------
            # RULE 3: Replaces a non-empty file with "", "None", or "null"
            # -----------------------------------------------------------------
            if op == "modify":
                if not file_exists:
                    fail(
                        None,
                        f"Target file '{canonical}' does not exist on disk to modify. "
                        f"Use op='create' instead.",
                    )
                elif (existing_content or "").strip():
                    stripped_content = content.strip()
                    if (
                        stripped_content == ""
                        or stripped_content.lower() == "none"
                        or stripped_content.lower() == "null"
                    ):
                        fail(
                            "3",
                            f"Rule 3 Violation: Tried to replace non-empty file '{canonical}' "
                            f"with empty, 'None', or 'null' content.",
                        )

            # A create whose body is empty is just as useless as an emptied modify.
            if op == "create" and not content.strip():
                fail(
                    "3",
                    f"Rule 3 Violation: Tried to create '{canonical}' with empty content. "
                    f"Write the complete file body.",
                )

            # -----------------------------------------------------------------
            # Deletions: Rule 5's shrinkage ceiling cannot constrain them, so
            # cap how much one patch may erase outright.
            # -----------------------------------------------------------------
            if op == "delete":
                deletion_count += 1
                if not file_exists:
                    fail(
                        None,
                        f"Target file '{canonical}' does not exist on disk to delete.",
                    )
                if deletion_count > MAX_DELETIONS:
                    fail(
                        "5",
                        f"Rule 5 Violation: Patch deletes {deletion_count} files. "
                        f"At most {MAX_DELETIONS} file may be deleted per patch.",
                    )

            # -----------------------------------------------------------------
            # RULE 4: Defines a function whose body is only a stub (pass, ...,
            # return None, raise NotImplementedError).
            # Parsing the AST is also the syntax gate: a patch that does not even
            # compile must be refused here, before it is written to disk, so the
            # retry loop is fed a precise error instead of a generic py_compile dump.
            # -----------------------------------------------------------------
            if op in ("create", "modify") and canonical.endswith(".py"):
                try:
                    tree = ast.parse(content, filename=canonical)
                except (SyntaxError, ValueError) as e:
                    detail = (
                        f"{e.msg} at line {e.lineno}: {describe_syntax_error(content, e)}"
                        if isinstance(e, SyntaxError) else str(e)
                    )
                    fail(
                        "0",
                        f"Rule 0 Violation: Generated content for '{canonical}' is not valid "
                        f"Python. {detail}",
                    )
                    continue

                old_tree: Optional[ast.AST] = None
                if existing_content is not None:
                    try:
                        old_tree = ast.parse(existing_content)
                    except (SyntaxError, ValueError):
                        pass
                # A stub the file ALREADY had, unchanged, is not one this patch
                # defines; flagging it would make that file impossible to patch.
                preexisting: Set[str] = {
                    ast.dump(n) for n in ast.walk(old_tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                } if old_tree is not None else set()
                comments = comments_by_row(content)
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if ast.dump(node) in preexisting:
                        continue
                    if is_stub_function(node):
                        fail(
                            "4",
                            f"Rule 4 Violation: Function '{node.name}' in '{canonical}' "
                            f"(line {node.lineno}) is only a stub body "
                            f"(pass, ..., return None or raise NotImplementedError). "
                            f"Implementation is required.",
                        )
                        continue
                    note = placeholder_note(node, comments)
                    if note is not None:
                        fail(
                            "4",
                            f"Rule 4 Violation: Function '{node.name}' in '{canonical}' "
                            f"(line {node.lineno}) only returns a constant and is marked as a "
                            f"placeholder ({plain_text(note)}). Implementation is required.",
                        )

                # Code that can never run. Validation cannot see it, so the
                # requested behaviour put there went green without ever running.
                never_runs = added_dead_code(old_tree, tree)
                if never_runs:
                    fail(None, plain_text(
                        f"Patch adds code to '{canonical}' that never runs: {'; '.join(never_runs[:3])}. "
                        f"Write the requested behaviour so that it actually runs: not behind a condition "
                        f"that never holds, not after a return or raise, not inside a try whose except "
                        f"swallows it."
                    ))
                if op == "modify" and existing_content is not None:
                    modified_py.append((canonical, existing_content, content))

            # -----------------------------------------------------------------
            # RULE 5: Would shrink an existing file by more than 60%
            # -----------------------------------------------------------------
            if op == "modify" and existing_content is not None:
                old_len = len(existing_content.strip())
                new_len = len(content.strip())

                # If original file has substantial content, check shrinkage
                if old_len > 10:
                    # Example: 100 chars -> 35 chars (shrank by 65% > 60% -> reject!)
                    shrinkage = (old_len - new_len) / old_len
                    if shrinkage > 0.60:
                        pct = round(shrinkage * 100, 1)
                        fail(
                            "5",
                            f"Rule 5 Violation: Modification would shrink '{canonical}' by {pct}% "
                            f"(from {old_len} to {new_len} characters). "
                            f"Maximum allowed shrinkage is 60%.",
                        )

        # Unrequested removals. Rule 5 allows a file to lose up to 60 %, and
        # the model once "renamed add to plus" by also dropping main() and the
        # __main__ block from main.py (about 30 %): `python3 main.py` then ran
        # nothing, exited 0, and the loop reported GREEN.
        for canonical, before, after in modified_py:
            try:
                before_tree, after_tree = ast.parse(before), ast.parse(after)
            except (SyntaxError, ValueError):
                continue
            removed = unrequested_removals(before_tree, after_tree, intent)
            if removed:
                fail(
                    None,
                    f"Patch removes {', '.join(removed)} from '{canonical}', which the intent "
                    f"does not ask for. Keep every definition you were not asked to change - "
                    f"copy it verbatim - or name it in the intent to remove it.",
                )
            # The same for the error checks inside the definitions it keeps.
            lost = dropped_checks(before_tree, after_tree, intent)
            if lost:
                fail(None, plain_text(
                    f"Patch removes or changes error checks in '{canonical}' that the intent does not ask "
                    f"about: {'; '.join(lost[:3])}. Keep every check of a function you were not asked to "
                    f"change exactly as it is, condition included."
                ))

        # A rename replaces uses of the old name; it never puts the new name
        # where the old one was not.
        rename = parse_rename(intent)
        if rename and any(defines(before, rename[0]) for _, before, _ in modified_py):
            old, new = rename
            for canonical, before, after in modified_py:
                strays = rename_strays(before, after, old, new)
                if strays is None:
                    continue
                where = "; ".join(strays[:3]) if strays else f"'{new}' is used more often than '{old}' was"
                # Plain text, no backticks or double quotes: this goes back
                # into the prompt of a model that must not emit raw quotes.
                fail(
                    None,
                    f"Renaming '{old}' to '{new}' in '{canonical}': '{new}' was put where '{old}' "
                    f"was never used - {where}. A rename changes only the existing uses of "
                    f"'{old}'; leave every other line as it was.",
                )

        # A patch whose every entry rewrites a file with its current bytes does
        # nothing; letting it "pass" validation would report an unfulfilled
        # intent as a success.
        if not errors and entries_checked and entries_unchanged == entries_checked:
            fail(
                None,
                "Patch makes no change: every file it modifies already has exactly this "
                "content. Implement the requested change.",
            )

        return result()
