# Code Review & Refactoring Proposals — sokoban-tree-based

**Reviewer:** Claude Code
**Date:** 2026-02-26
**Scope:** Full codebase review (all project Python files, templates, and configuration)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Critical Bugs](#2-critical-bugs)
3. [Architecture & Design Issues](#3-architecture--design-issues)
4. [File-by-File Review](#4-file-by-file-review)
5. [Code Quality & Maintainability](#5-code-quality--maintainability)
6. [Security Concerns](#6-security-concerns)
7. [Performance](#7-performance)
8. [Testing](#8-testing)
9. [Refactoring Roadmap](#9-refactoring-roadmap)

---

## 1. Executive Summary

The project implements an AI-powered Sokoban puzzle solver using LangGraph with an LLM reflection loop. The core idea is sound, but the codebase suffers from **critical bugs in game logic**, **dangerous global mutable state**, **dead/duplicate code**, and **no test coverage**. The most impactful improvements are fixing the inverted box-push logic, eliminating global state, and removing duplicated code.

### Severity Legend

| Severity | Meaning |
|----------|---------|
| **CRITICAL** | Bug that causes incorrect behavior at runtime |
| **HIGH** | Significant design flaw or potential data corruption |
| **MEDIUM** | Code quality issue that hinders maintainability |
| **LOW** | Style, naming, or minor improvement |

---

## 2. Critical Bugs

### 2.1 CRITICAL — Inverted box-push logic in `makePlayerMove()`

**Files:** `agent/agent.py:196-211`, `agent/agent copy.py:214-229`

The box-pushing logic is **inverted**. When a box is at the target position, the code checks `isBlocked(box_push_x, box_push_y)` and pushes the box **if the space IS blocked**, and rejects the move **if the space is NOT blocked**. This is backwards.

```python
# CURRENT (WRONG):
if isBlocked(box_push_x, box_push_y):
    # Push the box  <-- pushes when BLOCKED (should reject)
    ...
else:
    return "Cannot move, because the box's new position ... is blocked"
    # <-- rejects when NOT blocked (should push)
```

**Fix:**
```python
# CORRECT:
if isBlocked(box_push_x, box_push_y):
    return "Cannot move, because the box's new position is blocked"
else:
    # Push the box
    box_index = boxes.index((target_x, target_y))
    boxes[box_index] = (box_push_x, box_push_y)
    sokobanGame.gameStateObj['boxes'] = boxes
    sokobanGame.gameStateObj['player'] = (target_x, target_y)
    ...
```

### 2.2 CRITICAL — Direction parsing matches too aggressively

**Files:** `agent/agent.py:173-184`, `agent/agent.py:283-305`

The direction parsing uses broad substring checks. For example, `"U" in direction` will match any line containing the letter "U" (e.g., "push", "solution", "Updated"). This causes spurious moves on nearly every line of LLM output.

In `reflection_processing_moves()`, the problem is worse because it uses `if` (not `elif`), so a single line like `"Move Up from (3,3) to (2,3)"` triggers **both** U and D (because "Up" contains "U" and the coordinates contain "D"-adjacent text).

**Fix:** Use strict matching with word boundaries or exact token matching:
```python
DIRECTION_MAP = {
    'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0)
}

def parse_direction(raw: str) -> tuple[int, int] | None:
    token = raw.strip().upper()
    if token in DIRECTION_MAP:
        return DIRECTION_MAP[token]
    # For tagged format: <U>, <D>, etc.
    for key, offset in DIRECTION_MAP.items():
        if f"<{key}>" == token:
            return offset
    return None
```

In `reflection_processing_moves()`, change all `if` to `elif` at minimum, and preferably extract the direction tag from the line with a regex like `r'<([UDLR])>'`.

### 2.3 CRITICAL — `isWall()` returns `False` for out-of-bounds positions

**File:** `agent/agent.py:131-138`

```python
def isWall(x, y):
    if x < 0 or x >= len(sokobanGame.mapObj) or ...:
        return False  # BUG: out-of-bounds should be treated as blocked
```

A position outside the map boundaries is treated as "not a wall", allowing the player or boxes to move out of bounds. This should return `True` (or the caller `isBlocked` should be used instead, which does handle OOB correctly).

### 2.4 HIGH — `map` and `mapObj` are the same reference (aliasing bug)

**File:** `agent/agent.py:15-18`

```python
map = sokobanGame.levelObj['mapObj']
starting_map = sokobanGame.levelObj['mapObj']  # same reference!
```

Both `map` and `starting_map` point to the same nested list. Writing to `map[i][j]` also modifies `starting_map[i][j]` and the original `levelObj['mapObj']`. This corrupts the original map data permanently.

**Fix:** Deep-copy the map:
```python
import copy
map_copy = copy.deepcopy(sokobanGame.levelObj['mapObj'])
```

### 2.5 HIGH — `map` shadows the Python built-in

**File:** `agent/agent.py:15`

Using `map` as a variable name shadows Python's built-in `map()` function. Use `game_map` or `grid` instead.

---

## 3. Architecture & Design Issues

### 3.1 HIGH — Global mutable state (`global_sokobanGame`)

**File:** `sokoban/sokoban_tools.py:83-84`

```python
global global_sokobanGame
global_sokobanGame = SokobanRules(os.path.join(os.getcwd(), "dataset/test/1_4.txt"))
```

This global instance is imported and mutated across the entire codebase (`agent/agent.py`, `nodes/nodes.py`, `graph/graph.py`). This causes:

1. **Race conditions** in concurrent Gradio sessions — all users share the same game state.
2. **Stale state** across games — the global is initialized at import time with `1_4.txt` and is never properly reset.
3. **Untestable code** — functions depend on hidden global state.

**Fix:** Pass `SokobanRules` instances explicitly through the `SokobanState` or as function parameters. The `executor_node` already does this correctly — extend this pattern everywhere.

### 3.2 HIGH — Duplicated code: `agent/agent.py` vs `agent/agent copy.py`

The file `agent/agent copy.py` is a 475-line near-duplicate of `agent/agent.py` with minor differences (e.g., the old copy has an infinite `while not LEVEL_COMPLETED` loop with no iteration guard, and imports `torch`, `create_agent`, etc. that are unused in the main file). It also imports a non-existent `.output_entity` module.

**Fix:** Delete `agent/agent copy.py`. It is dead code.

### 3.3 MEDIUM — `main.py` is a dead stub

`main.py` only prints `"Hello from sokoban-tree-based!"`. The actual entry point is `sokoban.py`. This is confusing.

**Fix:** Either make `main.py` the real entry point (import and launch the Gradio app) or delete it.

### 3.4 MEDIUM — Hardcoded model name and test file in `run_superstep()`

**File:** `graph/graph.py:87`

```python
state = initiate_state(model_name="qwen3:latest",
                       test_file=os.path.join(os.getcwd(), "dataset/test/1_4.txt"))
```

This ignores the file the user actually uploaded via the Gradio UI. The `file_setup()` method stores the file in `sokobanGame` global but `run_superstep()` always uses `1_4.txt`.

**Fix:** Store the uploaded file path in `self` and pass it to `initiate_state()`.

### 3.5 MEDIUM — `util.py` is unused

`util.py` defines CSS, JS, and a `Color` enum but none are imported anywhere in the project.

---

## 4. File-by-File Review

### `sokoban/sokoban_tools.py` — SokobanRules

| Line | Severity | Issue |
|------|----------|-------|
| 24 | MEDIUM | `str(data_file).rsplit("/", 1)[1]` crashes if path has no `/` (e.g., Windows or bare filename). Use `pathlib.Path(data_file).name`. |
| 30 | LOW | Only reads lines starting with `#`. If the first line is blank or a comment, parsing breaks silently. |
| 45-60 | MEDIUM | Does not handle `+` (player on goal) or `*` (box on goal) during initial parsing for player position and goals. The `*` case adds to both `goals` and `boxes` (correct), but `+` is not handled at all. |
| 65 | LOW | Assertion error messages use `%s` format but don't actually format: `'Level is impossible to solve. It has %s goals but only %s stars.'` — the `%s` are never substituted. |
| 68 | LOW | Redundant f-string nesting: `f"{",".join([str(f"({box[0]}, {box[1]})") for box in boxes])}"` |
| 83-84 | HIGH | Global instance created at import time — see Section 3.1. |

### `agent/agent.py` — SokobanAgentic & makePlayerMove

| Line | Severity | Issue |
|------|----------|-------|
| 14-45 | HIGH | `convert_current_state_to_map()` — aliasing bug (see 2.4), shadows `map` built-in (see 2.5). |
| 131-138 | CRITICAL | `isWall()` OOB returns False (see 2.3). |
| 173-184 | CRITICAL | Direction parsing too broad (see 2.2). |
| 196-211 | CRITICAL | Inverted box-push logic (see 2.1). |
| 239 | MEDIUM | `MAX_ITERATIONS = 5` is hardcoded inside the agent method, not configurable. |
| 253 | LOW | llama3 tool-binding check: `if "llama3" not in model_name` — fragile string check. Should use a config flag or capability check. |
| 278-308 | CRITICAL | `reflection_processing_moves()` uses `if` instead of `elif` — a single line can trigger multiple moves (see 2.2). |
| 333 | LOW | `raise f"Invalid_Tool"` — raises a string, not an exception. Should be `raise ValueError("Invalid tool: ...")`. |

### `agent/instructions.py` — Prompt Templates

| Line | Severity | Issue |
|------|----------|-------|
| 76 | MEDIUM | Contains LLaMA-specific tokens (`<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>`) hardcoded in the ReAct template. These are model-specific and will confuse non-LLaMA models. |
| 160-247 | MEDIUM | `___sokoban_assist_template()` (triple-underscore prefix) is dead code — never called anywhere. |
| 81-158 | LOW | `sokoban_reflection_template()` repeats the entire Sokoban rules block that's also in `sokoban_system_template()`. Extract to a shared constant. |

### `nodes/nodes.py` — Workflow Nodes

| Line | Severity | Issue |
|------|----------|-------|
| 27 | HIGH | `state['current_iteration'] = state['current_iteration'] + 1` — directly mutating the state dict inside a LangGraph node. LangGraph expects nodes to return new state values, not mutate in place. This works with `InMemorySaver` but may break with other checkpointers. |
| 30 | MEDIUM | `state['visited_map_state'] = []` — resets visited states every iteration, losing history from previous attempts. |
| 72 | MEDIUM | `executor_node` creates a new `SokobanRules(state['test_file'])` each time — good pattern, but the `move_node` still uses the global `sokobanGame`. Inconsistent. |
| 88-91 | MEDIUM | Cycle detection logic truncates `total_moves` using list index, but uses string slicing. If the map state at index `idx` corresponds to a different number of characters in `total_moves`, the truncation is incorrect. |

### `edges/edges.py` — Routing Logic

| Line | Severity | Issue |
|------|----------|-------|
| 20-30 | LOW | No handling for unexpected status values. If `state['status']` is anything other than the 4 known values, the function returns `None`, which will crash LangGraph. Add a default case. |

### `graph/graph.py` — SokobanChat & Workflow

| Line | Severity | Issue |
|------|----------|-------|
| 49 | LOW | `graph.get_graph().draw_mermaid_png()` runs every time the graph is built. This has a dependency on external rendering (Graphviz/mermaid) and will fail silently or throw errors in environments without it. Guard with try/except or make optional. |
| 63 | MEDIUM | `setup()` creates a new `SokobanRules` but assigns it to a local variable, not `self`. The global `sokobanGame` imported at the top of the file is what actually gets used. |
| 87 | HIGH | Hardcoded test file path — see Section 3.4. |
| 92 | MEDIUM | `"\n ------ \n".join(result['visited_map_state'])` — if `visited_map_state` is empty (e.g., all moves were invalid), this produces an empty string with no feedback to the user. |

### `sokoban.py` — Gradio UI

| Line | Severity | Issue |
|------|----------|-------|
| 33-36 | MEDIUM | `free_resources()` calls `sokobanChat.setup()` synchronously but `setup()` is `async`. This will create an unawaited coroutine. |
| 38 | LOW | Gradio 6.0 deprecation warning — `theme` should be passed to `launch()`, not `Blocks()`. |
| 61 | LOW | `share=True` in production can be a security risk. Make configurable via environment variable. |

---

## 5. Code Quality & Maintainability

### 5.1 Naming Conventions

- **Inconsistent casing:** `playerx`/`playery` (no separator), `target_x`/`target_y` (snake_case), `xOffset`/`yOffset` (camelCase), `box_push_x` (snake_case) — all in the same function. Standardize on snake_case per PEP 8.
- **Misleading names:** `sokoban_game` is used for both the game state string and the game object in different contexts.
- **`SokobanRules`** doesn't just contain rules — it's the entire game engine (state, map, parsing). Consider renaming to `SokobanGame` or `SokobanEngine`.

### 5.2 Code Duplication

| Duplicated Code | Locations | Lines |
|-----------------|-----------|-------|
| `makePlayerMove()` (full function) | `agent/agent.py`, `agent/agent copy.py` | ~85 lines x2 |
| `convert_current_state_to_map()` | `agent/agent.py`, `agent/agent copy.py` | ~30 lines x2 |
| Direction parsing (`if "<U>" in ...`) | `agent/agent.py` (3 places), `agent/agent copy.py` (3 places) | ~20 lines x6 |
| `reflection_processing_moves()` | `agent/agent.py`, `agent/agent copy.py` | ~30 lines x2 |
| Sokoban rules description | `agent/instructions.py` (4 templates) | ~30 lines x4 |
| `SokobanAgentic` class | `agent/agent.py`, `agent/agent copy.py` | ~100 lines x2 |

**Estimated duplicated code: ~500 lines out of ~1100 total project lines (~45%).**

### 5.3 Missing Type Hints

Most functions lack return type annotations and parameter types. The `SokobanState` TypedDict is well-defined but not consistently used (nodes accept `SokobanState` but return `{**state}` dicts).

### 5.4 Error Handling

- Broad `except Exception as e` blocks in every node silently swallow errors and return the unchanged state, making debugging very difficult.
- `raise f"Invalid_Tool"` raises a string, not an exception.
- No input validation on uploaded files in the Gradio UI.

---

## 6. Security Concerns

| Severity | Issue | Location |
|----------|-------|----------|
| MEDIUM | `share=True` exposes the app publicly with no authentication (`auth=None`). | `sokoban.py:61` |
| LOW | User-uploaded files are read without sanitization or size limits. A malformed file could crash the parser. | `sokoban/sokoban_tools.py:27-33` |
| LOW | `load_dotenv(override=True)` can override system env vars. | `agent/agent.py:11` |

---

## 7. Performance

| Issue | Impact | Fix |
|-------|--------|-----|
| `isLevelFinished()` iterates all goals on every single move attempt (called before direction parsing). | Low impact for small puzzles, but wasteful. | Move the check to after a successful move only. |
| `(x, y) in boxes` does a linear scan of a list. | O(n) per check. | Use a `set` for box positions instead of a list. |
| `convert_current_state_to_map()` rebuilds the entire map string on every call. | Moderate — called multiple times per LLM iteration. | Cache or only rebuild when state changes. |
| `graph.get_graph().draw_mermaid_png()` generates an image file on every graph build. | Slow startup. | Move to a debug/CLI command, not runtime. |

---

## 8. Testing

**Current test coverage: 0%.** There are no unit tests, integration tests, or test runner configuration.

### Recommended Test Plan

1. **Unit tests for `makePlayerMove()`:** Test all directions, wall collisions, box pushing, box-into-wall, level completion, invalid input.
2. **Unit tests for `SokobanRules.read_map()`:** Test various puzzle formats, edge cases (empty map, no player, no goals).
3. **Unit tests for `reflection_processing_moves()`:** Test direction parsing from various LLM output formats.
4. **Integration test for the full workflow:** Load a puzzle with a known solution, run it through the executor, verify success.
5. **Test for `convert_current_state_to_map()`:** Verify map rendering matches expected output after specific moves.

---

## 9. Refactoring Roadmap

### Phase 1 — Fix Critical Bugs (Immediate)

1. **Invert the box-push condition** in `makePlayerMove()` (both files, then delete the copy).
2. **Fix direction parsing** — use `elif` and strict matching.
3. **Fix `isWall()` OOB handling** — return `True` for out-of-bounds.
4. **Fix map aliasing** — deep-copy in `convert_current_state_to_map()`.
5. **Delete `agent/agent copy.py`** — dead code.

### Phase 2 — Eliminate Global State (High Priority)

1. Move `SokobanRules` instance creation into `SokobanState` or pass it explicitly.
2. Remove the `global global_sokobanGame` from `sokoban/sokoban_tools.py`.
3. Update all functions (`makePlayerMove`, `convert_current_state_to_map`, etc.) to accept the game instance as a required parameter (some already support this via default args).
4. Fix `SokobanChat.run_superstep()` to use the uploaded file path instead of hardcoded `1_4.txt`.

### Phase 3 — Code Cleanup (Medium Priority)

1. Delete dead code: `main.py` stub, `util.py` (unused), `___sokoban_assist_template()`.
2. Extract shared Sokoban rules text in prompts to a constant.
3. Standardize naming to PEP 8 snake_case.
4. Extract direction parsing into a single reusable function.
5. Fix `free_resources()` to properly await async `setup()` or make cleanup synchronous.
6. Add a default/fallback in `route_after_executor_node()` for unexpected states.
7. Fix `raise f"Invalid_Tool"` to raise a proper exception.

### Phase 4 — Add Testing (Medium Priority)

1. Set up `pytest` in `pyproject.toml`.
2. Write unit tests for game logic (`makePlayerMove`, `SokobanRules`, direction parsing).
3. Write integration test with a known-solvable puzzle.
4. Add CI configuration (GitHub Actions).

### Phase 5 — Architecture Improvements (Lower Priority)

1. Use `set` instead of `list` for box positions.
2. Make model name and max iterations configurable via environment variables or UI.
3. Guard `draw_mermaid_png()` with try/except or a debug flag.
4. Add file validation and size limits for uploaded puzzles.
5. Make `share=True` configurable via env var.
6. Consider separating the game engine (`SokobanRules` + move logic) from the agent layer — the move logic currently lives in `agent/agent.py` but is pure game logic with no AI dependency.

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Critical bugs | 4 |
| High severity issues | 6 |
| Medium severity issues | 13 |
| Low severity issues | 9 |
| Dead/duplicate files | 2 (`agent/agent copy.py`, `main.py`) |
| Estimated duplicated code | ~45% |
| Test coverage | 0% |
| Total project Python lines (excl. .venv) | ~1,100 |
