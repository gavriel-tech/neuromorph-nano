# Language fixtures

Small repositories written for these tests, one per language family, with hand-written expected
results. Nothing here is copied from other projects. `neuromorph-nano check languages` checks the
lexer snippets and every fixture with an `expected.json`; the questions run through the context pack
with `neuromorph-nano bench --mode plain --root tests/languages/<fixture>/repo --questions
tests/languages/<fixture>/questions.json`.

| Folder | What it holds |
|---|---|
| `lexer/<language>.txt` | Snippets and their hand token lists (`=== name`, source lines, `---`, tokens). Strings render as `$S`, regular expressions as `$R`; comments are dropped |
| `typescript/` | An npm workspace (two packages, `tsconfig.json` paths with `extends`, `.ts`, `.tsx`, `.css`) |
| `javascript/` | A command line tool (`.js`, `.mjs`, `.cjs`, `.jsx`, CommonJS and ES modules) |
| `python/` | A `src` layout package with relative imports, a test folder and scripts beside it |
| `c-cuda/` | A C library with a header in `include/`, CUDA kernels, macros and `extern "C"` |
| `powershell/` | Scripts with functions, a class, an enum, here-strings and block comments |
| `own-folder/` | Ten questions about the Nano's own interface and kernels (no expected definitions) |
| `gen/` | `plants.py` writes the planted files; `questions.py` writes each `questions.json` |

Each fixture's `repo/` also holds planted files the reader must not read: installed packages
(`node_modules/`), build output (`dist/`, `build/`, `out/`), a minified file, a lock file, a binary
file, a generated file and a file left out by `.gitignore`.

## What counts as a definition (the hand lists)

- TypeScript and JavaScript: `function`, `class`, `interface`, `type`, `enum`, `namespace`
  declarations; class methods and constructors; class fields holding an arrow function; module
  level `const`, `let` and `var` (kind `function` when the value is a function or arrow function,
  `class` for a class expression, else `const` or `variable`). Not counted: interface members,
  object literal methods, destructuring, local variables.
- Python: every `def` and `class` (kind `method` directly inside a class); module level
  assignments (`const` when the name is upper case, else `variable`).
- C, C++, CUDA: function definitions with a body (first line where the declaration starts,
  `template` header included), named `struct`, `union` and `enum` with a body, `typedef` names,
  `#define` macros, file level `const` and `constexpr` with an initialiser. Not counted: prototypes.
- PowerShell: `function` and `filter`, `class`, `enum`, class methods and constructors.
- CSS: each selector of a rule, custom properties (`--name`), `@keyframes` names.

The last line of a definition is the line of its closing brace, of the `;` that ends its
statement, or (Python) the last code line of its body.

## Pass rules

Set before the first run (languages note, steps L1 to L6 and L10): reader file list exact and every
planted file skipped with its reason; every lexer snippet equal to its hand list; definitions
recall and precision each at least 0.95 (recall 0.9 for C and CUDA), last lines exact for at least
0.9; at least 0.9 of the hand-listed imports resolved to the right file or outside package; package
list equal to the hand list; answering file in the pack's top three for at least 8 of 10 questions
per fixture on the plain path.

The fixtures and their hand lists were written by the same hands as the lexer and tagger: they
show the code follows its stated rules, not that the rules suit every file in the wild.
