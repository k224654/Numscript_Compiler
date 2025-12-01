# Numscript_Compiler
# NumScript Compiler

A complete compiler for the NumScript programming language implemented in a single Python file.

## 📋 Features
- **Lexical Analysis**: Tokenization with regex patterns
- **Syntax Analysis**: Recursive descent parsing with AST generation
- **Semantic Analysis**: Type checking and scope management
- **Intermediate Code**: Three-address code, quadruples, triples
- **Optimization**: Constant folding, dead code elimination
- **Execution**: IR interpreter with built-in functions

## 🚀 Quick Start
```bash
# Run a NumScript program
python numscript_compiler.py test.ns

# Interactive REPL mode
python numscript_compiler.py
