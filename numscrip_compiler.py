#!/usr/bin/env python3
"""
NumScript compiler/interpreter single-file implementation.
Save as numscript.py and run: python numscript.py program.ns
"""

import re
import sys
from collections import namedtuple, defaultdict, deque

# ------------------------
# LEXER
# ------------------------
Token = namedtuple("Token", ["type", "value", "lineno", "col"])

TOKEN_SPEC = [
    ("NUMBER",   r"\d+"),
    ("ID",       r"[A-Za-z_][A-Za-z0-9_]*"),
    ("EQ",       r"=="),
    ("ASSIGN",   r"="),
    ("NEQ",      r"!="),
    ("LE",       r"<="),
    ("GE",       r">="),
    ("LT",       r"<"),
    ("GT",       r">"),
    ("PLUS",     r"\+"),
    ("INCR",     r"\+\+"),
    ("MINUS",    r"-"),
    ("DECR",     r"--"),
    ("PLUS_EQ",  r"\+="),
    ("MINUS_EQ", r"-="),
    ("MUL",      r"\*"),
    ("DIV",      r"/"),
    ("MOD",      r"%"),
    ("POW",      r"\^"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("COMMA",    r","),
    ("SEMICOL",  r";"),
    ("COMMENT",  r"//.*"),
    ("NEWLINE",  r"\n"),
    ("SKIP",     r"[ \t\r]+"),
    ("UNKNOWN",  r"."),
]
TOK_REGEX = "|".join("(?P<%s>%s)" % pair for pair in TOKEN_SPEC)
KEYWORDS = {"let", "print", "for", "seq"}

def lex(code):
    lineno = 1
    line_start = 0
    scanner = re.finditer(TOK_REGEX, code)
    for m in scanner:
        kind = m.lastgroup
        val = m.group(kind)
        col = m.start() - line_start + 1
        if kind == "NUMBER":
            yield Token("NUMBER", int(val), lineno, col)
        elif kind == "ID":
            if val in KEYWORDS:
                yield Token(val.upper(), val, lineno, col)
            else:
                yield Token("ID", val, lineno, col)
        elif kind == "NEWLINE":
            lineno += 1
            line_start = m.end()
        elif kind == "SKIP" or kind == "COMMENT":
            continue
        elif kind == "UNKNOWN":
            raise SyntaxError(f"Unknown token {val!r} at line {lineno}")
        else:
            yield Token(kind, val, lineno, col)

# ------------------------
# PARSER (Recursive Descent)
# ------------------------
class ParseError(Exception):
    pass

class Parser:
    def __init__(self, tokens):
        self.tokens = [t for t in tokens]
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token("EOF", "", 0, 0)

    def next(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, typ):
        tok = self.peek()
        if tok.type == typ:
            return self.next()
        raise ParseError(f"Expected {typ} but got {tok.type} at {tok.lineno}:{tok.col}")

    def parse(self):
        stmts = []
        while self.peek().type != "EOF":
            stmts.append(self.parse_stmt())
        return ("program", stmts)

    def parse_stmt(self):
        t = self.peek()
        if t.type == "LET":
            return self.parse_var_decl()
        elif t.type == "PRINT":
            return self.parse_print()
        elif t.type == "SEQ":
            return self.parse_seq_decl()
        elif t.type == "FOR":
            return self.parse_for()
        elif t.type == "ID":
            return self.parse_assign()
        else:
            raise ParseError(f"Unexpected token {t.type} at {t.lineno}:{t.col}")
        

    def parse_var_decl(self):
        self.expect("LET")
        idt = self.expect("ID")
        self.expect("ASSIGN")
        expr = self.parse_expr()
        self.expect("SEMICOL")
        return ("let", idt.value, expr)

    def parse_assign(self):
        idt = self.expect("ID")
        self.expect("ASSIGN")
        expr = self.parse_expr()
        self.expect("SEMICOL")
        return ("assign", idt.value, expr)

    def parse_print(self):
        self.expect("PRINT")
        self.expect("LPAREN")
        expr = self.parse_expr()
        self.expect("RPAREN")
        self.expect("SEMICOL")
        return ("print", expr)

    def parse_seq_decl(self):
        self.expect("SEQ")
        idt = self.expect("ID")
        self.expect("ASSIGN")
        # Check for 'seq' keyword or directly for '['
        if self.peek().value == "seq":
            self.next()  # skip 'seq'
        if self.peek().type == "LBRACKET":
            self.expect("LBRACKET")
            e1 = self.parse_expr()
            self.expect("COMMA")
            e2 = self.parse_expr()
            self.expect("COMMA")
            e3 = self.parse_expr()
            self.expect("RBRACKET")
            self.expect("SEMICOL")
            return ("seq", idt.value, ("seq-range", e1, e2, e3))
        else:
            raise ParseError("seq syntax error; expected '['")

    def parse_for(self):
        self.expect("FOR")
        self.expect("LPAREN")
        # Initialization
        if self.peek().type == "LET":
            init = self.parse_var_decl_no_semicol()
        else:
            init = None
        # Condition (semicolon expect karo)
        self.expect("SEMICOL")
        cond = self.parse_expr()
        # Post expression (semicolon expect karo)  
        self.expect("SEMICOL")
        post = self.parse_post_expr()
        self.expect("RPAREN")
        block = self.parse_block()
        return ("for", init, cond, post, block)

    def parse_var_decl_no_semicol(self):
        self.expect("LET")
        idt = self.expect("ID")
        self.expect("ASSIGN")
        expr = self.parse_expr()
        # no semicolon consumed here
        return ("let", idt.value, expr)

    def parse_post_expr(self):
        tok = self.peek()
        if tok.type == "ID":
            idt = self.next().value
            if self.peek().type == "INCR":
                self.next()
                return ("incr", idt)
            elif self.peek().type == "DECR":
                self.next()
                return ("decr", idt)
            elif self.peek().type == "PLUS_EQ":
                self.next()
                expr = self.parse_expr()
                return ("plus_eq", idt, expr)
            elif self.peek().type == "MINUS_EQ":
                self.next()
                expr = self.parse_expr()
                return ("minus_eq", idt, expr)
            elif self.peek().type == "ASSIGN":
                # Allow i = i + 1
                self.next()
                expr = self.parse_expr()
                return ("assign_post", idt, expr)
            else:
                raise ParseError("Invalid post expression in for")
        else:
            raise ParseError("Invalid post expression in for")
        
    def parse_block(self):
        self.expect("LBRACE")
        stmts = []
        while self.peek().type != "RBRACE":
            stmts.append(self.parse_stmt())
        self.expect("RBRACE")
        return ("block", stmts)

    # Expression parsing (precedence climbing via recursive functions)
    def parse_expr(self):
        return self.parse_equality()

    def parse_equality(self):
        node = self.parse_additive()
        while self.peek().type in ("EQ", "NEQ", "LT", "GT", "LE", "GE"):
            op = self.next().type
            right = self.parse_additive()
            node = (op, node, right)
        return node

    def parse_additive(self):
        node = self.parse_multiplicative()
        while self.peek().type in ("PLUS", "MINUS"):
            op = self.next().type
            right = self.parse_multiplicative()
            node = (op, node, right)
        return node

    def parse_multiplicative(self):
        node = self.parse_power()
        while self.peek().type in ("MUL", "DIV", "MOD"):
            op = self.next().type
            right = self.parse_power()
            node = (op, node, right)
        return node

    def parse_power(self):
        node = self.parse_unary()
        while self.peek().type == "POW":
            self.next()
            right = self.parse_unary()
            node = ("POW", node, right)
        return node

    def parse_unary(self):
        if self.peek().type == "PLUS":
            self.next()
            return ("u+", self.parse_unary())
        if self.peek().type == "MINUS":
            self.next()
            return ("u-", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        t = self.peek()
        if t.type == "NUMBER":
            self.next()
            return ("num", t.value)
        if t.type == "ID":
            # could be function call or identifier
            idt = self.next().value
            if self.peek().type == "LPAREN":
                # function call
                self.next()
                args = []
                if self.peek().type != "RPAREN":
                    args.append(self.parse_expr())
                    while self.peek().type == "COMMA":
                        self.next()
                        args.append(self.parse_expr())
                self.expect("RPAREN")
                return ("call", idt, args)
            else:
                return ("var", idt)
        if t.type == "LPAREN":
            self.next()
            e = self.parse_expr()
            self.expect("RPAREN")
            return e
        # Agar koi aur token hai to error
        raise ParseError(f"Unexpected token {t.type} in primary at {t.lineno}:{t.col}")
# ------------------------
# SEMANTIC ANALYSIS
# ------------------------
class SemanticError(Exception):
    pass

class SemanticAnalyzer:
    def __init__(self, ast):
        self.ast = ast
        self.scopes = [dict()]  # list of dicts for symbol tables

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        self.scopes.pop()

    def declare(self, name, info):
        if name in self.scopes[-1]:
            raise SemanticError(f"Variable {name} already declared in this scope")
        self.scopes[-1][name] = info

    def lookup(self, name):
        for s in reversed(self.scopes):
            if name in s:
                return s[name]
        return None

    def analyze(self):
        return self._analyze_program(self.ast)

    def _analyze_program(self, node):
        kind, stmts = node
        if kind != "program": raise SemanticError("Ast root not program")
        for s in stmts:
            self._analyze_stmt(s)

    def _analyze_stmt(self, s):
        kind = s[0]
        if kind == "let":
            _, name, expr = s
            self._analyze_expr(expr)
            self.declare(name, {"type": "int"})
        elif kind == "assign":
            _, name, expr = s
            info = self.lookup(name)
            if not info:
                raise SemanticError(f"Assignment to undeclared variable {name}")
            self._analyze_expr(expr)
        elif kind == "print":
            _, expr = s
            self._analyze_expr(expr)
        elif kind == "seq":
            _, name, seqinfo = s
            ty = seqinfo[0]
            if ty != "seq-range":
                raise SemanticError("Unknown seq form")
            _, e1,e2,e3 = seqinfo
            self._analyze_expr(e1); self._analyze_expr(e2); self._analyze_expr(e3)
            self.declare(name, {"type":"seq"})
        elif kind == "for":
            _, init, cond, post, block = s
            self.enter_scope()
            if init:
                # init is ("let", name, expr)
                _, name, expr = init
                self._analyze_expr(expr)
                self.declare(name, {"type":"int"})
            self._analyze_expr(cond)
            # post not deeply checked here
            # block:
            _, stmts = block
            for st in stmts:
                self._analyze_stmt(st)
            self.exit_scope()
        else:
            raise SemanticError(f"Unknown stmt kind {kind}")

    def _analyze_expr(self, e):
        if e is None: return
        k = e[0]
        if k == "num":
            return
        if k == "var":
            name = e[1]
            if not self.lookup(name):
                raise SemanticError(f"Use of undeclared variable {name}")
            return
        if k == "call":
            fname = e[1]; args = e[2]
            # allow built-ins fib, fact, range
            for a in args: self._analyze_expr(a)
            if fname not in ("fib","fact","range"):
                raise SemanticError(f"Unknown function {fname}")
            return
        if k in ("u+","u-"):
            self._analyze_expr(e[1])
            return
        if k in ("PLUS","MINUS","MUL","DIV","MOD","POW","EQ","NEQ","LT","GT","LE","GE"):
            self._analyze_expr(e[1]); self._analyze_expr(e[2]); return
        # equality op tokens are directly stored as strings sometimes:
        if k in ("PLUS","MINUS","MUL","DIV"):
            self._analyze_expr(e[1]); self._analyze_expr(e[2]); return
        raise SemanticError(f"Unknown expr kind {k}")

# ------------------------
# IR (three-address) generation
# ------------------------
class IRInstr:
    def __init__(self, op, dest=None, src1=None, src2=None, comment=None):
        self.op = op
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
        self.comment = comment
    def __repr__(self):
        parts = [self.op]
        if self.dest is not None: parts.append(str(self.dest))
        if self.src1 is not None: parts.append(str(self.src1))
        if self.src2 is not None: parts.append(str(self.src2))
        if self.comment: parts.append("//"+self.comment)
        return " ".join(parts)

class IRGenerator:
    def __init__(self):
        self.temp_counter = 0
        self.ir = []
    def new_temp(self):
        t = f"t{self.temp_counter}"; self.temp_counter += 1; return t
    def emit(self, op, dest=None, src1=None, src2=None, comment=None):
        i = IRInstr(op, dest, src1, src2, comment)
        self.ir.append(i)
        return i

    def gen_program(self, ast):
        _, stmts = ast
        for s in stmts:
            self.gen_stmt(s)
        return self.ir

    def gen_stmt(self, s):
        kind = s[0]
        if kind == "let":
            _, name, expr = s
            val = self.gen_expr(expr)
            self.emit("ASSIGN", name, val)
        elif kind == "assign":
            _, name, expr = s
            val = self.gen_expr(expr)
            self.emit("ASSIGN", name, val)
        elif kind == "print":
            _, expr = s
            val = self.gen_expr(expr)
            self.emit("PRINT", None, val)
        elif kind == "seq":
            _, name, seqinfo = s
            _, a,b,c = seqinfo
            ta = self.gen_expr(a); tb = self.gen_expr(b); tc = self.gen_expr(c)
            self.emit("SEQ", name, (ta,tb,tc))
        elif kind == "for":
            _, init, cond, post, block = s
            if init:
                _, name, expr = init
                val = self.gen_expr(expr)
                self.emit("ASSIGN", name, val)
            # cond -> generate compare
            loop_label = f"L{len(self.ir)}"
            after_label = f"A{len(self.ir)}"
            # label
            self.emit("LABEL", loop_label)
            cond_val = self.gen_expr(cond)
            self.emit("JZ", None, cond_val, after_label)
            # block
            _, stmts = block
            for st in stmts:
                self.gen_stmt(st)
                # post
            if post[0] == "incr":
                self.emit("INC", post[1], 1)
            elif post[0] == "decr":
                self.emit("DEC", post[1], 1)
            elif post[0] == "plus_eq":
                self.emit("ADD", post[1], post[1], self.gen_expr(post[2]))
            elif post[0] == "minus_eq":
                self.emit("SUB", post[1], post[1], self.gen_expr(post[2]))
            elif post[0] == "assign_post":  # YEH NAYA ADD KARO
                val = self.gen_expr(post[2])
                self.emit("ASSIGN", post[1], val)
                self.emit("JMP", None, loop_label)  # YEH BAHAR NIKALO
                self.emit("LABEL", after_label)
        elif kind == "expr_stmt":
            self.gen_expr(s[1])
        else:
            raise RuntimeError("Unknown stmt in IR gen: "+str(kind))

    def gen_expr(self, e):
        k = e[0]
        if k == "num":
            return e[1]
        if k == "var":
            return e[1]
        if k == "call":
            fname = e[1]; args = e[2]
            vals = [self.gen_expr(a) for a in args]
            t = self.new_temp()
            self.emit("CALL", t, (fname, vals))
            return t
        if k == "u+":
            return self.gen_expr(e[1])
        if k == "u-":
            v = self.gen_expr(e[1])
            t = self.new_temp()
            self.emit("NEG", t, v)
            return t
        if k in ("PLUS","MINUS","MUL","DIV","MOD","POW","EQ","NEQ","LT","GT","LE","GE"):
            left = self.gen_expr(e[1]); right = self.gen_expr(e[2])
            t = self.new_temp()
            opmap = {"PLUS":"ADD","MINUS":"SUB","MUL":"MUL","DIV":"DIV","MOD":"MOD","POW":"POW",
                     "EQ":"EQ","NEQ":"NEQ","LT":"LT","GT":"GT","LE":"LE","GE":"GE"}
            self.emit(opmap[k], t, left, right)
            return t
        # fallback: raw tuple like ("+",..)
        if isinstance(k, str) and k in ("PLUS","MINUS","MUL","DIV"):
            return self.gen_expr(("PLUS", e[1], e[2]))
        raise RuntimeError("Unknown expr in IR gen: "+str(e))

# ------------------------
# OPTIMIZER
# - constant folding (simple) and dead-code elimination (simple: drop ASSIGN to unused vars)
# ------------------------
def constant_folding(ir):
    # simplistic: fold operations with literal integer operands
    new_ir = []
    consts = {}
    for instr in ir:
        if instr.op in ("ADD","SUB","MUL","DIV","MOD","POW","NEG","EQ","NEQ","LT","GT","LE","GE"):
            s1 = instr.src1; s2 = instr.src2
            v1 = consts.get(s1, None) if isinstance(s1, str) and s1.startswith("t") else (s1 if isinstance(s1,int) else None)
            v2 = consts.get(s2, None) if isinstance(s2, str) and s2.startswith("t") else (s2 if isinstance(s2,int) else None)
            if instr.op == "NEG":
                if v1 is not None:
                    val = -v1
                    consts[instr.dest] = val
                    new_ir.append(IRInstr("LOAD_CONST", instr.dest, val))
                    continue
            else:
                if v1 is not None and v2 is not None:
                    try:
                        if instr.op == "ADD": val = v1 + v2
                        elif instr.op == "SUB": val = v1 - v2
                        elif instr.op == "MUL": val = v1 * v2
                        elif instr.op == "DIV": val = v1 // v2 if v2!=0 else 0
                        elif instr.op == "MOD": val = v1 % v2 if v2!=0 else 0
                        elif instr.op == "POW": val = v1 ** v2
                        elif instr.op == "EQ": val = 1 if v1==v2 else 0
                        elif instr.op == "NEQ": val = 1 if v1!=v2 else 0
                        elif instr.op == "LT": val = 1 if v1<v2 else 0
                        elif instr.op == "GT": val = 1 if v1>v2 else 0
                        elif instr.op == "LE": val = 1 if v1<=v2 else 0
                        elif instr.op == "GE": val = 1 if v1>=v2 else 0
                        consts[instr.dest] = val
                        new_ir.append(IRInstr("LOAD_CONST", instr.dest, val))
                        continue
                    except Exception:
                        pass
        if instr.op == "ASSIGN":
            # if src is a known const temp propagate
            s = instr.src1
            if isinstance(s, str) and s in consts:
                new_ir.append(IRInstr("LOAD_CONST", instr.dest, consts[s]))
                consts[instr.dest] = consts[s]
                continue
        new_ir.append(instr)
    return new_ir

def dead_code_elim(ir):
    # very basic: find variables that are printed or used by others, keep their assignments
    used = set()
    # mark uses
    for instr in ir:
        if instr.op in ("PRINT", "JZ", "JMP", "CALL"):
            # mark srcs
            if instr.src1:
                if isinstance(instr.src1, tuple) and instr.op=="CALL":
                    # CALL dest (fname, args)
                    _, args = instr.src1
                    for a in args:
                        if isinstance(a, str): used.add(a)
                elif isinstance(instr.src1, str): used.add(instr.src1)
        else:
            if instr.src1 and isinstance(instr.src1, str): used.add(instr.src1)
            if instr.src2 and isinstance(instr.src2, str): used.add(instr.src2)
    # keep labels, prints and assignments to used
    new_ir = []
    for instr in ir:
        if instr.op in ("LABEL","JMP","JZ","PRINT","CALL","LOAD_CONST"):
            new_ir.append(instr)
        elif instr.op == "ASSIGN":
            if instr.dest in used:
                new_ir.append(instr)
        else:
            # arithmetic ops: if dest used keep; else drop
            if instr.dest and instr.dest in used:
                new_ir.append(instr)
    return new_ir

# ------------------------
# INTERPRETER (executes IR)
# Supports CALL for built-ins
# ------------------------
class Interpreter:
    def __init__(self, ir):
        self.ir = ir
        self.memory = {}   # varname -> int or seq
        self.labels = {}
        self.ip = 0
        # pre-scan for labels
        for i, instr in enumerate(self.ir):
            if instr.op == "LABEL":
                self.labels[instr.dest] = i

    def run(self):
        while self.ip < len(self.ir):
            instr = self.ir[self.ip]
            # print("EXEC", self.ip, instr)
            self.ip += 1
            op = instr.op
            if op == "LOAD_CONST":
                self.memory[instr.dest] = instr.src1
            elif op == "ASSIGN":
                val = self._eval_operand(instr.src1)
                self.memory[instr.dest] = val
            elif op == "PRINT":
                val = self._eval_operand(instr.src1)
                print(val)
            elif op == "SEQ":
                name = instr.dest
                a,b,c = instr.src1
                a = self._eval_operand(a); b = self._eval_operand(b); c = self._eval_operand(c)
                arr = list(range(a, b+1, c)) if c!=0 else []
                self.memory[name] = arr
            elif op == "CALL":
                dest = instr.dest; fname, args = instr.src1
                argvals = [self._eval_operand(a) for a in args]
                res = self._call_builtin(fname, argvals)
                self.memory[dest] = res
            elif op in ("ADD","SUB","MUL","DIV","MOD","POW","NEG"):
                dest = instr.dest
                if op == "NEG":
                    v = self._eval_operand(instr.src1)
                    self.memory[dest] = -v
                else:
                    v1 = self._eval_operand(instr.src1)
                    v2 = self._eval_operand(instr.src2)
                    if op=="ADD": self.memory[dest] = v1+v2
                    if op=="SUB": self.memory[dest] = v1-v2
                    if op=="MUL": self.memory[dest] = v1*v2
                    if op=="DIV": self.memory[dest] = v1//v2 if v2!=0 else 0
                    if op=="MOD": self.memory[dest] = v1%v2 if v2!=0 else 0
                    if op=="POW": self.memory[dest] = v1**v2
            elif op in ("EQ","NEQ","LT","GT","LE","GE"):
                dest = instr.dest
                v1 = self._eval_operand(instr.src1)
                v2 = self._eval_operand(instr.src2)
                if op=="EQ": self.memory[dest] = 1 if v1==v2 else 0
                if op=="NEQ": self.memory[dest] = 1 if v1!=v2 else 0
                if op=="LT": self.memory[dest] = 1 if v1<v2 else 0
                if op=="GT": self.memory[dest] = 1 if v1>v2 else 0
                if op=="LE": self.memory[dest] = 1 if v1<=v2 else 0
                if op=="GE": self.memory[dest] = 1 if v1>=v2 else 0
            elif op == "JZ":
                cond = self._eval_operand(instr.src1)
                label = instr.src2
                if cond == 0:
                    self.ip = self.labels[label]
            elif op == "JMP":
                label = instr.src1
                self.ip = self.labels[label]
            elif op == "LABEL":
                pass
            elif op == "INC":
                name = instr.dest
                v = self.memory.get(name,0)
                self.memory[name] = v + instr.src1
            elif op == "DEC":
                name = instr.dest
                v = self.memory.get(name,0)
                self.memory[name] = v - instr.src1
            else:
                # unknown / skip
                pass

    def _eval_operand(self, x):
        if isinstance(x, int):
            return x
        if isinstance(x, str):
            # immediate var or temp
            return self.memory.get(x, 0)
        return x

    def _call_builtin(self, name, args):
        if name == "fib":
            n = args[0]
            return self._fib(n)
        if name == "fact":
            n = args[0]
            return self._fact(n)
        if name == "range":
            a,b = args[0], args[1]
            s = args[2] if len(args)>2 else 1
            return list(range(a,b+1,s))
        raise RuntimeError("Unknown builtin "+name)

    def _fib(self,n):
        if n <= 0: return 0
        a,b=0,1
        for _ in range(n-1):
            a,b=b,a+b
        return b

    def _fact(self,n):
        v=1
        for i in range(2,n+1):
            v*=i
        return v

# ------------------------
# DRIVER / CLI
# ------------------------
def compile_and_run(source):
    tokens = list(lex(source))
    tokens.append(Token("EOF","",0,0))
    p = Parser(tokens)
    ast = p.parse()
    # semantic
    sem = SemanticAnalyzer(ast)
    sem.analyze()
    # IR
    irgen = IRGenerator()
    ir = irgen.gen_program(ast)
    # optimize
    ir = constant_folding(ir)
    ir = dead_code_elim(ir)
    
    # DEBUG: IR print karo
    print("=== GENERATED IR ===")
    for instr in ir:
        print(instr)
    print("=== OUTPUT ===")
    
    # run
    interp = Interpreter(ir)
    interp.run()

def repl():
    print("NumScript REPL. Type 'exit' to quit.")
    buf = ""
    while True:
        try:
            line = input(">>> ")
            if line.strip() == "exit":
                break
            buf += line + "\n"
            if line.strip().endswith(";") or line.strip()=="":
                try:
                    compile_and_run(buf)
                except Exception as e:
                    print("Error:", e)
                buf = ""
        except KeyboardInterrupt:
            break

def main():
    if len(sys.argv) == 1:
        repl()
    else:
        fname = sys.argv[1]
        with open(fname,'r') as f:
            source = f.read()
        compile_and_run(source)

if __name__ == "__main__":
    main()