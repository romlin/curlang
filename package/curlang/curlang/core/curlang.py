import io
import inspect
import re
import shlex
import tokenize

from lark import Lark, Transformer, Token

curlang_grammar = r"""
start: statement+
statement: assignment_stmt | if_stmt | cmd_stmt | delete_stmt | fail_stmt | find_stmt | get_stmt | make_stmt | pass_stmt | print_stmt | python_stmt | use_stmt | unzip_stmt
assignment_stmt: var "=" expr ";"?
if_stmt: "if" "(" condition ")" block ("else" block)?
condition: expr "==" expr
expr: STRING | var
cmd_block: "{" cmd_content? "}"
cmd_content: /[^}]+/
cmd_stmt: "cmd" (cmd_block | RAW) var? ";"?
delete_stmt: "delete" STRING ";"?
fail_stmt: "fail" STRING
find_stmt: KEYWORD STRING block "else" STRING ";"?
get_stmt: "get" STRING "as" STRING (block)?
make_stmt: "make" STRING ";"?
pass_stmt: "pass" STRING
print_stmt: "print" (STRING | var)
python_stmt: "python" python_block var? ";"?
python_block: "{" python_content* "}"
python_content: /[^{}]+/ | python_block
use_stmt: "use" module_list ";"?
module_list: module ("," module)*
module: MODULE (":" MODULE)?
unzip_stmt: "unzip" STRING ";"?
block: "{" statement+ "}"
KEYWORD: "!find" | "find"
COMMENT: /#[^\n]*/
RAW: /[^\n]+/
STRING: /"([^"\\]*(\\.[^"\\]*)*)"/
var: VAR
VAR: /@[a-zA-Z_]\w*/
MODULE: /[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*/
%import common.CNAME
%import common.WS
%ignore WS
%ignore COMMENT
"""


class CurlangTransformer(Transformer):
    def assignment_stmt(self, items):
        return {"type": "assignment", "var": items[0], "value": items[1],
                "runtime": "bash"}

    def if_stmt(self, items):
        result = {"type": "if", "condition": items[0], "block": items[1],
                  "runtime": "bash"}
        if len(items) == 3:
            result["else"] = items[2]
        return result

    def condition(self, items):
        return {"left": items[0], "operator": "==", "right": items[1]}

    def expr(self, items):
        return items[0]

    def block(self, items):
        result = []
        for stmt in items:
            if isinstance(stmt, list):
                result.extend(stmt)
            else:
                result.append(stmt)
        return result

    def cmd_block(self, items):
        return items[0] if items else ""

    def cmd_content(self, items):
        return items[0].value if hasattr(items[0], "value") else items[0]

    def cmd_stmt(self, items):
        command_text = items[0].strip()
        result = {"type": "cmd", "command": command_text, "runtime": "bash"}
        if len(items) > 1:
            result["capture"] = items[1]
        return result

    def delete_stmt(self, items):
        return {"type": "delete", "target": items[0], "runtime": "bash"}

    def fail_stmt(self, items):
        return {"type": "fail", "message": items[0], "runtime": "bash"}

    def find_stmt(self, items):
        return {"type": "find", "negated": (items[0] == "!find"),
                "filename": items[1], "block": items[2], "message": items[3],
                "runtime": "bash"}

    def get_stmt(self, items):
        r = {"type": "get", "url": items[0], "destination": items[1],
             "runtime": "bash"}
        if len(items) > 2:
            r["block"] = items[2]
        return r

    def make_stmt(self, items):
        return {"type": "make", "target": items[0], "runtime": "bash"}

    def pass_stmt(self, items):
        return {"type": "pass", "message": items[0], "runtime": "bash"}

    def print_stmt(self, items):
        return {"type": "print", "message": items[0], "runtime": "bash"}

    def python_content(self, items):
        item = items[0]
        if isinstance(item, Token):
            return item.value
        return item

    def python_block(self, items):
        return "".join(items)

    def python_stmt(self, items):
        code = items[0]
        cleaned_code = inspect.cleandoc(code)
        substituted_code = substitute_env_variables(cleaned_code)
        result = {"type": "python", "code": substituted_code,
                  "runtime": "python"}
        if len(items) > 1:
            result["capture"] = items[1]
        return result

    def use_stmt(self, items):
        modules = items[0]
        lines = []
        for mod in modules:
            if isinstance(mod, tuple):
                alias, mod_name = mod
                if alias.islower():
                    lines.append(f"import {mod_name} as {alias}")
                else:
                    lines.append(f"from {mod_name} import {alias}")
            else:
                lines.append(f"import {mod}")
        return {"type": "use", "runtime": "python", "imports": lines}

    def module_list(self, items):
        return items

    def module(self, items):
        if len(items) == 2:
            return (items[0], items[1])
        return items[0]

    def unzip_stmt(self, items):
        return {"type": "unzip", "target": items[0], "runtime": "bash"}

    def start(self, items):
        return items

    def statement(self, items):
        return items[0]

    def RAW(self, token):
        return token.value

    def STRING(self, token):
        return token.value[1:-1]

    def var(self, token):
        return token[0].value


parser = Lark(curlang_grammar, parser="lalr", transformer=CurlangTransformer())


def substitute_env_variables(code):
    code = code + "\n"
    try:
        code_bytes = code.encode("utf-8")
        token_gen = list(tokenize.tokenize(io.BytesIO(code_bytes).readline))
    except tokenize.TokenError:
        return code
    tokens = []
    i = 0
    while i < len(token_gen):
        tok = token_gen[i]
        if tok.type in (
        tokenize.ENCODING, tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT):
            tokens.append(tok)
            i += 1
            continue
        if tok.type == tokenize.OP and tok.string == "@":
            if tok.line.lstrip().startswith("@"):
                tokens.append(tok)
                i += 1
                continue
            if i + 1 < len(token_gen) and token_gen[
                i + 1].type == tokenize.NAME:
                next_tok = token_gen[i + 1]
                replacement = f'os.environ.get("{next_tok.string}")'
                new_tok = tokenize.TokenInfo(
                    type=tokenize.NAME,
                    string=replacement,
                    start=tok.start,
                    end=next_tok.end,
                    line=tok.line
                )
                tokens.append(new_tok)
                i += 2
                continue
        tokens.append(tok)
        i += 1
    try:
        new_code = tokenize.untokenize(tokens).decode("utf-8")
    except Exception:
        new_code = code
    return new_code


def command_to_code(cmd):
    t = cmd.get("type")
    if t == "find":
        runtime = "bash"
        f = shlex.quote(cmd["filename"])
        m = shlex.quote(cmd["message"])
        c = ""
        if cmd.get("block"):
            blk = cmd["block"]
            if isinstance(blk, dict):
                blk = [blk]
            c = "\n".join(command_to_code(x) for x in blk)
        if cmd["negated"]:
            code = f'if [ ! -f {f} ]; then\n{c}\nelse\n    echo {m}\nfi'
        else:
            code = f'if [ -f {f} ]; then\n{c}\nelse\n    echo {m}\nfi'
        return f'# runtime: {runtime}\n{code}'
    elif t == "get":
        runtime = "bash"
        u = shlex.quote(cmd["url"])
        d = shlex.quote(cmd["destination"])
        dl = shlex.quote(
            f'Downloading from {cmd["url"]} to {cmd["destination"]}')
        if cmd.get("block"):
            blk = cmd["block"]
            if isinstance(blk, dict):
                blk = [blk]
            s, f_ = process_get_inner_block(blk)
            code = f'echo {dl}\ncurl -L {u} -o {d}\nret=$?\nif [ $ret -eq 0 ]; then\n    {s}\nelse\n    {f_}\nfi'
        else:
            code = f'echo {dl}\ncurl -L {u} -o {d}'
        return f'# runtime: {runtime}\n{code}'
    elif t == "cmd":
        runtime = "bash"
        command = cmd["command"]
        if "capture" in cmd:
            capture = cmd["capture"]
            var_name = capture[1:] if capture.startswith("@") else capture
            code = (f'{var_name}=$({command})\n'
                    f'export {var_name}\n'
                    f'var_value_json=$(printf \'%s\' "${{{var_name}}}" | python3 -c "import json, sys; print(json.dumps(sys.stdin.read()))")\n'
                    f'send_code_to_python_and_wait <<EOF_CODE\n'
                    f'import os\n'
                    f'os.environ["{var_name}"] = $var_value_json\n'
                    f'EOF_CODE')
        else:
            code = command
        return f'# runtime: {runtime}\n{code}'
    elif t == "delete":
        runtime = "bash"
        target = shlex.quote(cmd["target"])
        code = f'rm -rf {target}'
        return f'# runtime: {runtime}\n{code}'
    elif t == "make":
        runtime = "bash"
        target = shlex.quote(cmd["target"])
        code = f'mkdir -p {target}'
        return f'# runtime: {runtime}\n{code}'
    elif t == "pass":
        runtime = "bash"
        m = shlex.quote(f'PASS: {cmd["message"]}')
        code = f'echo {m}'
        return f'# runtime: {runtime}\n{code}'
    elif t == "python":
        runtime = "python"
        py_code = cmd["code"]
        if "capture" in cmd:
            capture = cmd["capture"]
            var_name = capture[1:] if capture.startswith("@") else capture
            code = (
                f'{var_name}=$(send_code_to_python_and_wait <<\'EOF_CODE\'\n'
                f"{py_code}\n"
                f"EOF_CODE\n"
                f')\n'
                f'export {var_name}')
        else:
            code = (f"send_code_to_python_and_wait << 'EOF_CODE'\n"
                    f"{py_code}\n"
                    f"EOF_CODE")
        return f'# runtime: {runtime}\n{code}'
    elif t == "fail":
        runtime = "bash"
        m = shlex.quote(f'FAIL: {cmd["message"]}')
        code = f'echo {m}'
        return f'# runtime: {runtime}\n{code}'
    elif t == "print":
        runtime = "bash"
        message = cmd["message"]
        if isinstance(message, str) and re.fullmatch(r"@[a-zA-Z_]\w*",
                                                     message):
            var_name = message[1:]
            code = f'echo "${{{var_name}}}"'
        else:
            m = shlex.quote(message)
            code = f'echo {m}'
        return f'# runtime: {runtime}\n{code}'
    elif t == "use":
        runtime = "python"
        py_code = "\n".join(cmd.get("imports", []))
        code = (f"send_code_to_python_and_wait << 'EOF_CODE'\n"
                f"{py_code}\n"
                f"EOF_CODE")
        return f'# runtime: {runtime}\n{code}'
    elif t == "assignment":
        runtime = "bash"
        var_name = cmd["var"][1:]
        value = cmd["value"]
        if isinstance(value, str):
            value = shlex.quote(value)
        code = (f'export {var_name}={value}\n'
                f'var_value_json=$(printf "%s" "${{{var_name}}}" | python3 -c "import json, sys; print(json.dumps(sys.stdin.read().strip()))")\n'
                f'send_code_to_python_and_wait <<EOF_CODE\n'
                f'import os\n'
                f'os.environ["{var_name}"] = $var_value_json\n'
                f'EOF_CODE')
        return f'# runtime: {runtime}\n{code}'
    elif t == "if":
        runtime = "bash"
        cond = cmd["condition"]
        left = cond["left"]
        right = cond["right"]
        if isinstance(left, str) and left.startswith("@"):
            left = '"${%s}"' % left[1:]
        else:
            left = shlex.quote(left)
        if isinstance(right, str) and right.startswith("@"):
            right = '"${%s}"' % right[1:]
        else:
            right = shlex.quote(right)
        blk = cmd["block"]
        if isinstance(blk, dict):
            blk = [blk]
        block_code = "\n".join(command_to_code(x) for x in blk)
        if "else" in cmd:
            els = cmd["else"]
            if isinstance(els, dict):
                els = [els]
            else_code = "\n".join(command_to_code(x) for x in els)
            code = f'if [ {left} == {right} ]; then\n{block_code}\nelse\n{else_code}\nfi'
        else:
            code = f'if [ {left} == {right} ]; then\n{block_code}\nfi'
        return f'# runtime: {runtime}\n{code}'
    elif t == "unzip":
        runtime = "bash"
        target = shlex.quote(cmd["target"])
        code = f'gunzip {target}'
        return f'# runtime: {runtime}\n{code}'
    else:
        runtime = "bash"
        u = shlex.quote(f'Unknown command type: {cmd}')
        code = f'echo {u}'
        return f'# runtime: {runtime}\n{code}'


def process_get_inner_block(block):
    success_cmds = []
    failure_cmds = []
    for cmd in block:
        if not isinstance(cmd, dict):
            raise ValueError(f"Expected dict in get inner block, got: {cmd}")
        if cmd["type"] == "pass":
            runtime = "bash"
            msg = shlex.quote(f'PASS: {cmd["message"]}')
            success_cmds.append(f'# runtime: {runtime}\necho {msg}')
        elif cmd["type"] == "fail":
            runtime = "bash"
            msg = shlex.quote(f'FAIL: {cmd["message"]}')
            failure_cmds.append(f'# runtime: {runtime}\necho {msg}')
        elif cmd["type"] == "print":
            runtime = "bash"
            msg = shlex.quote(cmd["message"])
            success_cmds.append(f'# runtime: {runtime}\necho {msg}')
        else:
            runtime = "bash"
            unknown = shlex.quote(f'Unknown command type in get block: {cmd}')
            success_cmds.append(f'# runtime: {runtime}\necho {unknown}')
            failure_cmds.append(f'# runtime: {runtime}\necho {unknown}')
    return ("\n".join(success_cmds), "\n".join(failure_cmds))


def run_curlang_block(code):
    try:
        ast = parser.parse(code)
    except Exception as e:
        raise ValueError(e)
    if not isinstance(ast, list):
        if hasattr(ast, "children"):
            ast = ast.children
        else:
            ast = [ast]
    r = []
    runtimes = set()
    for cmd in ast:
        if not isinstance(cmd, dict):
            raise ValueError(f"Expected a dict, but got: {cmd}")
        r.append(command_to_code(cmd))
        if "runtime" in cmd:
            runtimes.add(cmd["runtime"])
    final_code = "\n".join(r)
    if len(runtimes) == 1:
        overall_runtime = runtimes.pop()
    else:
        overall_runtime = "mixed(" + ", ".join(runtimes) + ")"
    return {"runtime": overall_runtime, "code": final_code}
