
from abc import abstractmethod
import math
import os
import random
from tatsu import compile 
import sys
from neptune.config import get_classes_inheriting, get_config_list
import neptune.base_provider


class IdentifierProvider:
    @abstractmethod
    def provide(self, name):
        raise NotImplementedError()

class Context:
    def __init__(self, providers):
        self.providers = providers
        self.identifiers = {} 
    
    def get(self, name):
        if name in self.identifiers:
            return self.identifiers[name]
        for provider in self.providers:
            value = provider.provide(name)
            if value is not None:
                return value
        return None
    
    def __getitem__(self, name):
        return self.get(name)
    
    def __setitem__(self, name, value):
        self.identifiers[name] = value
    
    def items(self):
        all_items = {}
        for provider in self.providers:
            if hasattr(provider, 'items'):
                all_items.update(provider.items())
        all_items.update(self.identifiers)
        return all_items.items()

    def __contains__(self, name):
        if name in self.identifiers:
            return True
        for provider in self.providers:
            value = provider.provide(name)
            if value is not None:
                return True
        return False
    
    def copy(self):
        new_context = Context(self.providers.copy())
        new_context.identifiers = self.identifiers.copy()
        return new_context
    
    def __delitem__(self, name):
        if name in self.identifiers:
            del self.identifiers[name]

class HasIdentifier:
    def evaluate(self, context, args, params):
        identifier = args[0].name
        return identifier in context.identifiers



class Constant:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"Constant({self.value})"
    def __str__(self) -> str:
        return str(self.value)
    
    def evaluate(self, context: dict):
        return self.value

class Identifier:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"Identifier({self.name})"
    def __str__(self) -> str:
        return self.name
    def evaluate(self, context: dict):
        return context[self.name]

GRAMMAR_OPS = """

    start = statements ;
    statements = head:statement ";" tail:statements | head:statement ;
    string = /"[^"]*"/ ;
    identifier = /[a-zA-Z0-9_]+/ | /[|:=+*\\/<>!@$%^&\\-.?]+/ ;
    number = /[0-9]+/ | /[0-9]*\\.[0-9]+/ ;
    boolean = "true" | "false" ;
    null = "null" ;
    array = "[" list:items "]" | x:"[" "]";
    items = head:expression "," tail:items | head:expression;
    value =
        value:string 
        | value:number 
        | value:boolean 
        | value:null 
        | value:array
        | value:identifier
        | "(" expr:expression_list ")"
        | x:"(" ")"
        | "{" symbolic_expr:statements "}"
        ;
    expression_list = head:expression "," tail:expression_list | head:expression ;
    expression = first:value rest:expression | first:value ;
    statement = expr:expression | empty:{} ;
"""

class ListExpression:
    def __init__(self, items):
        self.items = items
    def __repr__(self):
        return f"ListExpression({self.items})"
    def evaluate(self, context: dict):
        return ListExpression([evaluate(context, item) for item in self.items])
    
    def __len__(self):
        return len(self.items)
    
    def __add__(self, other):
        if isinstance(other, ListExpression):
            return ListExpression(self.items + other.items)
        elif isinstance(other, list):
            return ListExpression(self.items + other)
        else:
            raise ValueError("Can only add ListExpression or list to ListExpression")

class ExpressionList:
    def __init__(self, items):
        self.items = items
    def __repr__(self):
        return f"ExpressionList({self.items})"
    def evaluate(self, context: dict):
        result = [evaluate(context, item) for item in self.items]
        if len(result) == 1:
            return result[0]
        return result
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __add__(self, other):
        if isinstance(other, ExpressionList):
            return ExpressionList(self.items + other.items)
        elif isinstance(other, list):
            return ExpressionList(self.items + other)
        else:
            raise ValueError("Can only add ExpressionList or list to ExpressionList")

class Expression:
    def __init__(self, parts):
        if isinstance(parts, list):
            self.parts = parts
        else:
            raise ValueError("Expression parts must be a list but got " + str(type(parts)))
    def __repr__(self):
        return ",".join(" ".join(str(t) for t in part) for part in self.parts)

    

class CustomOpSemantics:
    def string(self, s):
        s = s[1:-1].encode().decode('unicode_escape')
        return Constant(s)

    def identifier(self, id):
        return Identifier(str(id))

    def number(self, n):
        if '.' in n:
            return Constant(float(n))
        else:
            return Constant(int(n))

    def boolean(self, b):
        return Constant(b == 'true')

    def null(self, _):
        return Constant(None)

    def value(self, v):
        if v.symbolic_expr is not None:
            return Expression(v.symbolic_expr)
        if v.expr is not None:
            return v.expr
        if v.value is not None:
            return v.value
        else:
            return ExpressionList([])
    
    def items(self, items):
        if items.tail is None:
            return [items.head]
        else:
            return [items.head] + items.tail
    
    def array(self, items):
        return ListExpression(items.list or [])

    def expression(self, parts):
        
        left = parts.first
        right = parts.rest or [] 
        return [left] + right

    def expression_list(self, parts):
        if parts.tail is None:
            return ExpressionList([parts.head])
        else:
            return ExpressionList([parts.head]) + parts.tail

    def statements(self, stmts):
        if stmts.tail is None:
            return [stmts.head] if stmts.head is not None else []
        else:
            return [stmts.head] + stmts.tail if stmts.head is not None else stmts.tail
    
    def statement(self, stmt):
        return stmt.expr if stmt.expr is not None else None
    
    def start(self, stmts):
        return stmts

parser_ops = compile(GRAMMAR_OPS, semantics=CustomOpSemantics())

def evaluate(context, expr):
    if isinstance(expr, list) or isinstance(expr, tuple):
        if not expr:
            return None
        if len(expr) == 1:
            return evaluate(context, expr[0])
        else:
            # find operator with highest precedence
            min_precedence = 1000000
            op_index = -1
            min_op = None
            for i in range(len(expr)):
                if isinstance(expr[i], Identifier):
                    op = context.get(expr[i].name)
                    if op is None:
                        continue
                    if isinstance(op, tuple):
                        op, precedence = op
                    elif isinstance(op, Operator):
                        precedence = 100
                    else:
                        continue
                    if precedence < min_precedence:
                        min_precedence = precedence
                        op_index = i
                        min_op = op
            if op_index == -1 and len(expr) == 1:
                return evaluate(context, expr[0])
            if op_index == -1:
                return [evaluate(context, e) for e in expr]
            left = expr[:op_index]
            right = expr[op_index+1:]
            if not min_op:
                raise ValueError(f"Operator {expr[op_index].name} not found in context")
            if hasattr(min_op, "evaluate"):
                print("Evaluating %s lhs=%s rhs=%s" % (min_op, left, right))
                return min_op.evaluate(context, left, right)
            else:
                left_evaluated = evaluate(context, left)
                right_evaluated = evaluate(context, right)
                if left_evaluated is not None and right_evaluated is not None:
                    if isinstance(left_evaluated, list) and not isinstance(right_evaluated, list):
                        return [min_op(item, right_evaluated) for item in left_evaluated]
                    elif isinstance(right_evaluated, list) and not isinstance(left_evaluated, list):
                        return [min_op(left_evaluated, item) for item in right_evaluated]
                    else:
                        return min_op(left_evaluated, right_evaluated)
                else:
                    if isinstance(left_evaluated, list):
                        return min_op(*left_evaluated)
                    elif isinstance(right_evaluated, list):
                        return min_op(*right_evaluated)
                    else:
                        return min_op(left_evaluated, right_evaluated)

    else:
        if hasattr(expr, 'evaluate'):
            return expr.evaluate(context)
        else:
            return expr
        


class Operator:
    pass 

class Id:
    def evaluate(self, context, left, right):
        if len(left) > 0:
            raise ValueError("Left side of operator id has to be empty")
        args = evaluate(context, right)
        if args is None:
            raise ValueError("arguments of id evaluate to nothing")
        if len(args) != 1:
            raise ValueError("Number of arguments of id operator must be 1")
        if not isinstance(args[0], str):
            raise ValueError("Argument of id operator must be string")
        identifier = args[0]
        return Identifier(identifier)
        

class ExprEval:
    
    def evaluate(self, context, left, right):
        if left:
            raise ValueError("Left side of $ operator must be empty")
        if right and len(right) == 1:
            e = evaluate(context, right[0])
            local_context = context.copy()
            result = evaluate(local_context, e.parts)
            if isinstance(result, list):
                return result[-1] if result else None
            return result
        else:
            raise ValueError("Right side of $ operator must be a single Expression")

class ContextExtraction():
    def evaluate(self, context, left, right):
        if len(left) == 0:
            my_context = context.copy()
            expr = evaluate(my_context, right[0].parts) 
            return my_context.identifiers
        if len(left) != 1:
            raise ValueError("Left side of context extraction must be a single identifier name")
        if not right:
            raise ValueError("Right side of context extraction must be a single expression")
        right = evaluate(context, right)
        print(right)
        if not isinstance(right, Expression) and not isinstance(right, dict):
            raise ValueError("Right side of context extraction must be an Expression or object")
        my_context = context.copy()
        identifier = evaluate(my_context, left[0])
        if isinstance(identifier, str):
            if isinstance(right, dict):
                return right.get(identifier, None)
            elif isinstance(right, Expression):
                result = evaluate(my_context, right.parts)
                return my_context[identifier]
        elif isinstance(identifier, Expression):
            assert isinstance(right, Expression)
            result = evaluate(my_context, right.parts)
            return evaluate(my_context, identifier.parts)

class Identifiers():
    def evaluate(self, context, left, right):
        left = evaluate(context, left)
        right = evaluate(context, right)
        if right is None:
            raise ValueError("Right side of identifiers() must be an Expression or empty")
        if left:
            raise ValueError("Left side of identifiers() must be empty")
        if len(right) == 0:
            return ListExpression(context.identifiers.keys())
        elif len(right) == 1:
            if not isinstance(right[0], Expression):
                raise ValueError("Right side of identifiers() must be an Expression or empty")
            my_context = context.copy()
            expr = evaluate(my_context, right[0].parts)
            return ListExpression([k for k, v in my_context.items() if not k in context.identifiers])


class FunDef(Operator):
    def evaluate(self, context, left, right):
        if isinstance(left[0], ExpressionList):
            arguments = [x[0].name  for x in  left[0].items]
        else:
            arguments = [x.name for x in left[0]]
        if len(right) == 1 and isinstance(right[0], Expression):
            body = right[0].parts
        else:
            body = right 
        class Execution(Operator):
            def __init__(self, args, bindings=None):
                self.bindings = bindings or {}
                self.arguments = args
                for arg in args:
                    if arg in self.bindings:
                        del self.bindings[arg]
            def evaluate(self, ctx, left=None, right=None):
                ctx = self.bindings.copy()
                if left is None and right is None:
                    return Execution(self.arguments, ctx.copy())
                assert len(left) == 0 
                if isinstance(right[0], ExpressionList):
                    vals = right[0].items
                else:
                    vals = right[0]
            
                evals = [evaluate(context, val) for val in vals]
                
                if len(evals) > len(self.arguments):
                    raise ValueError("Too many arguments provided to function")
                for arg, val in zip(self.arguments, evals):
                    if arg not in ctx:
                        ctx[arg] = val
                unassigned = [arg for arg in self.arguments if arg not in ctx]
                if not unassigned:
                    return evaluate(ctx, body)
                else:
                    return Execution(unassigned, ctx.copy())
            def __repr__(self):
                result =  "(%s) -> %s" % (
                    ", ".join(self.arguments),
                    str(body)
                )
                return result
        return Execution(arguments, context.copy())


class Access(Operator):
    def evaluate(self, context, left, right):
        assert len(right) == 1
        my_context = context.copy()
        left = evaluate(my_context, left)
        result = evaluate(my_context, left)
        if result is not None:
            if isinstance(right[0], Identifier):
                identifier = right[0].name 
                if isinstance(result, Expression):
                    _ = evaluate(my_context, result.parts)
                return context[identifier]
            elif isinstance(right[0], Expression):
                if isinstance(result, Expression):
                    _ = evaluate(my_context, result.parts)
                    return evaluate(my_context, right[0].parts)
                elif isinstance(result, dict):
                    for k,v in result.items():
                        my_context[k] = v
                    return evaluate(my_context, right[0].parts)



class Assignment:
    def evaluate(self, context, left, right):
        if len(left) != 1:
            left = evaluate(context, left)
            if not isinstance(left, Identifier):
                raise ValueError("Left side of assignment must be a single identifier")
        else:
            left = left[0]
        if not isinstance(left, Identifier):
            raise ValueError("Left side of assignment must be an identifier")
        right_value = evaluate(context, right)
        context[left.name] = right_value
        print(f"Assigned {left.name} = {repr(right_value)}")
        return right_value

class If:
    def evaluate(self, context, left, right):
        if evaluate(context, left):
            return evaluate(context, right)


class OperatorDefinition:
    def evaluate(self, context, left, right):
        if not isinstance(left[0], Constant): 
            raise ValueError("Only integers should be set as priority")   
        left = evaluate(Context(providers=[]), left)
        assert isinstance(left, int)
        print(right)
        name = right[0].name 
        lhs_name = right[2].items[0][0].name
        rhs_name = right[2].items[1][0].name
        body = right[4:]
        class OpEval:
            def evaluate(self, ctx, lhs, rhs):
                old_ctx = ctx 
                ctx = ctx.copy()
                ctx[lhs_name] = Expression([lhs])
                ctx[rhs_name] = Expression([rhs])
                for k,v in ctx.items():
                    if k not in old_ctx:
                        old_ctx[k] = v 
                print("Body: %s"  % body)
                return evaluate(ctx, body)
        context[name] = (OpEval(), left)


class Use:
    def evaluate(self, context, lhs, rhs):
        assert len(lhs) == 0
        assert len(rhs) > 0
        if len(rhs) == 1:
            args = [] 
        else:
            ctx = Context(providers=[])
            args = evaluate(ctx, rhs[1:])
        name = rhs[0].name
        providers = get_classes_inheriting(neptune.base_provider.LibraryProvider)
        
        for p in providers:
            if p().provide(context, name, args):
                return
        raise ValueError("Cannot import name %s" % name)
            

            
            



BUILTINS = {
    ":": (OperatorDefinition(), 0),
    "use": (Use(), 0),
    "=": (Assignment(), 1),
    "->": (FunDef(), 5),
    "?": (If(), 6),
    "==": (lambda a,b: a==b, 7),
    "!": (lambda a: not a,7),
    ">": (lambda a,b: a > b, 7),
    "<": (lambda a,b: a < b, 7),
    "+": (lambda a, b: a + b if a else b, 10),
    "-": (lambda a, b: a - b if a else -b, 10),
    "*": (lambda a, b: a * b if a else 0, 20),
    "/": (lambda a, b: a / b if a else 1, 20),
    "sum": (lambda *lst: sum(lst) if lst else 0, 100),
    "identifiers": (Identifiers(), 100),
    "id": (Id(), 100),
    "random": (lambda: random.random(), 100),
    ".": (Access(), 999),
    "$": (ExprEval(), 1000),
    "@": (ContextExtraction(), 1000),
}

class BuiltinFunctionProvider(IdentifierProvider):
    def provide(self, name):
        return BUILTINS.get(name, None)

def load(filename):
    context = Context(providers = [BuiltinFunctionProvider()])
    code = "" 
    with open(filename, 'r') as file:
        last_ident = None
        for line in file:
            stripped = line.lstrip()
            ident = len(line) - len(stripped); 
            if last_ident is not None and ident <= last_ident and code.strip():
                code += " ; "
            code += stripped
            last_ident = ident
        code = parser_ops.parse(code)
        value = None
        for r in code:
            value = evaluate(context, r)  
        return value, context
    

if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else None
    if f:
        value, context = load(f)
        print(value)
    else:
        while True:
            result = input("> ") + ";"
            # result = EXAMPLE
            result = parser_ops.parse(result)
            value = None
            for r in result:
                if hasattr(r, 'evaluate'):
                    value = evaluate(context, r)
                elif isinstance(r, list) or isinstance(r, tuple):
                    value = evaluate(context, r)
                else:
                    print(f"Unknown: {r}")
            print(value)
        