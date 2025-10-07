
import  neptune.base_provider
from neptune.lang import Identifier 


class EqualsOverride:

    def __init__(self, existing):
        self.existing = existing
    def evaluate(self, context, left, right):
        name = left[0].name
        print("Assigning to %s value %s" % (name, right))
        return self.existing.evaluate(context, left, right)

class DebugProvider(neptune.base_provider.LibraryProvider):
    def provide(self, ctx, name, args):
        if name == "debug":
            op, prio = ctx["="]
            ctx["="] = (EqualsOverride(op), prio)
            self.identifiers = ctx.identifiers.copy()  
            return True
        return False
    def finish(self, ctx):
        for k,v in ctx.identifiers.items():
            if k not in self.identifiers:
                print("%s = %s" % (k, v))