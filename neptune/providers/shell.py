
import shutil
import neptune.base_provider
import subprocess 
from neptune.lang import evaluate, IdentifierProvider

class CommandOperator:
    def __init__(self, command):
        self.command = command

    def evaluate(self, ctx, lhs, rhs):
        assert len(lhs) == 0 
        args = [evaluate(ctx, arg) for arg in rhs if arg]
        result = subprocess.run([self.command] + args, capture_output=True, text=True)
        return result.stdout.strip()

class CommandProvider(IdentifierProvider):
    def provide(self, name):
        path = shutil.which(name).encode('utf-8') if shutil.which(name) else None
        return (CommandOperator(path), 100) if path else None

class ShellProvider(neptune.base_provider.LibraryProvider):
    def provide(self, ctx, name, args):
        if name == "shell":
            ctx.providers.append(CommandProvider())
            return True
        return False