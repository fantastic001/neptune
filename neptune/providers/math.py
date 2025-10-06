
import neptune.base_provider
class MathProvider(neptune.base_provider.LibraryProvider):
    def provide(self, ctx, name, args):
        if name == "math":
            return True
        return False 