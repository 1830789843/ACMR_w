from os.path import dirname, basename, isfile
import glob

# print(__file__)
# print(dirname(__file__))
# print(basename(dirname(__file__)))
modules = glob.glob(dirname(__file__)+"/*.py")
# print(modules)
# print(modules[0])
# print(isfile(modules[0]))
__all__ = [basename(f)[:-3] for f in modules if isfile(f)]
# print(__all__)
# __all__2 = [basename(f) for f in modules if isfile(f)]
# print(__all__2)

modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f)]
