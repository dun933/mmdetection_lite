import pkgutil
import torchvision
from importlib import import_module
model_urls = dict()
for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
    if ispkg:
        continue
    _zoo = import_module('torchvision.models.{}'.format(name))
    if hasattr(_zoo, 'model_urls'):
        _urls = getattr(_zoo, 'model_urls')
        model_urls.update(_urls)
print(model_urls.keys())