from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("hello config")
ex.observers.append(MongoObserver())


@ex.config
def my_config():
    foo = 42
    bar = "baz"


@ex.automain
def my_main():
    print("Hello, World!")
