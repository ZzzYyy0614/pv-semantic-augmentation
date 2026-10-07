"""Small explicit registries for interchangeable network components."""

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, category: str):
        self.category = category
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str):
        def decorator(factory: Callable[..., T]):
            if name in self._factories:
                raise ValueError(f"Duplicate {self.category}: {name}")
            self._factories[name] = factory
            return factory

        return decorator

    def build(self, name: str, **kwargs) -> T:
        if name not in self._factories:
            raise ValueError(
                f"Unknown {self.category} '{name}'; available: {', '.join(self.names)}"
            )
        return self._factories[name](**kwargs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
