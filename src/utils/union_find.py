from collections import defaultdict
from typing import Generic, TypeVar, Dict, List, Iterable

T = TypeVar("T")


class UnionFind(Generic[T]):
    def __init__(self):
        self.parents: Dict[T, T] = {}

    def find(self, x: T) -> T:
        parent = self.parents.get(x, x)
        if parent != x:
            parent = self.find(parent)
            self.parents[x] = parent
        return parent

    def union(self, child: T, parent: T):
        self.parents[self.find(child)] = self.find(parent)

    def groups(self) -> Iterable[List[T]]:
        groups = defaultdict(list)
        for x in self.parents:
            groups[self.find(x)].append(x)
        return groups.values()
