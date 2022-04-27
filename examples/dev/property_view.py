from rich import print
from napari.utils.events import EventedModel, EventedList
from napari.utils.property_view import property_view


class A(EventedModel):
    x: EventedList[int] = [1, 2]
    y: EventedList[int] = [3, 5]

    class Config:
        dependencies = {'z': ['x', 'y']}

    @property_view
    def z(self) -> list[int]:
        return [x + y for x, y in zip(self.x, self.y)]

    @z.setter
    def z(self, value):
        self.x = [(z - y) for y, z in zip(self.y, value)]

    @property_view
    def xy(self) -> list[list[int]]:
        return [[x, y] for x, y in zip(self.x, self.y)]

    @xy.setter
    def xy(self, value):
        self.x, self.y = zip(*value)


a = A()
a.events.connect(lambda x: print(f'event {x.type} from source {x.source}'))

print(a)
print('a.z[0] = 8')
a.z[0] = 8
print(a)
print('a.xy[0][0] = 100')
a.xy[0][0] = 100
print(a)
