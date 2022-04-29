from rich import print
from napari.utils.events import EventedModel, EventedList
from napari.utils.property_view import property_view


class A(EventedModel):
    a: EventedList = EventedList([10, 20])


class B(EventedModel):
    b: A = A()


class M(EventedModel):
    x: B = B()
    y: B = B(b=A(a=EventedList(([1, 2]))))
    f: int = 2

    class Config:
        dependencies = {
            'z': ['x', 'y'],
            'w': ['x', 'f'],
        }

    @property_view
    def z(self):
        return B(b=A(a=EventedList([self.x.b.a[0], self.y.b.a[1]])))

    @z.setter
    def z(self, value):
        self.x.b.a[0] = value.b.a[0]
        self.y.b.a[1] = value.b.a[1]

    @property_view
    def w(self):
        return [[el * self.f for el in self.x.b.a], [el // self.f for el in self.y.b.a]]

    @w.setter
    def w(self, value):
        self.x.b.a[0] = value[0][0] // self.f
        self.x.b.a[1] = value[0][1] // self.f
        self.y.b.a[0] = value[1][0] * self.f
        self.y.b.a[1] = value[1][1] * self.f


m = M()
for field in 'xyzw':
    getattr(m.events, field).connect(lambda _, field=field: print(f'Event {field} triggered'))

print(f'>>> {m=}')
print('>>> m.z.b.a[1] = 12')
# deep nested properties are fine
m.z.b.a[1] = 12
print('>>> m.w = [[1, 2], [3, 4]]')
# complex properties trigger all the appropriate events
m.w = [[1, 2], [3, 4]]
print('>>> m.w[1][0] = 2')
# but if only some parts of nested properties are updated, only the relative events are triggered
m.w[1][0] = 2
print(f'>>> {m=}')
