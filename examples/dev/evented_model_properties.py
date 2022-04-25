from napari.utils.events import EventedModel, EventedList


class A(EventedModel):
    x: EventedList[int] = [1, 2]
    y: int = 2

    class Config:
        dependencies = {'z': ['x', 'y']}

    @property
    def z(self) -> EventedList[int, int]:
        return [x + self.y for x in self.x]

    @z.setter
    def z(self, value):
        self.x = [z - self.y for z in value]

def rep(x):
    print(repr(x))

a = A()
rep(a)

a.events.connect(lambda e: print(f'Event: {e.type} was updated'))
print('a.z = [0, 0]')
a.z = [0, 0]
rep(a)

print('a.y = 1')
a.y = 1
rep(a)

print('a.x[1] = 5')
a.x[1] = 5
rep(a)

print('a.z[0] = -3')
a.z[0] = -3
rep(a)
