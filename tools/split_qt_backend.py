import sys

names = ["MAIN", "SECOND", "THIRD", "FOURTH"]
num = int(sys.argv[1])
values = sys.argv[2].split(",")

if num < len(values):
    print(f"{names[num]}={values[num]}")
else:
    print(f"{names[num]}=none")
