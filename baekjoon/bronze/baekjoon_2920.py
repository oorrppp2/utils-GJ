melody = list(map(int, input().split(' ')))
asc = [i for i in range(1,9)]
dsc = [i for i in range(8,0,-1)]

if melody == asc:
    print("ascending")
elif melody == dsc:
    print("descending")
else:
    print("mixed")