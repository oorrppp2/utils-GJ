sugar = int(input())
carrier = 0
if sugar % 5 == 0:
    carrier = sugar // 5
else:
    while sugar > 0:
        carrier += 1
        sugar -= 3
        if sugar % 5 == 0:
            carrier += sugar // 5
            break
            
if sugar < 0:
    carrier = -1
print(carrier) 
