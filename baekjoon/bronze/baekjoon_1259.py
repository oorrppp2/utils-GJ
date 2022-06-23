while True:
    n = str(input())
    if n == '0':
        break
    ans = ''
    if n == n[::-1]:
        ans = 'yes'
    else:
        ans = 'no'

    print(ans)