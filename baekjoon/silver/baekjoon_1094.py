x = int(input())

stick = 64
stick_list = []
stick_list.append(stick)            # 처음에 64cm 막대기 하나 가지고있음.

while ??? != x:                     # ??? 가 x와 같지 않은동안 반복 
    if ??? > x:                     # 문제의 1번 문장을 잘 읽어보면, ???이 X보다 크다면 아래의 과정을 반복하라고 써있음.
        s = stick_list.pop()        # stick_list의 마지막에 저장되어있는 가ㅄ이 가장 짧은 가ㅄ 이므로 pop하여 s에 저장해둠.
                                    # 1.1에서 가장 짧은 것을 절반으로 자른다고 했으므로 s를 반으로 자를것임.
        stick_list.append(???)      # 가지고 있는 막대 중 가장 짧은 것을 절반으로 잘라서 가지고 있는 막대 list (stick_list)에 저장해둔다.
        if ???:                     # 1.2 문장을 잘 읽어볼 것. 위에서 절반으로 자른 막대 두개 중 하나를 버리고
                                    # 남아있는 막대 list (stick_list) 의 합이 x보다 크다면 자른 막대의 절반 중 하나를 버린다.
            ???
        else:                       # 만약 남아있는 막대 list의 합이 x보다 크지 않다면 (else) 막대의 절반 중 하나를 버리지 않는다.
                                    # (둘 다 막대 list에 추가되어야 함.) 
            ???

print(???)                          # 마지막으로 남은 막대기의 갯수를 출력.