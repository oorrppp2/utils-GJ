"""
    파일 입출력
"""

# 2 7 11

# 2
# f = open('./data/input.txt', 'r')
# longest_word = ""
# lines = f.readlines()
# for i in range(len(lines)):
#     line = lines[i]
#     line_list = line.split(' ')
#     for word in line_list:
#         longest_word = word if len(word) > len(longest_word) else longest_word
# f.close()
# print("가장 긴 단어는 ", longest_word, "입니다.")

# 7

# while 1:
#     try:
#         file_name = ""
#         file_name = input("입력 파일 이름: ")
#         f = open("./data/"+file_name)
#         f.close()
#         print("파일이 성공적으로 열렸습니다.")
#         break
#     except IOError:
#         print("파일 "+file_name+" 이 없습니다. 다시 입력하시오.")

# 11
f = open("./data/prob_11.txt", 'r')
lines = f.readlines()
save_txt = ""
for i in range(len(lines)):
    save_txt += str(i+1) + ": " + lines[i]
f.close()

print(save_txt)

f = open("./data/prob_11.txt", 'wt')
f.write(save_txt)
f.close()

