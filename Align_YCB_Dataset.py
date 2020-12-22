import numpy as np

r = open('/home/user/python_projects/DenseFusion/datasets/ycb/dataset_config/test_data_list.txt', mode='rt', encoding='utf-8')

save_path_txt = ""
cnt = 0
splitlinestr = str.splitlines(r.read())
save_txt = ""
for str_line in splitlinestr:
    save_line = ""

    if str_line[4] == "/":
        save_line += str_line + "-color.png\t"
        save_line += str_line + "-label.png\n"
    else:
        str_line = str_line[:8] + str_line[10] + str_line[8:]
        save_line += str_line + "-color.png\t"
        save_line += str_line + "-label.png\n"

        # if str_line[11] == "7":
        #     print(save_txt)
        #     cnt += 1
    save_txt += save_line

    # if cnt > 10:
    #     break
    pass

r.close()

ff = open("valid.ycb", mode='wt')
ff.write(save_txt)
ff.close()
