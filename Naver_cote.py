def solution(n, m, survey):
    answer = ["-" for _ in range(m)]
    score = [0 for _ in range(m)]
    score_dict = {}
    grade_dict = {} # 'A': ['product1', 'product2', ...], ...

    for i in range(m):
        grade_dict[chr(ord('A')+i)] = []

    # Make grade list
    for ele in survey:
        name, grade = ele.split()
        try:
            score_dict[name][ord(grade) - ord('A')] += 1
        except:
            score_dict[name] = score.copy()
            score_dict[name][ord(grade) - ord('A')] += 1


    for ele in score_dict:
        best_score = 0
        best_score_index = -1
        for i in range(len(score_dict[ele])):
            if score_dict[ele][i] > best_score:
                best_score = score_dict[ele][i]
                best_score_index = i
        score_dict[ele][best_score_index] = 0
        grade_dict[chr(ord('A') + best_score_index)].append(ele)

    for i in range(m):
        c = chr(ord('A')+i)
        if len(grade_dict[c]) == 1:
            answer[i] = grade_dict[c][0]
        elif len(grade_dict[c]) > 1:
            # Set rank order
            rank = {}
            for ele in grade_dict[c]:
                rank[ele] = [c]
                while sum(score_dict[ele]) != 0:
                    best_score = 0
                    best_score_index = -1
                    for j in range(len(score_dict[ele])):
                        if score_dict[ele][j] > best_score:
                            best_score = score_dict[ele][j]
                            best_score_index = j
                    score_dict[ele][best_score_index] = 0
                    rank[ele].append(chr(ord('A') + best_score_index))

            max_len = 0
            for ele in rank:
                max_len = len(rank[ele]) if len(rank[ele]) > max_len else max_len

            for ele in rank:
                if len(rank[ele]) < max_len:
                    last_grade = rank[ele][0]
                    for m in range(len(rank[ele]), max_len):
                        rank[ele].append(last_grade)


            rank_list = []
            for ele in rank:
                rank_int = 0
                for ii in range(len(rank[ele])):
                    rank_int += (ord('A') - ord(rank[ele][ii])) * pow(10, max_len - ii)
                rank_list.append([rank_int, ele])


            rank_list.sort(key=lambda x:x[1])
            rank_list.sort(reverse=True, key=lambda x:x[0])

            ans_str = ""
            for r in rank_list:
                ans_str += r[1] + " "
            answer[i] = ans_str[:-1]

    return answer