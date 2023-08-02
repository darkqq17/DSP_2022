# coding: Big5-HKSCS
import sys

with open(sys.argv[1], 'r', encoding='big5-hkscs') as f:
    lines = f.readlines()

zhuyin_list = ['�t', '�u', '�v', '�w', '�x', '�y', '�z', '�{', '�|', '�}', '�~', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��']
inverse_lines1 = [zhuyin_list[i] + "   " for i in range(len(zhuyin_list))]        # �o�Ӧs37�Ӫ`����inverse
inverse_lines2 = [line[0] + "   " + line[0] for line in lines]        # �o�Ӧsbig5 to big5�C�Ӧr
inverse_str = []
for line in lines:
    zhuyin_for_word = line[2:].split('/')           # �`����list(���]�t��r)
    for i in range(len(zhuyin_list)):               # ���C�Ӫ`��
        for z in zhuyin_for_word:                   # ���C�ӯ}�����k���r�� 
            if z[0]==zhuyin_list[i]:
                inverse_lines1[i] += (line[0] + " ")
                break


# �����S�������r���`��
for i in range(len(inverse_lines1)):
    if i==len(inverse_lines1):
        break
    if len(inverse_lines1[i])==4:
        inverse_lines1.pop(i)

final_inverse_line = inverse_lines1 + inverse_lines2

for i in range(len(final_inverse_line)):
    final_inverse_line[i] += '\n'

with open(sys.argv[2], 'w', encoding='big5-hkscs') as f2:
    f2.writelines(final_inverse_line)