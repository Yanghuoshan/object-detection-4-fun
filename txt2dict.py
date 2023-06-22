cnt = 1
item = dict()
List = []

with open('goods/Main/labels.txt','r') as f:
    for line in f.readlines():
        item[line.strip()]=cnt
        List.append(line.strip())
        cnt+=1

print(List)
print(item)