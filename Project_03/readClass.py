classnum = []
classname = []
with open('classnames') as f:
    text = f.readlines()
    for line in text:
        if line.strip() == "": continue
        split = line.split(" ")
        classnum.append(int(split[1]))
        classname.append("{0} ".format(split[0])+" ".join(split[2:]).strip("\n"))
