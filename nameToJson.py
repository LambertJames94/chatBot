file = open('names.txt', 'r')
f = file.readlines()

file2 = open('test2 - Copy.json', 'w')

newList = []
	
for line in f:
	if line[-1] == '\n':
		file2.write("{" + '"tag"' + ": " + '"' + line[:-1] + '"' + "," + '"patterns"' + ": " + "[" + '"' + line[:-1] + '"' + "]," + 
			'"responses"' + ": [" + '"Nice to meet you ' + line[:-1] + '."],' + '"context_set"' + ": " '"'+'"' + "},\n")
	else:
		file2.write("{" + '"tag"' + ": " + '"' + line + '"' + "," + '"patterns"' + ": " + "[" + '"' + line + '"' + "]," + 
		'"responses"' + ": [" + '"Nice to meet you ' + line + '."],' + '"context_set"' + ": " '"'+'"' + "}\n")
