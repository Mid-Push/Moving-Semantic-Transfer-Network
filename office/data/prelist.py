fo=open('webcam_list.txt','rwb')
nc=open('webcam_few_list.txt','wb')
lines=fo.readlines()
for line in lines:
	items=line.split()
	if 'back_pack' in line:
		nc.write(items[0]+' '+'0'+'\n')
	if 'bike' in line:
		nc.write(items[0]+' '+'1'+'\n')
fo.close()
nc.close()

