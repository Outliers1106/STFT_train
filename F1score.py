def F1socre(pred_y,label):
	n = label.size
	AA = np.zeros(4,4)
	for i in range(n):
		if label[i] == 0:
			if pred_y[i] == 0:
				AA[0][0] = AA[0][0]+1
			elif pred_y[i] ==1:
				AA[0][1] = AA[0][1]+1
			elif pred_y[i] ==2:
				AA[0][2] = AA[0][2]+1
			elif pred_y[i] ==3:
				AA[0][3] = AA[0][3]+1
		elif label[i] == 1:
			if pred_y[i] == 0:
				AA[1][0] = AA[1][0]+1
			elif pred_y[i] ==1:
				AA[1][1] = AA[1][1]+1
			elif pred_y[i] ==2:
				AA[1][2] = AA[1][2]+1
			elif pred_y[i] ==3:
				AA[1][3] = AA[1][3]+1
		elif label[i] == 2:
			if pred_y[i] == 0:
				AA[2][0] = AA[2][0]+1
			elif pred_y[i] ==1:
				AA[2][1] = AA[2][1]+1
			elif pred_y[i] ==2:
				AA[2][2] = AA[2][2]+1
			elif pred_y[i] ==3:
				AA[2][3] = AA[2][3]+1
		elif label[i] == 3:
			if pred_y[i] == 0:
				AA[3][0] = AA[3][0]+1
			elif pred_y[i] ==1:
				AA[3][1] = AA[3][1]+1
			elif pred_y[i] ==2:
				AA[3][2] = AA[3][2]+1
			elif pred_y[i] ==3:
				AA[3][3] = AA[3][3]+1
	F1n=2*AA[0][0]/(sum(AA[0][:]))+sum(AA[:][0]);
	F1a=2*AA[1][1]/(sum(AA[1][:]))+sum(AA[:][1]);
	F1o=2*AA[2][2]/(sum(AA[2][:]))+sum(AA[:][2]);
	F1n=2*AA[3][3]/(sum(AA[3][:]))+sum(AA[:][3]);
	F1=(F1n+F1a+F1o+F1p)/4;
