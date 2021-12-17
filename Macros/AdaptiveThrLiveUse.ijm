blockSize = 100;
subtract = 15;
for (i=1; i<=nSlices; i++){
	setSlice(i);
	run("adaptiveThr ", "using=[Weighted mean] from=" + blockSize + " then=-" + subtract + " slice");
	}