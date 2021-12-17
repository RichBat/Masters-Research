input_path = "C:/RESEARCH/Mitophagy_data/RACCdata/";
output_path = "C:/RESEARCH/Mitophagy_data/RACCdata/";

list = getFileList(input_path);
for(index = 0; index < list.length; index++){
	fName = list[index];
	open(input_path + fName);
	selectWindow(fName);
	run("RGB Color");
	saveAs("Tiff", output_path + fName);
	close("*");
	}