input_path = "C:/RESEARCH/Mitophagy_data/N1/N1 Preprocessed/";
output_path = "C:/RESEARCH/Mitophagy_data/RGBdata/";

list = getFileList(input_path);

for(index = 0; index < list.length; index++)
{
	fName = list[index];
	open(input_path + fName);
	selectWindow(fName);
	run("RGB Color");
	saveAs("Tiff", output_path + "RGB" + fName);
	close();
}


