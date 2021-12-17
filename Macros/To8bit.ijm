input_path = "C:/RESEARCH/Mitophagy_data/Autophagy/n2/PreProcessed/";
output_path = "C:/RESEARCH/Mitophagy_data/Autophagy/n2/PreProcessed/";

list = getFileList(input_path);

for(index = 0; index < list.length; index++)
{
	fName = list[index];
	open(input_path + fName);
	selectWindow(fName);
	run("8-bit");
	saveAs("Tiff", output_path + fName);
	close();
}


