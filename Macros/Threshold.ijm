input_path = "C:/RESEARCH/Mitophagy_data/3.Pre-Processed/";
output_path = "C:/RESEARCH/Mitophagy_data/4.Thresholded/";

if(!File.exists(output_path))
	File.makeDirectory(output_path);

list = getFileList(input_path);

SF = 3; //originally 3

blockSize = 100; //6 * SF
subtract = 50;
outlierRadius = 1*SF;

close("*");

for(index = 0; index < list.length; index++)
{
	fName = list[index];

	print("Processing " + fName);

	if(endsWith(fName, ".tif") || endsWith(fName, ".tiff") )
	{

		run("Bio-Formats", "open=[" + input_path + fName + "] color_mode=Default rois_import=[ROI manager] split_timepoints view=Hyperstack stack_order=XYCZT");
		// open(deconv_path + fName);
		numFrames = nImages;
		for(i = 0; i < numFrames; i++)
		{
			selectImage(input_path + fName + " - T=" + i);		
			run("8-bit");	
			numSlices = nSlices;
			for (s = 1; s <= numSlices; s++)
			{
				setSlice(s);
				run("adaptiveThr ", "using=[Weighted mean] from=" + blockSize + " then=-" + subtract + " slice");
			}
	
			run("Despeckle", "stack");
			run("Remove Outliers...", "radius=" + (outlierRadius) + " threshold=50 which=Bright stack");
			run("3D Fill Holes");
		}
		run("Concatenate...", "all_open open");
		saveAs("Tiff", output_path + fName);
		
		close("*");
		print("Finished with file");
	}
}




