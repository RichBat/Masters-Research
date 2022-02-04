deconv_path = "C:/Users/richy/Desktop/Sholto Crops/Sholto's try/";
output_path = "C:/Users/richy/Desktop/Sholto Crops/Output/";

if(!File.exists(output_path))
	File.makeDirectory(output_path);

list = getFileList(deconv_path);

SF = 3;

close("*");

for(index = 0; index < list.length; index++)
{
	fName = list[index];

	print("Processing " + fName);

	if(endsWith(fName, ".tif") || endsWith(fName, ".tiff") )
	{
		run("Bio-Formats", "open=[" + deconv_path + fName + "] color_mode=Default rois_import=[ROI manager] split_timepoints view=Hyperstack stack_order=XYCZT");
		// open(deconv_path + fName);
	
		numFrames = nImages;
		for(i = 0; i < numFrames; i++)
		{
			print(i);
			selectImage(deconv_path + fName + " - T=" + i);
			//run("8-bit");
			run("Scale...", "x="+SF+" y="+SF+" z=1.0 interpolation=Bicubic average process create");
			close(deconv_path + fName + " - T=" + i); // close the old un-upscaled image
			print("  Scale");
			run("Subtract Background...", "rolling="+(6*SF)+" stack");
			print("  Subtract Background");
			run("Sigma Filter Plus", "radius="+(1*SF)+" use=2 minimum=0.2 outlier stack");
			print("  Sigma Filter Plus");
			run("Enhance Local Contrast (CLAHE)", "blocksize=64 histogram=1024 maximum=1.5 mask=*None* fast_(less_accurate)");
			print("  Enhance Local Contrast");
			run("Gamma...", "value=0.9 stack");
			print("  Gamma");
			//run("Enhance Contrast...", "saturated=0.3 normalize process_all");
			run("8-bit");
		}
		if(numFrames > 1){
			run("Concatenate...", "all_open open");
		}
		saveAs("Tiff", output_path + fName);
		
		close("*");
	}
}
