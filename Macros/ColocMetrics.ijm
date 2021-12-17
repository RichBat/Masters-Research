specimens = newArray("N1CCCP_1C", 
							"N1CCCP_2C",
							"N1CCCP+Baf_2C",
							"N1CCCP+Baf_3C",
							"N1Con_2C",
							"N1Con_3C",
							"N2CCCP+Baf_1C",
							"N2CCCP+Baf_4C",
							"N2Con_3C",
							"N2Con_4C",
							"N2Rapa+CCCP+Baf_1C",
							"N3CCCP_4C",
							"N3CCCP+Baf_2C",
							"N3Con_1C",
							"N4CCCP_2C",
							"N4CCCP+BafC",
							"N4Con_1C");
RGBspecimens = newArray("RGBN1CCCP_1C", 
							"RGBN1CCCP_2C",
							"RGBN1CCCP+Baf_2C",
							"RGBN1CCCP+Baf_3C",
							"RGBN1Con_2C",
							"RGBN1Con_3C",
							"RGBN2CCCP+Baf_1C",
							"RGBN2CCCP+Baf_4C",
							"RGBN2Con_3C",
							"RGBN2Con_4C",
							"RGBN2Rapa+CCCP+Baf_1C",
							"RGBN3CCCP_4C",
							"RGBN3CCCP+Baf_2C",
							"RGBN3Con_1C",
							"RGBN4CCCP_2C",
							"RGBN4CCCP+BafC",
							"RGBN4Con_1C");

thresholds = newArray("40,25","18,10","14,19","11,20","26,23","30,18","15,20","12,12","25,25","23,23",
"25,25","15,20","25,25","30,30","15,15","15,15","15,20");
RGBthresholds = newArray("1,1","1,1","1,1","1,1","1,1","1,1","1,1","1,1","1,1","1,1",
"1,1","1,1","1,1","1,1","1,1","1,1","1,1");

input_path = "C:/RESEARCH/Mitophagy_data/MaskedData/";
RGBinput_path = "C:/RESEARCH/Mitophagy_data/8bitData/";

output_path = "C:/RESEARCH/Mitophagy_data/RACCdata/";
running_fine = true;
if(specimens.length != thresholds.length){
	print("Number of specimens: ", specimens.length);
	print("Number of thresholds: ", thresholds.length);
	print("Number of specimens and thresholds do not match");
	running_fine = false;
	}
if(running_fine){
	list_of_bad_thresholds = "";
	for(i = 0; i < specimens.length; i++){
		threshold_values = split(thresholds[i], ",");
		if(threshold_values.length == 2){
			imgA = specimens[i] + "=0.tif";
			imgB = specimens[i] + "=1.tif";
			
			open(input_path + imgA);
			selectWindow(imgA);
			run("Stack Slicer", "split_timepoints stack_order=XYCZT");
			selectWindow(imgA + " - T=0");
			saveAs("Tiff", output_path + imgA);
			imgA = imgA + " - T=0";
			
			open(input_path + imgB);
			selectWindow(imgB);
			run("Stack Slicer", "split_timepoints stack_order=XYCZT");
			selectWindow(imgB + " - T=0");
			saveAs("Tiff", output_path + imgB);
			imgB = imgB + " - T=0";

			run("JACoP ", "imga=" + imgA + " imgb=" + imgB + " thra=" + threshold_values[0] + " thrb=" + threshold_values[1] + " pearson overlap mm");
			close("*");
			}
		else{
			list_of_bad_thresholds = list_of_bad_thresholds + specimens[i] + "\n";
			}
		}
		print("Specimens with more or less thresholds: ", list_of_bad_thresholds);
}