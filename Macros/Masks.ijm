data_path = "C:/RESEARCH/Mitophagy_data/8bitData/";
mask_path = "C:/RESEARCH/Mitophagy_data/BinaryMasks/";
output_path = "C:/RESEARCH/Mitophagy_data/MaskedData/";
//imageCalculator("AND create stack", "BMN1CCCP_1C=0.tif","RGBN1CCCP_1C=0.tif");
//selectWindow("Result of BMN1CCCP_1C=0.tif");

dataList = getFileList(data_path);
maskList = getFileList(mask_path);


for(index = 0; index < dataList.length; index++){
	
	dataName = dataList[index];
	print("Masking " + dataName);

	mask_match = true;
	open(data_path + dataName);
	for(maskIndex = 0; maskIndex < maskList.length; maskIndex++){
		mask_name = maskList[maskIndex];
		//print(substring(dataName, 3));
		//print(substring(mask_name, 2));
		if(substring(dataName, 3) == substring(mask_name, 2)){
			mask = mask_name;
			mask_match = false;
			break;
			}
		}
	//print("match? " + mask_match);
	if (mask_match) {
		close("*");
		break;
	}
	open(mask_path + mask);
	imageCalculator("AND create stack", mask, dataName);
	selectWindow("Result of " + mask);
	saveAs("Tiff", output_path + substring(mask, 2));
	close("*");
	}
