psf_path = "C:/RESEARCH/Mitophagy_data/Sholto Data Crunch/Mitophagy/PSF/"

ch0 = "561";
ch1 = "488";
ch2 = "633";

input_path = "C:/RESEARCH/Mitophagy_data/1.MitoTimeLapse/";
output_path = "C:/RESEARCH/Mitophagy_data/2.Deconvolved/";

if(!File.exists(output_path))
	File.makeDirectory(output_path);

list = getFileList(input_path);

close("*");
for(index = 0; index < list.length; index++)
{
	fName = list[index];
	print("Processing " + fName);
	print("Index " + index);
	if(endsWith(fName, ".tif") || endsWith(fName, ".tiff") || endsWith(fName, ".czi"))
	{
		//run("Bio-Formats Importer", "open=["input_path + fName"]", "split_channels color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");

		run("Bio-Formats", "open=[" + input_path  + fName+ "] color_mode=Default open_all_series rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
		List.setList(getMetadata("Info"));
		series = parseInt(List.get("Series"));
		if(series == 0){
			selectWindow(input_path + fName + " - C=0");
			nFrames = List.get("SizeT");
			nChannels = List.get("SizeC");
			print(nChannels);
			for(j = 0; j < nChannels; j++){
				print("channel " + j);
				psf_file = psf_select(j, nSlices);
				print(fName);
				print(psf_file);
				concat_name = "";
				selectWindow(input_path + fName + " - C=" + j);
				run("Stack Slicer", "split_timepoints stack_order=XYCZT");
				//numFrames = nImages;
				for(i = 0; i < nFrames; i++)
				{
						selectImage(input_path + fName + " - C=" + j + " - T=" + i);
						prevn = nImages;
						deconvTitle = "T=" + i + "-C=" + j + "_deconv";
						concat_name += " image"+(i+1)+"=[" + deconvTitle + "]";
						run("DeconvolutionLab2 Run", " -image platform  "+ input_path + fName + " - C=" + j + " - T=" + i + " -psf file "+ psf_file + " -algorithm RLTV 20 1.000E-05 -out stack " + deconvTitle + " nosave");	
						while(nImages == prevn)
							wait(50);
						selectImage(input_path + fName + " - C=" + j + " - T=" + i);
						close();	
				}
				print("Name to Concatenate " + concat_name);
				if(nFrames > 1){ 
					run("Concatenate...", " title=Channel" + j + " open " + concat_name);
					selectWindow("Channel" + j);
					saveName = 
				}
				else {
					selectWindow("T=0" + "-C=" + j + "_deconv");
				}
				saveAs("Tiff", output_path + fName.substring(0, fName.lastIndexOf(".")) + "C=" + j + ".tif");
			}
		}
		else {
			List.setList(getMetadata("Info"));
			parts = series + 1;
			print(parts);
			for(p = 1; p <= parts; p++){
				print("Made it");
				nFrames = List.get("Information|Image|SizeT");
				nChannels = List.get("Information|Image|SizeC");
				//print(List.get("Information|User|DisplayName"));
				//print(List.getList);
				//print("stuff " + nChannels.substring(nChannels.lastIndexOf(" "), nChannels.length));
				channels = nChannels.substring(nChannels.lastIndexOf(" "), nChannels.length);
				for(j = 0; j < channels; j++){
					print("channel " + j);
					psf_file = psf_select(j, nSlices);
					print(fName);
					print(psf_file);
					concat_name = "";
					selectWindow(input_path + fName + " - " + fName.substring(0, fName.lastIndexOf(".")) + " #" + p + " - C=" + j);
					//numFrames = nImages;
					if(nFrames != ""){
						run("Stack Slicer", "split_timepoints stack_order=XYCZT");
						print("Oops");
						for(i = 0; i < nFrames; i++)
						{
								selectImage(input_path + fName + " - C=" + j + " - T=" + i);
								prevn = nImages;
								deconvTitle = "T=" + i + "-C=" + j + "_deconv";
								concat_name += " image"+(i+1)+"=[" + deconvTitle + "]";
								run("DeconvolutionLab2 Run", " -image platform  "+ input_path + fName + " - " + fName.substring(0, fName.lastIndexOf(".")) + " #" + p + " - T=" + i +  " - C=" + j + " -psf file "+ psf_file + " -algorithm RLTV 20 1.000E-05 -out stack " + deconvTitle + " nosave");	
								while(nImages == prevn)
									wait(50);
								selectImage(input_path + fName + " - " + fName.substring(0, fName.lastIndexOf(".")) + " #" + p + " - C=" + j + " - T=" + i);
								close();	
						}
					}
					else{
						print("Fish");
						selectImage(input_path + fName + " - " + fName.substring(0, fName.lastIndexOf(".")) + " #" + p + " - C=" + j);
						deconvTitle = "T=0-C=" + j + "_deconv";
						prevn = nImages;
						run("DeconvolutionLab2 Run", " -image platform  "+ input_path + fName + " - " + fName.substring(0, fName.lastIndexOf(".")) + " #" + p + " - C=" + j + " -psf file "+ psf_file + " -algorithm RLTV 20 1.000E-05 -out stack " + deconvTitle + " nosave");
						while(nImages == prevn)
							wait(50);
						selectImage(input_path + fName + " - " + fName.substring(0, fName.lastIndexOf(".")) + " #" + p + " - C=" + j);
						close();	
						}
					print("Name to Concatenate " + concat_name);
					if(nFrames > 1){ 
						run("Concatenate...", " title=Channel" + j + " open " + concat_name);
						selectWindow("Channel" + j);
					}
					else {
						selectWindow("T=0" + "-C=" + j + "_deconv");
					}
					saveAs("Tiff", output_path + fName.substring(0, fName.lastIndexOf(".")) + "Series=" + p + " C=" + j + ".tif");
					close();
				}
			}
		}
		close("*");
		}
		
	}

function psf_select(channel, z){
	psf_returning = "";
	chan = "";
	if(channel == 0){
		chan = ch0;
		}
	else if(channel == 1){
		chan = ch1;
		}
	else{
		chan = ch2;
		}
	psf_returning = psf_path + chan + "Z" + z + ".tif";
	print(psf_returning);
	return psf_returning;
}

