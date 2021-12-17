input_path = "C:/RESEARCH/Mitophagy_data/N1/N1 Deconvolved/";
coloc_path = "C:/RESEARCH/Mitophagy_data/N1/RACC Results/"
output_path = "C:/RESEARCH/Mitophagy_data/RACC Tests/test1/test_results/";

//Retrieve the colocalisation channels
old_coloc_list = getFileList(coloc_path);
cList = "";
for(a=0;a < old_coloc_list.length; a++){
	//print(old_coloc_list[a]);
	if(endsWith(old_coloc_list[a], ".tif")){
		if(cList == ""){
			cList = old_coloc_list[a];
			}
		else{
			cList = cList + "," + old_coloc_list[a];
			}
		}
	}
coloc_List = split(cList, ",");

function colocCheck(core_name) { 
	for(c = 0; c < coloc_List.length; c++){
		if(startsWith(coloc_List[c], core_name)){
			return "1 " + coloc_List[c];
			}
		}
	return "0 0";
}



close("*");
list = getFileList(input_path);
no_of_samples = list.length/2;
covered_files = newArray(no_of_samples);
for(counter = 0; counter < no_of_samples; counter++){
	covered_files[counter] = "";
	}

encountered_names = 0;
for(index = 0; index < list.length; index++){
	Channel0 = list[index];
	//print("\n" + Channel0);
	core_name = substring(Channel0, 0, Channel0.length - 7);
	unseen = true;
	//print("Number of encountered core names: " + encountered_names);
	for(checker = 0; checker < encountered_names; checker++){
		//print("Check for name comparison " + covered_files[checker] + " " + core_name);
		if(covered_files[checker] == core_name){
			unseen = false;
			}
		}
	if(unseen){
		//print("Unseen Flag: " + unseen);
		//print("Channel 0 and core name: " + Channel0 + " " + " " + core_name);
		covered_files[encountered_names] = core_name;
		//print("Recently added name: " + covered_files[encountered_names]);
		encountered_names++;
		second_channel = false;
		//print(core_name);
		result = split(colocCheck(core_name), " ");
		//print(result[0], result[1]);
		if(result[0] == 1){
			for(index2 = 0; index2 < list.length; index2++){
				Channel1 = list[index2];
				//print(Channel1 + " " + Channel0 + " " + substring(Channel1, 0, Channel0.length - 7) + " " + core_name);
				if(Channel1 != Channel0 && substring(Channel1, 0, Channel0.length - 7) == core_name){
					second_channel = true;
					break;
					}
				}
			if(second_channel){
				overlay(Channel0, Channel1, result[1], core_name);
				}
		}
	}
}

function overlay(ch1, ch2, coloc, core){
	//print("Files to be opened " + ch1 + " " + ch2 + " " + coloc);
	open(input_path + ch1);
	open(input_path + ch2);
	open(coloc_path + coloc);
	selectWindow(coloc);
	run("32-bit");
	run("Canvas Size...", "width=1024 height=1024 position=Center"); //Dimensions should be taken from ch1 dimensions

	selectWindow(ch1);
	run("Stack Slicer", "split_timepoints stack_order=XYCZT");

	selectWindow(ch2);
	run("Stack Slicer", "split_timepoints stack_order=XYCZT");
	
	run("Merge Channels...", "c1=[" + ch1 + " - T=0] c2=[" + ch2 + " - T=0] c4=" + coloc + " create");
	
	selectWindow("Composite");
	saveAs("Tiff", output_path + core + "0" + ".tif");
	close("*");
	}

//selectWindow("CCCP_10.tif"); //selects coloc window
//run("32-bit"); //convert to 32-bit (need compatible bit depth)
//run("Canvas Size...", "width=1024 height=1024 position=Center"); //Need the same dimensions (look into interpolation methds)
//run("Merge Channels...", "c1=[CCCP_1C=1.tif - T=0] c2=[CCCP_1C=0.tif - T=0] c4=CCCP_10.tif create"); //creates the composite image
//run("Stack to Hyperstack...", "order=xyczt(default) channels=3 slices=6 frames=1 display=Color"); //converts colour stack into a hyperstack so the channels can be viewed individually

//run("Stack Slicer", "split_timepoints stack_order=XYCZT"); //Time split for deconvolved channels
	