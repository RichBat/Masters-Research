run("Bio-Formats Macro Extensions");
//Ext.setId("C:/RESEARCH/MEL_2020_CHOSEN/Original/Con001.czi")
input_path = "C:/RESEARCH/Mitophagy_data/Sholto Data Crunch/Mitophagy/Round 3/"
//Ext.setId("C:/RESEARCH/Mitophagy_data/N1/Con_3.czi")
list = getFileList(input_path);

close("*");
for(index = 0; index < list.length; index++)
{
	if(endsWith(list[index], ".czi")){
		run("Bio-Formats", "open=[" + input_path  + list[index] + "] color_mode=Default open_all_series rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
		List.setList(getMetadata("Info"));
		series = parseInt(List.get("Series"));
		if (series > 0) {
			parts = series + 1;
			for(p = 1; p <= parts; p++){
				selectWindow(input_path + list[index] + " - " + list[index].substring(0, list[index].lastIndexOf(".")) + " #" + p);
				Ext.setId(input_path + list[index])
				print(" ");
				Ext.getMetadataValue("Information|Image|RefractiveIndex #1",refractiveIndexImmersion)
				Ext.getMetadataValue("Information|Image|Channel|Wavelength #1",wavelength1)
				Ext.getMetadataValue("Information|Image|Channel|Wavelength #2",wavelength2)
				Ext.getMetadataValue("Information|Image|Channel|Wavelength #3",wavelength3)
				Ext.getMetadataValue("Information|Instrument|Objective|LensNA #1",NA)
				Ext.getMetadataValue("Scaling|Distance|Value #1",pixelsizeXY)
				getPixelSize(unit, pixelWidth, pixelHeight);
				//getVoxelSize(width, height, depth, unit);
				//width = getWidth();
				//height = getHeight();
				//depth = nSlices
				Ext.getMetadataValue("Scaling|Distance|Value #3",zStep)
				Ext.getMetadataValue(" SizeX",SizeX)
				Ext.getMetadataValue(" SizeY",SizeY)
				Ext.getMetadataValue(" SizeZ",SizeZ)
				Stack.getDimensions(width, height, channels, slices, frames);
				
				print(list[index] + " Series " + p);
				print("Refractive index immersion: " + refractiveIndexImmersion);
				print("Wavelength 1: " + wavelength1);
				print("Wavelength 2: " + wavelength2);
				print("Wavelength 3: " + wavelength3);
				print("NA: " + NA);
				print("Pixelsize XY: " + (parseFloat(pixelsizeXY)*1000000000.0));
				print("Pixelsize XY v2: " + pixelWidth);
				print("Z-step: " + (parseFloat(zStep)*1000000000.0));
				print("Size X " + width + " Size Y " + height + " Size Z " + slices);
				close();
				
			}	
		}
		else {

				//selectWindow(input_path + list[index]);
				Ext.setId(input_path + list[index])
				print(" ");
				Ext.getMetadataValue("Information|Image|RefractiveIndex",refractiveIndexImmersion)
				Ext.getMetadataValue("Information|Image|Channel|Wavelength #1",wavelength1)
				Ext.getMetadataValue("Information|Image|Channel|Wavelength #2",wavelength2)
				Ext.getMetadataValue("Information|Image|Channel|Wavelength #3",wavelength3)
				Ext.getMetadataValue("Information|Instrument|Objective|LensNA",NA)
				Ext.getMetadataValue("Scaling|Distance|Value #1",pixelsizeXY)
				getPixelSize(unit, pixelWidth, pixelHeight);
				//getVoxelSize(width, height, depth, unit);
				//width = getWidth();
				//height = getHeight();
				//depth = nSlices
				Ext.getMetadataValue("Scaling|Distance|Value #3",zStep)
				Ext.getMetadataValue(" SizeX",SizeX)
				Ext.getMetadataValue(" SizeY",SizeY)
				Ext.getMetadataValue(" SizeZ",SizeZ)
				Stack.getDimensions(width, height, channels, slices, frames);
				
				print(list[index]);
				print("Refractive index immersion: " + refractiveIndexImmersion);
				print("Wavelength 1: " + wavelength1);
				print("Wavelength 2: " + wavelength2);
				print("Wavelength 3: " + wavelength3);
				print("NA: " + NA);
				print("Pixelsize XY: " + (parseFloat(pixelsizeXY)*1000000000.0));
				print("Pixelsize XY v2: " + pixelWidth);
				print("Z-step: " + (parseFloat(zStep)*1000000000.0));
				print("Size X " + width + " Size Y " + height + " Size Z " + slices);
				close();
			
		}
		close("*");
	}
}