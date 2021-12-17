from os import listdir
from os.path import isfile, join
import copy
import json

psf_parameters = {"Parameters":{},
                  "Sets":{}
                  }
psfs = {}


metadata_path = "C:\\RESEARCH\Mitophagy_data\\PSF Metadata\\sholto_mitophagy\\"
#print(metadata_files)
variant = 0
def psf_check(parameters, set, name):
    global variant
    if not psfs:
        psfs["Variant " + str(variant)] = copy.deepcopy(psf_parameters)
        psfs["Variant " + str(variant)]["Parameters"] = parameters
        psfs["Variant " + str(variant)]["Sets"][set].append(name)
        return True
    else:
        for variation, contents in psfs.items():
            if contents["Parameters"] == parameters:
                psfs[variation]["Sets"][set].append(name)
                return True
        variant += 1
        psfs["Variant " + str(variant)] = copy.deepcopy(psf_parameters)
        psfs["Variant " + str(variant)]["Parameters"] = parameters
        psfs["Variant " + str(variant)]["Sets"][set].append(name)
        return False


if __name__ == '__main__':
    metadata_files = [metadata_path + f for f in listdir(metadata_path) if isfile(join(metadata_path, f))]
    for files in metadata_files:
        filename = files.split(sep=metadata_path)[1]
        psf_parameters["Sets"][filename] = []
    for file in metadata_files:
        file_ = open(file, "r")
        file_contents = file_.read()
        samples = file_contents.split(sep=" \n")
        filename = file.split(sep=metadata_path)[1]
        #psf_parameters["Sets"][filename] = []
        del samples[0]
        for s in samples:
            parameters = s.split(sep="\n")
            sample_name = parameters.pop(0)
            del parameters[-1]
            param_values = {}
            for p in parameters:
                if ":" in p:
                    param_name = p.split(sep=":")[0]
                    param_values[param_name] = p.split(sep=":")[1]
                else:
                    param_name = p.split(sep=" ")
                    for i in range(2, len(param_name), 3):
                        param_values[param_name[i-2] + " " + param_name[i-1]] = param_name[i]
            seen = psf_check(param_values, set=filename, name=sample_name)
        file_.close()
    json = json.dumps(psfs)
    to_save = open(metadata_path + "PSF Compilation.json", "w")
    to_save.write(json)
    to_save.close()




