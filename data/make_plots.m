clc 
clear
close all

files = dir("infer/");
fileNames = {files.name};
fileNames = fileNames(startsWith(fileNames, "infer_"));

for i = 1:length(fileNames)
    file_name = fileNames(i);
    file_path = strcat("infer/", file_name);

    plot_inferred(file_name, file_path);
end