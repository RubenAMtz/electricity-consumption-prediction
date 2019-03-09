function [] = tdmstocsv(path,save_path,columnstoconv,mic)
%path               is the path to find the tdms files
%save_path          is the path where the csv will be saved
%columnstoconv      is the number of files to concatenate in a single csv
%mic                is mic index (1 is for main mic, 2 is for environment
%                   mic)
files = dir(strcat(path,'\','*.tdms'));
files = extractfield(files,'name');
files = natsort(files)';
C = zeros(768000,1);
contador = 0;
for i= 1:length(files)
    [x]=convertTDMS(1,strcat(path,'\',files(i)));
    if C(1,1) == 0 %checks if its a clean variable
        C = x.Data.MeasuredData(mic+2).Data;
    else
        C = horzcat(C,x.Data.MeasuredData(mic+2).Data);
    end
    tamano = size(C); %looking for total columns
    if tamano(2)==columnstoconv || i == length(files) %in case there are less cols than expected
        contador = contador + 1; %to assign an index to the files
        csvwrite(strcat(save_path,'\','SonidoMic',num2str(mic),'_',num2str(contador),'.csv'),C);
        C = zeros(768000,1);
    end
end