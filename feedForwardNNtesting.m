clc
clear
load net

%we use the rest of the images in car and NoCar folder
%46 car image and 46 NoCar image

%error measure
TP=46;
FP=0;
TN=46;
FN=0;

%read car images and obtain the error measure

for i=109:154
%get the name of the image file
imageNumber=string(i);
imageName = sprintf('%06s',imageNumber);
filename=strcat("C:\Users\dell\Documents\MATLAB\CI\project2\dataset\car\",imageName,".png");
filename=char(filename);

%readimages 
getImage=imread(filename);

%convert the image to gray scale
getImage=rgb2gray(getImage);

%extract the image feature
[imageFeature,hogVisualization] = extractHOGFeatures(getImage);
featureVector=imageFeature;

%using trained network
featureVector = transpose(featureVector);
output=net(featureVector);

%estimated label
estimatedLabel=sign(output);

%calculate error
if(estimatedLabel~=label(i-108))
    TP=TP-1;
    FP=FP+1;
end
end

%read NoCar images and obtain the error measure

for i=109:154
%get the name of the image file
imageNumber=string(i);
imageName = sprintf('%06s',imageNumber);
filename=strcat("C:\Users\dell\Documents\MATLAB\CI\project2\dataset\NoCar\",imageName,".png");
filename=char(filename);

%read images 
getImage=imread(filename);

%convert the image to gray scale
getImage=rgb2gray(getImage);

%extract the image feature
[imageFeature,hogVisualization] = extractHOGFeatures(getImage);
featureVector=imageFeature;

%using trained network
featureVector = transpose(featureVector);
output=net(featureVector);

%estimated label
estimatedLabel=sign(output);

%calculate error
if(estimatedLabel~=label(i))
    TN=TN-1;
    FN=FN+1;
end
end

TP=TP/46;
FP=FP/46;
TN=TN/46;
FN=FN/46;