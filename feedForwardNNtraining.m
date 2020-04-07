%create the feature vector will be used in training as an input
%the data set is 108 car images and 108 no car images
featureVector=zeros(1764,216);

%create the label vector
% -1 NoCar  1 Car
label=zeros(1,216);

%feature extraction car images
for i=1:108
    
    %get the name of the image file
    imageNumber=string(i);
    imageName = sprintf('%06s',imageNumber);
    filename=strcat("C:\Users\dell\Documents\MATLAB\CI\project2\dataset\car\",imageName,".png");
    filename=char(filename);
    
    %read the car images 
    getImage=imread(filename);
    
    %convert the image to gray scale
    getImage=rgb2gray(getImage);
 
    %extract the image feature
    [imageFeature,hogVisualization] = extractHOGFeatures(getImage);
    featureVector(:,i)=imageFeature;
   
    %label the images as cars
    label(i)=1;
    
end

%feature extraction for NoCar images
for i=1:108
    
    %get the name of the image file
    imageNumber=string(i);
    imageName = sprintf('%06s',imageNumber);
    filename=strcat("C:\Users\dell\Documents\MATLAB\CI\project2\dataset\NoCar\",imageName,".png");
    filename=char(filename);
    
    %read the car images 
    getImage=imread(filename);
    
    %convert the image to gray scale
    getImage=rgb2gray(getImage);
 
    %extract the image feature
    [imageFeature,hogVisualization] = extractHOGFeatures(getImage);
    featureVector(:,i+108)=imageFeature;
    
    %label the images as NOcar
    label(i+108)=-1;
end

% create a neural network
net = feedforwardnet(10);

% train a neural network
net.divideParam.trainRatio= 1; % training set [%]
net.divideParam.valRatio= 0; % validation set [%]
net.divideParam.testRatio= 0; % test set [%]

[net,tr,Y,E] = train(net,featureVector,label);

% show network
view(net)
