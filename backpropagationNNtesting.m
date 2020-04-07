load  backpropagationV
load  backpropagationW
%error measure
TP=46;
FP=0;
TN=46;
FN=0;

%read car images and obtain the error measure
input=ones(1765,1);
o=zeros(128,1);

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

    input(1:1764,1)=featureVector;

    %feed forward
    a=v*input;
    y=logsig(a);
    z=w*y;
    o(i-108)=logsig(z);
    
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

    input(1:1764,1)=featureVector;

    %feed forward
    a=v*input;
    y=logsig(a);
    z=w*y;
    o(i-108+64)=logsig(z);
   
end

%calculate error
for i=1:128
    
    if(i<=64)
        if(o(i)==0)
            TP=TP-1;
            FP=FP+1;
        end
    end
    
     if(i>64)
        if(o(i)==0330250004489925)
            TN=TN-1;
            FN=FN+1;
        end
     end

end
TP=TP/46;
FP=FP/46;
TN=TN/46;
FN=FN/46;