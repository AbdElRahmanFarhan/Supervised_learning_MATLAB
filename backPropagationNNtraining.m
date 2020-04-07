tic
%create the feature vector will be used in training as an input
%the data set is 108 car images and 108 no car images
featureVector=zeros(1764,216);

%create the label vector
% -1 NoCar  1 Car
label=zeros(1,216);

%feature extraction car images
for samples=1:108
    
    %get the name of the image file
    imageNumber=string(samples);
    imageName = sprintf('%06s',imageNumber);
    filename=strcat("C:\Users\dell\Documents\MATLAB\CI\project2\dataset\car\",imageName,".png");
    filename=char(filename);
    
    %read the car images 
    getImage=imread(filename);
    
    %convert the image to gray scale
    getImage=rgb2gray(getImage);
 
    %extract the image feature
    [imageFeature,hogVisualization] = extractHOGFeatures(getImage);
    featureVector(:,samples)=imageFeature;
   
    %label the images as cars
    label(samples)=1;
    
end

%feature extraction for NoCar images
for samples=1:108
    
    %get the name of the image file
    imageNumber=string(samples);
    imageName = sprintf('%06s',imageNumber);
    filename=strcat("C:\Users\dell\Documents\MATLAB\CI\project2\dataset\NoCar\",imageName,".png");
    filename=char(filename);
    
    %read the car images 
    getImage=imread(filename);
    
    %convert the image to gray scale
    getImage=rgb2gray(getImage);
 
    %extract the image feature
    [imageFeature,hogVisualization] = extractHOGFeatures(getImage);
    featureVector(:,samples+108)=imageFeature;
    
    %label the images as NOcar
    label(samples+108)=0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%%NNtraining
%1 input layer number of neurons =1764
%1 hidden layer number of neurons =2
%1 output layer
%size of V is 1764*2
%size of W is 2*1

%weights initilizations
w=ones(1,3);
v=ones(3,1765);

%initilizations
x=ones(1765,1);
a=ones(3,1);
y=zeros(3,1);

%partial derv. initilizations
de_dw=zeros(1,3);
de_dv=zeros(3,1765);

%minimization variables
error=0;
epsilon=10^-3;
rate=1.5;
iterations=0;

while 1   
    iterations=iterations+1;
    for samples=1:216   
        
        x(1:1764,1)=featureVector(:,samples);
        %feed forward
        a(1:2,1)=v(1:2,:)*x;
        y=logsig(a);
        z=w*y;
        o=logsig(z);
        %error calculation
        error=0.5*(o-label(samples))^2;
        
        %back propagation  
        for i=1:3
            de_dw(1,i)=(o-label(samples))*((1-o)*o)*y(i);  
            %%%%%%update output weights              
            w(1,i)=w(1,i)-rate*de_dw(1,i);
                
            for j=1:1765  
                
                de_dv(i,j)=((1-o)*o)*w(1,i)*((1-y(i))*y(i))*x(j,1);
                %%%%%%update hiddern layer weights
                v(i,j)=v(i,j)-rate*de_dv(i,j);
            %%%%%%%%%                
            end  
        end
        
    end
    
    %condition we stop at
    if(error<=epsilon)
        break;
    end   
end  
toc