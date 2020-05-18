clc
clear all
close all
% clear memory
[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick an Image File');
I = imread([pathname,filename]);
figure, imshow(I); title('Input Image');
I = imresize(I,[200,200]);
% Convert to grayscale
gray = rgb2gray(I);
% Otsu Binarization for segmentation
level = graythresh(I);
img = im2bw(I,level);
figure, imshow(img);title('Otsu Thresholded Image');
cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 1;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',1);
pixel_labels = reshape(cluster_idx,nrows,ncols);
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1,1,3]);
for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end
figure, imshow(segmented_images{1});title('Objects');
seg_img = im2bw(segmented_images{1});
figure, imshow(seg_img);title('Segmented Image');

x = double(seg_img);
m = size(seg_img,1);
n = size(seg_img,2);

signal1 = seg_img(:,:);

[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);
whos DWT_feat
whos G
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
%Skewness = skewness(img)
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
% Inverse Difference Movement
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    CostFunction=@(DTM) Sphere(DTM);      % Cost Function
% CostFunction = DDM;

nVar = 10;          % Number of Decision Variables

VarSize=[1 nVar];   % Decision Variables Matrix Size

VarMin=-10;         % Decision Variables Lower Bound
VarMax= 10;         % Decision Variables Upper Bound

%% Cultural Algorithm Settings

MaxIt=1000;         % Maximum Number of Iterations

nPop=50;            % Population Size

pAccept=0.35;                   % Acceptance Ratio
nAccept=round(pAccept*nPop);    % Number of Accepted Individuals

alpha=0.3;

beta=0.5;

%% Initialization

% Initialize Culture
Culture.Situational.Cost=inf;
Culture.Normative.Min=inf(VarSize);
Culture.Normative.Max=-inf(VarSize);
Culture.Normative.L=inf(VarSize);
Culture.Normative.U=inf(VarSize);

% Empty Individual Structure
empty_individual.Position=[];
empty_individual.Cost=[];

% Initialize Population Array
pop=repmat(empty_individual,nPop,1);

% Generate Initial Solutions
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost=CostFunction(pop(i).Position);
end

% Sort Population
[~, SortOrder]=sort([pop.Cost]);
pop=pop(SortOrder);

% Adjust Culture using Selected Population
spop=pop(1:nAccept);
aCulture=AdjustCulture(Culture,spop);

% Update Best Solution Ever Found
BestSol=Culture.Situational;

% Array to Hold Best Costs
BestCost=zeros(MaxIt,1);

%% multi object Cultural Algorithm Main Loop

for it=1:MaxIt
                                                                     
    % Influnce of Culture
    for i=1:nPop
        
        % % 1st Method (using only Normative component)
%         sigma=alpha*Culture.Normative.Size;
%         pop(i).Position=pop(i).Position+sigma.*randn(VarSize);
        
        % % 2nd Method (using only Situational component)
        for j=1:nVar
           sigma=0.1*(VarMax-VarMin);
           dx=sigma*randn;
%            if pop(i).Position(j)<Culture.Situational.Position(j)
%                dx=abs(dx);
%            elseif pop(i).Position(j)>Culture.Situational.Position(j)
%                dx=-abs(dx);
%            end
%            pop(i).Position(j)=pop(i).Position(j)+dx;
        end
        
        % % 3rd Method (using Normative and Situational components)
%         for j=1:nVar
%           sigma=alpha*Culture.Normative.Size(j);
%           dx=sigma*randn;
%           if pop(i).Position(j)<Culture.Situational.Position(j)
%               dx=abs(dx);
%           elseif pop(i).Position(j)>Culture.Situational.Position(j)
%               dx=-abs(dx);
%           end
%           pop(i).Position(j)=pop(i).Position(j)+dx;
%         end        
        
        % % 4th Method (using Size and Range of Normative component)
%         for j=1:nVar
%           sigma=alpha*Culture.Normative.Size(j);
%           dx=sigma*randn;
%           if pop(i).Position(j)<Culture.Normative.Min(j)
%               dx=abs(dx);
%           elseif pop(i).Position(j)>Culture.Normative.Max(j)
%               dx=-abs(dx);
%           else
%               dx=beta*dx;
%           end
%           pop(i).Position(j)=pop(i).Position(j)+dx;
%         end        
        
        pop(i).Cost=CostFunction(pop(i).Position);
        
    end
    
    % Sort Population
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);

    % Adjust Culture using Selected Population
    spop=pop(1:nAccept);
    Culture=AdjustCulture(Culture,spop);

    % Update Best Solution Ever Found
    BestSol=Culture.Situational;
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end

feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];

% load t.mat
load tweet.mat
%  xdata = meas;
meas = dataset;
xdata = meas;
  group = label1;
 SVMModel = fitcsvm(meas,group,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
CVSVMModel = crossval(SVMModel);
CVSVMModel = crossval(SVMModel);
[~,scorePred] = kfoldPredict(CVSVMModel);
outlierRate = mean(scorePred<0)
%   svmStruct1 = fitcsvm(xdata,group);
%   species = ClassificationSVM(svmStruct1,feat,'showplot',false);
%   Accuracy = Evaluate(img,seg_img)
% Accuracy = Evaluate(cH3,cD3)

sv = SVMModel.SupportVectors;
figure
gscatter(xdata(:,1),xdata(:,2),group)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',0.01)
legend('versicolor','virginica','Support Vector')
hold off
d = 0.02;
Mdl2 = fitcsvm(meas,group,'KernelFunction','rbf');
Accuracy = outlierRate(:,2)*100
% Mdl2 = fitcsvm(meas,label,'KernelFunction','mysigmoid2','Standardize',true);
% [~,scores2] = predict(Mdl2,xGrid);
[x1Grid,x2Grid] = meshgrid(min(meas(:,1)):d:max(meas(:,1)),...
    min(meas(:,2)):d:max(meas(:,2)));
xGrid = [x1Grid(:),x2Grid(:)]; 
% [~,scores2] = predict(Mdl2,xGrid);
figure;
h(1:2) = gscatter(meas(:,1),meas(:,2),group);
hold on
h(3) = plot(meas(SVMModel.IsSupportVector,1),meas(SVMModel.IsSupportVector,2),'ko','MarkerSize',10);
title('Scatter Diagram with the Decision Boundary')
hold off
BestCost=imresize(BestCost,[29083 1]);
% scatter(BestCost,rankOfOccurrences', 100, rankIndex', 'filled');
[Xsvm,Ysvm,AUCsvm] = perfcurve(label1,scorePred(:,1),'REAL');
figure,plot(Xsvm,Ysvm);
[prec, tpr, fpr, thresh]=prec_rec(Xsvm,Ysvm);
 Precision=mean2(prec)
 Recall = mean2(tpr)
% f_measure = 2*((prec1*tpr1)/(prec1 + tpr1))
f_measure = 2*((Precision*Recall)/(Precision + Recall))