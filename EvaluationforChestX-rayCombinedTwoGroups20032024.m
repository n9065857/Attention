%% Load
A= 'C:\Users\zaena\OneDrive - Queensland University of Technology\Desktop\Group2\Test';
parentDir = A;
 dataDir = 'Test';
%%  Divide into Training and Validation Data

Test = imageDatastore(fullfile(parentDir,dataDir),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
 Test.ReadFcn = @customReadDatastoreImage;
 disp(['Number of training images: ',num2str(numel(Test.Files))]);
%  allImages.ReadFcn = @customReadDatastoreImage1;
%% Measure network accuracy
%% Measure network accuracy
predictedLabels = classify(XceptionChestXrayGroup1,TestGroup1); 
accuracy = mean(predictedLabels == TestLabelAllGroup1);
%% confusion metrix 
RE= TestGroup1.Labels;                      
cm = confusionmat(RE, predictedLabels);
[cm,order] = confusionmat(RE, predictedLabels,'Order',{'COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS'})
cm1= bsxfun (@rdivide, cm, sum(cm,2));
mean(diag(cm1))
cm2 = confusionchart(RE, predictedLabels);


%%
tp_m = diag(cm);

 for i = 1:2
    TP = tp_m(i);
    FP = sum(cm(:, i), 1) - TP;
    FN = sum(cm(i, :), 2) - TP;
    TN = sum(cm(:)) - TP - FP - FN;
    Po=(TP+TN)/(TP+TN+FP+FN);
    Ppositive=(TP+FP)*(TP+FN)/(TP+TN+FP+FN)^2;
    Pnegative=(FN+TN)*(FP+TN)/(TP+TN+FP+FN)^2;
    Pe=Ppositive+Pnegative;  
    Accuracy = (TP+TN)./(TP+FP+TN+FN);

    TPR = TP./(TP + FN);%tp/actual positive  RECALL SENSITIVITY
    if isnan(TPR)
        TPR = 0;
    end
    PPV = TP./ (TP + FP); % tp / predicted positive PRECISION
    if isnan(PPV)
        PPV = 0;
    end
    TNR = TN./ (TN+FP); %tn/ actual negative  SPECIFICITY
    if isnan(TNR)
        TNR = 0;
    end
    FPR = FP./ (TN+FP);
    if isnan(FPR)
        FPR = 0;
    end
    FScore = (2*(PPV * TPR)) ./ (PPV+TPR);

    if isnan(FScore)
        FScore = 0;
    end
    K= (Po- Pe)./ (1 - Pe);
    if isnan(K)
        K = 0;
    end
 end

fprintf('Accuracy = %2.2f.\n',100*Accuracy);
fprintf('RECALL or SENSITIVITY = %2.2f.\n',100*TPR);
fprintf('PRECISION = %2.2f.\n',100*PPV);
fprintf('SPECIFICITY = %2.2f.\n',100*TNR);
fprintf('F1-Score = %2.2f.\n',100*FScore);
fprintf('K = %2.2f.\n',100*K);

 

%%
function data=customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[299 299]);
end