net =XceptionChestXrayWithAttention;

%%
X = imread('NORMAL2-IM-0357-0001.png');
inputSize = net.Layers(1).InputSize(1:2);
X = imresize(X,inputSize);
%%
imshow(X);

%%
label = classify(net,X);
%%
scoreMap = gradCAM(net,X,label);
[classfn,score] = classify(net,X);
imshow(X);
title(sprintf("%s (%.2f)", classfn, score(classfn)));
map = gradCAM(net,img,classfn);
[YPred,scores] = classify(net,X);
YPred;

imshow(X);
hold on;
%label = classify(net,X);
imagesc(map,'AlphaData',0.5);
%colormap jet
hold on;

gradcamMap = gradCAM(net,img,YPred);
[YPred,scores] = classify(net,X);
YPred;
title(sprintf("%s (%.2f)", classfn, score(classfn)));
colormap jet
hold on

%%
ax(1) = subplot(1,2,1);
imshow(X)
%title("True Rotation = " + trueRotation + '\newline Pred Rotation = ' + round(predRotation,0))
colormap(ax(1),'gray');
ax(2) = subplot(1,2,2);
imshow(X)
hold on
imagesc(scoreMap,'AlphaData',0.5)
colormap(ax(2),'jet')
title(sprintf("%s (%.2f)", classfn, score(classfn)));
hold on