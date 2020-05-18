function [fitnessvalue] = fitnessfcn(pop)
 
X1 = X_Train(:,pop);
NewY_Train=logical(Y_Train);
mdlSVM = fitcsvm(X1,NewY_Train,'Standardize',true);
CVSVMModel = crossval(mdlSVM);
classLoss = kfoldLoss(CVSVMModel); % 10 fold
% ----Compute the posterior probabilities (scores).
scoremdlSVM = fitPosterior(mdlSVM)
[~,score_svm] = resubPredict(scoremdlSVM);
% ----Compute the standard ROC curve using the scores from the SVM model.
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(NewY_Train,score_svm(:,scoremdlSVM.ClassNames),'true');
% -----Plot the ROC curves.
hold on
plot(Xsvm,Ysvm)
legend('Support Vector Machines','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curve for SVM')
hold off
[label,score]=predict(scoremdlSVM,X_Test(:,SelectedFeatures));

end