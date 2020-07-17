%run('../vlfeat-0.9.20/toolbox/vl_setup')
run('/Applications/VLFEATROOT/toolbox/vl_setup.m')
load('pos_neg_feats.mat')


feats_train = cat(1, pos_feats_train, neg_feats_train);
labels_train = cat(1, ones(floor(0.8 * pos_nImages), 1), -1 * ones(floor(0.8 * neg_nImages), 1));

lambda = 0.0001;
[w,b] = vl_svmtrain(feats_train',labels_train',lambda);

fprintf('Classifier performance on train data:\n')
confidences_train = [pos_feats_train; neg_feats_train] * w + b;
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences_train, labels_train);


%Part_1_5: 
labels_valid = cat(1, ones(pos_nImages - (floor(0.8 * pos_nImages)), 1), -1 * ones(neg_nImages - (floor(0.8 * neg_nImages)), 1));
fprintf('Classifier performance on valid data:\n')
confidences_valid = [pos_feats_valid; neg_feats_valid] * w + b;
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences_valid, labels_valid);

save('my_svm.mat','w','b');
