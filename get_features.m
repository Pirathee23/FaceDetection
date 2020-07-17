close all
clear
%run('../vlfeat-0.9.20/toolbox/vl_setup')
run('/Applications/VLFEATROOT/toolbox/vl_setup.m')

pos_imageDir = 'cropped_training_images_faces';
pos_imageList = dir(sprintf('%s/*.jpg',pos_imageDir));
pos_nImages = length(pos_imageList);

neg_imageDir = 'cropped_training_images_notfaces';
neg_imageList = dir(sprintf('%s/*.jpg',neg_imageDir));
neg_nImages = length(neg_imageList);

cellSize = 6;
featSize = 31*cellSize^2;
pos_feats = zeros(pos_nImages*2,featSize);

for i=1:pos_nImages
    im = im2single(imread(sprintf('%s/%s',pos_imageDir,pos_imageList(i).name)));
    imFlip = fliplr(im);

    feat = vl_hog(im,cellSize);
    pos_feats(i,:) = feat(:);
    fprintf('got feat for pos image %d/%d\n',i,pos_nImages);
    pos_feats_flip = vl_hog(imFlip, cellSize);
    pos_feats(pos_nImages + i,:) = pos_feats_flip(:);
end

neg_feats = zeros(neg_nImages,featSize);
for i=1:neg_nImages
    im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    neg_feats(i,:) = feat(:);
    fprintf('got feat for neg image %d/%d\n',i,neg_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

pos_nImages = pos_nImages*2;

%Split training images into training set and validation set
pos_feats_train = pos_feats(1 : floor(0.8 * pos_nImages), :);
neg_feats_train = neg_feats(1 : floor(0.8 * neg_nImages), :);
pos_feats_valid = pos_feats(1 + floor(0.8 * pos_nImages) : pos_nImages, :);
neg_feats_valid = neg_feats(1 + floor(0.8 * neg_nImages) : neg_nImages, :);

save('pos_neg_feats.mat','pos_feats','neg_feats','pos_nImages','neg_nImages', 'pos_feats_train', 'neg_feats_train', 'pos_feats_valid', 'neg_feats_valid')