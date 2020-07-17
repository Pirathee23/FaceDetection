% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
mkdir(new_imageDir);
index=1;
dim = 36;

while n_have < n_want*2
    % generate random 36x36 crops from the non-face images
    img = imread(sprintf('%s/%s', imageDir, imageList(index).name));
    img_gray = rgb2gray(img);
    img_single = im2single(img_gray);
    [imgy, imgx] = size(img_single);
    
    if (index >= nImages)
        index = 1;
    else
        index = index + 1;
        fprintf('Image Index: %d/%d\n', index, nImages);
    end
    
    y = randi(imgy - dim + 1); 
    x = randi(imgx - dim + 1);
    n = y + (0 : dim - 1);
    m = x + (0 : dim - 1);
    
    negatives = img_single (n, m);
    imwrite(negatives, sprintf('%s/%d.jpg', new_imageDir, n_have), 'jpg');
    n_have = n_have + 1; 
end