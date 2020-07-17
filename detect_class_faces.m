%run('../vlfeat-0.9.20/toolbox/vl_setup')
run('/Applications/VLFEATROOT/toolbox/vl_setup.m')
load('my_svm.mat');

bboxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

cellSize = 6;
dim = 36;

scalesMultiple = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00];
nScale = size(scalesMultiple, 2);

% load and show the image
origIm = imread('class.jpg');
im = im2single(rgb2gray(origIm));

bboxesIm = zeros(0,4);
confidencesIm = zeros(0,1);
image_namesIm = cell(0,1);
for j=1:nScale
    scaleValue = scalesMultiple (j);
    imScaled = imresize (im, scaleValue);
    scalingFactor = 1/scaleValue;
    % generate a grid of features across the entire image. you may want to 
    % try generating features more densely (i.e., not in a grid)
    feats = vl_hog(imScaled, cellSize);
    % concatenate the features into 6x6 bins, and classify them (as if they
    % represent 36x36-pixel faces)
    [rows,cols,~] = size(feats);    
    confs = zeros(rows - cellSize + 1, cols - cellSize + 1);

    for r = 1 : rows - 5
        for c = 1 : cols - 5
            % create feature vector for the current window and classify it using the SVM model, 
            feats_vector = feats(r : r + cellSize - 1, c : c + cellSize - 1, :);
            % take dot product between feature vector and w and add b,
            % store the result in the matrix of confidence scores confs(r,c)
            confs(r,c) = feats_vector(:)' * w + b;
        end
    end

    % get the most confident predictions 
    [~,inds] = sort(confs(:),'descend');
    recalls = 22;      
    inds = inds(1:recalls); % (use a bigger number for better recall)
    for n=1:numel(inds)        
        [row,col] = ind2sub([size(confs,1) size(confs,2)],inds(n));
        bbox = [ col * cellSize * scalingFactor ...
                 row * cellSize * scalingFactor ...
                (col + cellSize - 1) * cellSize * scalingFactor ...
                (row + cellSize - 1) * cellSize * scalingFactor];
        conf = confs(row,col);
        if conf < 0.99
            continue
        end
        image_name = {imageList(i).name};

        % save
        bboxesIm = [bboxesIm; bbox];
        confidencesIm = [confidencesIm; conf];
        image_namesIm = [image_namesIm; image_name];
    end
end

% non-maximum suppression
[sc,si]=sort(-confidencesIm);
image_namesIm = image_namesIm(si);
bboxesIm = bboxesIm(si,:);
nd=length(confidencesIm);
valid_bboxes = zeros(nd,1);
for d=1:nd
    bb = bboxesIm(d,:);
    validCheck=1;
    for j=1:nd
        if (valid_bboxes(j) == 1)
            bbgt=bboxesIm(j,:);
            bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            if iw>0 && ih>0       
                % compute overlap as area of intersection / area of union
                areaA = (bb(3)-bb(1)+1)*(bb(4)-bb(2)+1);
                areaB = (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1);
                ia = iw*ih;
                ua = areaA + areaB - ia;
                ov=ia/ua;
                if ov >= 1.0
                    if areaA > areaB
                        validCheck = 0;
                    else
                        validCheck = 1;
                    end
                else
                   validCheck = 0;
                end
            end
        end
    end
    valid_bboxes(d) = validCheck;
end    

% show image
figure(1)
imshow(origIm);
hold on

% plot
for j = 1 : nd
    if (valid_bboxes(j) == 1)
        bbox = bboxesIm(j, :);
        plot_rectangle = [bbox(1), bbox(2); ...
            bbox(1), bbox(4); ...
            bbox(3), bbox(4); ...
            bbox(3), bbox(2); ...
            bbox(1), bbox(2)];
        plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
    end
end

% save         
bboxes = [bboxes; bboxesIm];
confidences = [confidences; confidencesIm];
image_names = [image_names; image_namesIm];
