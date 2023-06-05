%% Testing single image
clc; clear; close all;

X = double(imread('BTS_test_1_S633_00000.tif'));

figure;imagesc(X);colorbar

figure;histogram(X) % x-axis intensity values

thresh = 0.3 * max(max(X)); %thresholding to obtain binary image
Xbw = imbinarize(X,thresh);

figure;imagesc(Xbw)

SE = strel("disk",1);
Xbw1 = imdilate(Xbw,SE);
%Xbw1_ROI_1 = Xbw1240:440, 690:840);

figure;imagesc(Xbw1)

CC = bwconncomp(Xbw1);

numOfPixels = cellfun(@numel,CC.PixelIdxList);
[~,indexOfMax] = max(numOfPixels);
indices = CC.PixelIdxList(indexOfMax);
indices = indices{1};

im = zeros(size(X));
im(indices) = 1;

average_I = mean(X(indices));
max_I = max(X(indices));

figure;imagesc(im)

props = regionprops(CC, X, 'all');
numberOfBlobs = numel(props);
%%
figure; imhist(X)
%figure; plot(X(313.5:761.5), 'o')
% [as,~,index] = unique(indices);
% f = fit(index, X(indices), 'gauss2');
% figure; plot(f, index, X(indices))
%% Iterating over multiple images
clc; clear; close all;

values = zeros(45,21);
for i = 1:45 %heating

    run = i + 631;
    output_dir = fullfile(get_data_path(run), 'output_images');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    for j = 1:21
        scan_num = j-1;
        X = double(get_image_data(run, scan_num));
        %X = double(imread('BTS_test_1_S257_00000.tif'));
    
        %figure;imagesc(X);colorbar
        
        %figure;histogram(X)
        
        thresh = 0.3 * max(max(X));
        Xbw = imbinarize(X,thresh);
        
        %figure;imagesc(Xbw)
        
        SE = strel("disk",1);
        Xbw1 = imdilate(Xbw,SE); %fills in surrounding pixels to match
        
        %figure;imagesc(Xbw1)
        % to obtain 2 small things on the right, increase dilation, decrease threshold
    
        CC = bwconncomp(Xbw1);
        
        numOfPixels = cellfun(@numel, CC.PixelIdxList); % number of connected components, each cell is # of pixels
        [~,indexOfMax] = max(numOfPixels);
        indices = CC.PixelIdxList(indexOfMax);
        indices = indices{1};
        
        im = zeros(size(X));
        im(indices) = 1;
        
        %figure;imagesc(im)        
        values(i,j) = max(X(indices));
        
        
%         outputFile = sprintf('%s/BTS_test_1_S%d_%05d.png', output_dir, run, scan_num);
%         saveas(gcf, outputFile, 'png');
        
% AVERAGE INTENSITY OF ROI AREA from EACH IMAGE        
    end
end

%% Plotting
close all;
output = max(values, [], 2); % 45 vs. 1
figure; plot(output, 'o') %x-axis is folder number

%figure; plot(values(1,:), 'o')
x_val = transpose(double(uint32(1)):double(uint32(21)));
y_val = transpose(values(1,:));
figure; plot(x_val, y_val, 'o')

f = fit(x_val, y_val, 'gauss2');
figure; plot(f, x_val, y_val)
%extract max I value + corresponding pixel coords -> convert to 2theta
%% Functions
function dataPath = get_data_path(run)
            rootDir = '/Users/clairewu/Documents/F22/BTS-ARMY/Heating';
            dataPath = strcat(rootDir, '/S', num2str(run));
        end
        
function imageData = get_image_data(run, scan_num)
    dataPath = get_data_path(run);
    imagePath = sprintf('%s/BTS_test_1_S%d_%05d.tif', dataPath, run, scan_num);
    imageData = imread(imagePath);
end

% function that converts pixel coords to 2 theta, 1 pixel = .0099 degrees 
% do we want coords of the centroid of CC or the coords of the brightest
% pixel
