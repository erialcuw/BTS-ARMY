
function dataPath = get_data_path(run)
            rootDir = '/Users/clairewu/Documents/F22/BTS-ARMY';
            dataPath = strcat(rootDir, '/S', num2str(run));
        end
        
function imageData = get_image_data(run, scan_num)
    dataPath = get_data_path(run);
    imagePath = sprintf('%s/BTS_test_1_S%d_%05d.tif', dataPath, run, scan_num);
    imageData = imread(imagePath);
end

function intensity = get_avgI_of_max_CC(image, indices)
    intensity = zeros(indices);
    for i = 1:length(indices)
        intensity = image(indices);
    end
end
% function that uses the cells of the array 'indices', calculates 
% average grayscale intensity for the pixels in the largest connected
% component