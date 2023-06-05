classdef processing_functions2
    methods
        function dataPath = get_data_path(run)
            rootDir = '/Users/clairewu/Documents/F22/BTS-ARMY';
            dataPath = strcat(rootDir, '/S', num2str(run));
        end
        
        function imageData = get_image_data(run, scan_num)
            dataPath = get_data_path(run);
            imagePath = sprintf('%s/BTS_test_1_S%d_%05d.tif', dataPath, run, scan_num);
            imageData = imread(imagePath);
        end
    end
end