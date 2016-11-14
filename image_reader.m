function img_data = image_reader(mode, img, show_img)
% image_reader either returns grayscale image data or red and green
% channel data, depending on the requested mode
%
% Usage:
%   img_data = image_reader
%   img_data = image_reader(mode)
%   img_data = image_reader(mode, img)
% Inputs:
%   mode = 'rg' (red and green board) or 'standard' (standard board)
%   img = image data (optional); if not provided, user can browse
%       or img can be a filename indicating which image to load
%   show_img = true to display image (default)
% Outputs:
%   img_data = structure with the following fields
%       .mode = string, 'rg' or 'standard'
%       .img_gray = filtered grayscale image
%       .fname = file name of image (if user browsed to file)
%       The following fields only apply to red/green mode
%       .img_r = filtered red channel binary image
%       .img_g = filtered green channel binary image

if nargin < 1 || isempty(mode)
    mode = 'rg'; % red and green board
end
if nargin < 2 || isempty(img)
    [fname, pname] = uigetfile(fullfile('Chess_pictures','*.jpg'));
    % load image with chess board
    img = imread(fullfile(pname, fname));
elseif ischar(img) % if input is a string, 
    fname = img;   % assume it's a filename
    img = imread(fname);
    [~, fname, ext] = fileparts(fname); % strip off path name
    fname = [fname, ext];
end
if nargin < 3
    show_img = true;
end
if ~exist('fname', 'var')
    fname = '';
end

img_data.mode = mode;
img_data.fname = fname;
img_data.img = img;

if show_img
    % display image
    figure('Color', 'w');
    imshow(img, 'InitialMagnification', 'fit');
end

% convert to grayscale
img_gray = rgb2gray(img);
img_gray = medfilt2(img_gray, [5, 5]); % median filter to remove noise
img_data.img_gray = img_gray;

if strcmp(mode, 'rg')
    % split image into red and green channels
    img_r = img(:, :, 1) - img(:, :, 2) - img(:, :, 3);
    img_r = medfilt2(img_r, [5, 5]); % median filter to remove noise
    thresh = graythresh(img_r);
    img_data.img_r = im2bw(img_r, thresh); % binarize red image
    img_g = img(:, :, 2) - img(:, :, 1) - img(:, :, 3);
    img_g = medfilt2(img_g, [5, 5]); % median filter to remove noise
    thresh = graythresh(img_g);
    img_data.img_g = im2bw(img_g, thresh); % binarize green image

    if show_img

    end

end


