function [squares,hedge] = board_detector(img_r, img_g, fname)
% board_detector takes an input image and returns an 8x8 array of squares

% Inputs:
%   img = an image (previously loaded using imread), either color or bw
%       if no input image is provided, user can browse for image file

% Outputs:
%   squares = 8x8 array of structures with the following fields
%       .corners = 4x2 array of pixel locations of square corners [x, y]
%   bw = optional output; binarized version of image

% 1) binarize image
% 2) find edges of everything in image
% 3) use hough transform to find straight lines
% 4) use distance and angle to identify board grid lines

squares = []; % initialize output argument
hedge=[];
% load image and do pre-processing, if needed

img_size = size(img_r);

%%
params.edge_thresh = []; % use MATLAB default
params.edge_sigma = 3;
params.n_peaks = 18;
params.hpeak_pct = 0.2; % percentage of maximum value used to find peaks in 
                    % output of hough transform; default is 0.2

% use canny edge detector
edges_r = edge(img_r, 'canny');
edges_g = edge(img_g, 'canny');
edg_img = edges_r | edges_g; % combine red and green edges
% % close gap between them
% edg_img2 = imclose(edg_img, strel('disk', 3));
% edg_img3 = edg_img2 - edg_img;
h_edges = figure('Color', 'w');
imshow(edg_img, 'InitialMagnification', 'fit'); hold on;

%% use hough transform to find lines in image
lines = find_lines(edg_img, img_size, params, h_edges);

% separate lines found into two sets based on orientation and identify a
% single square to get things started
[squares, lines, set_inds] = find_first_square(lines);

% find as many squares to the right as possible
delta = 1;
this_square = squares; 
while delta < 8 
    [found, next, set_inds] = find_neighbor(this_square, 'r', lines, set_inds);
    if ~found
        new_lines = no_match_processing(next, img_size, edg_img, params);
        if ~isempty(new_lines)
            new_inds = numel(lines)+1:numel(lines)+numel(new_lines);
            lines(new_inds) = new_lines;
            set_inds{2} = [set_inds{2}, new_inds];
            [found, next] = find_neighbor(this_square, 'r', lines, set_inds);
        end
        if ~found, break; end
    end
    squares(end+1) = next;
    this_square = next;
    delta = delta + 1;
end
% find as many squares to the left as possible
this_square = squares(1);
while delta < 8 
    [found, next, set_inds] = find_neighbor(this_square, 'l', lines, set_inds);
    if ~found
        new_lines = no_match_processing(next, img_size, edg_img, params);
        if ~isempty(new_lines)
            new_inds = numel(lines)+1:numel(lines)+numel(new_lines);
            lines(new_inds) = new_lines;
            set_inds{2} = [set_inds{2}, new_inds];
            [found, next] = find_neighbor(this_square, 'l', lines, set_inds);
        end
        if ~found, break; end
    end
    squares(end+1) = next;
    this_square = next;
    delta = delta + 1;
end
% find as many squares above as possible
delta = 1;
this_square = squares(1); 
while delta < 8 
    [found, next, set_inds] = find_neighbor(this_square, 'u', lines, set_inds);
    if ~found
        new_lines = no_match_processing(next, img_size, edg_img, params);
        if ~isempty(new_lines)
            new_inds = numel(lines)+1:numel(lines)+numel(new_lines);
            lines(new_inds) = new_lines;
            set_inds{1} = [set_inds{1}, new_inds];
            [found, next] = find_neighbor(this_square, 'u', lines, set_inds);
        end
        if ~found, break; end
    end
    squares(end+1) = next;
    this_square = next;
    delta = delta + 1;
end
% find as many squares below as possible
this_square = squares(1);
while delta < 8 
    [found, next, set_inds] = find_neighbor(this_square, 'd', lines, set_inds);
    if ~found
        new_lines = no_match_processing(next, img_size, edg_img, params);
        if ~isempty(new_lines)
            new_inds = numel(lines)+1:numel(lines)+numel(new_lines);
            lines(new_inds) = new_lines;
            set_inds{1} = [set_inds{1}, new_inds];
            [found, next] = find_neighbor(this_square, 'd', lines, set_inds);
        end
        if ~found, break; end
    end
    squares(end+1) = next;
    this_square = next;
    delta = delta + 1;
end
clear next;

% fill in the rest of the squares in the grid
orig_squares = squares;
squares = struct('corners', [], 'inds', [], 'irow', [], 'icol', [], 'P', []); 
all_rel_x = [orig_squares.rel_x];
[~, order_inds_x] = ismember(min(all_rel_x):max(all_rel_x), all_rel_x);
n_cols = numel(order_inds_x);
if n_cols < 8
    warning([mfilename,':number_cols'], 'Could not find all columns of board');
end
all_rel_y = [orig_squares.rel_y];
[~, order_inds_y] = ismember(min(all_rel_y):max(all_rel_y), all_rel_y);
n_rows = numel(order_inds_y);
if n_rows < 8
    warning([mfilename,':number_cols'], 'Could not find all rows of board');
end
% set up array for holding corner location and line index information
next.corners = zeros(4, 2); 
next.inds = zeros(2, 2);
%next.irow = 0; next.icol = 0; next.P = [];
% normalize indices to be 1 through 8 instead of whatever they were
norm_pos = [all_rel_x' - min(all_rel_x) + 1, all_rel_y' - min(all_rel_y) + 1];
for irow = 1:n_rows
    next.irow = irow;
    next.inds(1, :) = orig_squares(order_inds_y(irow)).inds(1, :);
    for icol = 1:n_cols
        next.icol = icol;
        next.inds(2, :) = orig_squares(order_inds_x(icol)).inds(2, :);
        % if we already have the square, add it to the array
        [found, sq_ind] = ismember([icol, irow], norm_pos, 'rows');
        if found
            next.corners = orig_squares(sq_ind).corners;
            next.P = orig_squares(sq_ind).P;
            squares(irow, icol) = next;
            continue;
        end
        % otherwise, need to find corner intersections
        for line1 = 1:2
            theta1 = lines(next.inds(1, line1)).theta * pi/180;
            rho1 = lines(next.inds(1, line1)).rho;
            for line2 = 1:2
                theta2 = lines(next.inds(2, line2)).theta * pi/180;
                rho2 = lines(next.inds(2, line2)).rho;
                A = [cos(theta1), sin(theta1);
                    cos(theta2), sin(theta2)];
                b = [rho1; rho2];
                x = (A \ b)';
                next.corners((line1-1)*2+line2, :) = x;
                plot(x(1), x(2), 'yo');
                next.P = [];
            end
        end
        squares(irow, icol) = next;
    end
end

if ~isempty(fname)
    f = getframe(h_edges);
    imwrite(f.cdata, strrep(fname, '.jpg', '_corners.jpg'));
end







%Functions used in the Program 
% --- 
function lines = find_lines(bw, img_size, p, h_edges)
% apply hough transform
rho_res = ceil(img_size(1)/500);
[H, theta, rho] = hough(bw, 'RhoResolution', rho_res);
p = houghpeaks(H, p.n_peaks, 'Threshold', p.hpeak_pct*max(H(:)));

lines = houghlines(bw, theta, rho, p, 'FillGap', 0.25*img_size(1), ...
    'MinLength', 0.1*img_size(1));
if nargin > 3
    figure(h_edges);
    plot_lines(lines, 'g');
    title(sprintf('%d lines', numel(lines)));
end

% --- separate lines found into two sets based on orientation and identify 
% a single square to get things started
function [square, lines, set_inds] = find_first_square(lines) 
% sort output from houghlines by increasing rho
[rho, sort_inds] = sort([lines(:).rho]);
lines = lines(sort_inds);
theta = [lines(:).theta]; 

% separate lines into two groups based on angle
[counts, locs] = hist(theta, -90:20:90); 
[~, sort_inds] = sort(counts, 'descend');
peak_thetas = locs(sort_inds(1:2)); % find two dominant angle ranges
set_inds = cell(2, 0);
for iset = 1:2
    set_inds{iset} = find(abs(theta - peak_thetas(iset)) < abs(diff(peak_thetas))/3);
    % get distances between lines in each group
    delta_r = diff(rho(set_inds{iset}));
    med_delta = median(delta_r);
    % keep lines that have median separation as good candidates
    keep_inds = abs(delta_r - med_delta) <= sqrt(var(delta_r));
    last_false = find(~keep_inds, 1, 'last');
    first_true = find(keep_inds, 1, 'first');
    last_true = find(keep_inds, 1, 'last');
    % only keep a consecutive set of lines
    if last_false > first_true && last_true > last_false
        if last_true - last_false > last_false - first_true
            keep_inds(1:last_false) = false;
        else
            keep_inds(last_false:end) = false;
        end        
    end
    last_true = find(keep_inds, 1, 'last');
    % add one more index back in because diff operation reduced size by 1
    keep_inds = [keep_inds(1:last_true), true, keep_inds(last_true+1:end)];
    % discard lines with very large or small separation
    best_inds = set_inds{iset}(keep_inds);
%     h_lines = plot_lines(lines(best_inds), 'm');
    n_best = numel(best_inds);
    if mod(n_best, 2) == 0
        square.inds(iset, 1:2) = best_inds(n_best/2:n_best/2+1);
    else
        square.inds(iset, 1:2) = best_inds((n_best-1)/2:(n_best+1)/2);
    end
    % remove indices of lines that have been assigned to a squares from set
    set_inds{iset} = setdiff(set_inds{iset}, square.inds(iset, 1:2));
    h_best = plot_lines(lines(square.inds(iset, 1:2)), 'c');        
end
% this is the first squares found, so it has relative position [0, 0]
square.rel_x = 0;
square.rel_y = 0;
square.P = []; % placeholder for transformation matrix
% find intersection of best two pairs of lines in each direction
square.corners = zeros(4, 2); 
for ind1 = 1:2
    theta1 = lines(square.inds(1, ind1)).theta * pi/180;
    rho1 = lines(square.inds(1, ind1)).rho;
    for ind2 = 1:2
        theta2 = lines(square.inds(2, ind2)).theta * pi/180;
        rho2 = lines(square.inds(2, ind2)).rho;
        A = [cos(theta1), sin(theta1);
            cos(theta2), sin(theta2)];
        b = [rho1; rho2];
        x = A \ b;
        square.corners((ind1-1)*2+ind2, :) = x;
        h_corners = plot(x(1), x(2), 'yo');
    end
end
% also get distances between pairs of lines
square.rhos = [lines(square.inds(1, :)).rho;
    lines(square.inds(2, :)).rho];
square.thetas = [lines(square.inds(1, :)).theta;
    lines(square.inds(2, :)).theta];



% --- search for neighboring squares
function [found, next, set_inds] = find_neighbor(square, direction, ...
    lines, set_inds)

switch direction 
    case 'r'
        s = 2; % candidate lines come from this set
        t = 1; % look for intersections with lines from this set
        old_c = [2, 4]; % these corners are shared with neighbor
        new_c = [1, 3]; % these will be the newly found corners
        % 3 of 4 lines for neighboring square are already known
        next.inds = zeros(2, 2);
        next.inds(1, 1:2) = square.inds(1, 1:2);
        next.inds(2, 1) = square.inds(2, 2);
        nl = 4; % new line goes here
        next.rel_x = square.rel_x + 1; % +1 to x position
        next.rel_y = square.rel_y;
        new_xy = [2, 0, 1; 2, 1, 1]; % new corners before transform
    case 'l'
        s = 2; % candidate lines come from this set
        t = 1; % look for intersections with lines from this set
        old_c = [1, 3]; % these corners are shared with neighbor
        new_c = [2, 4]; % these will be the newly found corners
        % 3 of 4 lines for neighboring square are already known
        next.inds = zeros(2, 2);
        next.inds(1, 1:2) = square.inds(1, 1:2);
        next.inds(2, 2) = square.inds(2, 1);
        nl = 2; % new line goes here
        next.rel_x = square.rel_x - 1; % -1 to x position
        next.rel_y = square.rel_y;
        new_xy = [-1, 0, 1; -1, 1, 1]; % new corners before transform
    case 'u'
        s = 1; % candidate lines come from this set
        t = 2; % look for intersections with lines from this set
        old_c = [3, 4]; % these corners are shared with neighbor
        new_c = [1, 2]; % these will be the newly found corners
        % 3 of 4 lines for neighboring square are already known
        next.inds = zeros(2, 2);
        next.inds(1, 1) = square.inds(1, 2);
        next.inds(2, 1:2) = square.inds(2, 1:2);
        nl = 3; % new line goes here
        next.rel_x = square.rel_x; % -1 to y position
        next.rel_y = square.rel_y - 1;
        new_xy = [0, 2, 1; 1, 2, 1]; % new corners before transform
    case 'd'
        s = 1; % candidate lines come from this set
        t = 2; % look for intersections with lines from this set
        old_c = [1, 2]; % these corners are shared with neighbor
        new_c = [3, 4]; % these will be the newly found corners
        % 3 of 4 lines for neighboring square are already known
        next.inds = zeros(2, 2);
        next.inds(1, 2) = square.inds(1, 1);
        next.inds(2, 1:2) = square.inds(2, 1:2);
        nl = 1; % new line goes here
        next.rel_x = square.rel_x; % +1 to y position
        next.rel_y = square.rel_y + 1;
        new_xy = [0, -1, 1; 1, -1, 1]; % new corners before transform
    otherwise
        error('Invalid direction')
end

% predict location of next point and angle of line going through it
P = get_transform(square.corners(:, 1)', square.corners(:, 2)');
next.P = P;
new_xy(:, 1:2) = new_xy(:, 1:2)*100;
tmp = new_xy * P'; 
pred_xy = tmp(:, 1:2)./repmat(tmp(:, 3), 1, 2);
dx_dy = diff(pred_xy);
next_theta = atan(dx_dy(2)/dx_dy(1)) * 180/pi; 
if next_theta > 0
    next_theta = next_theta - 90;
else
    next_theta = next_theta + 90;
end
% next_theta = mean(square.thetas(s, :))

% check candidate lines against expected angle
cand_thetas = [lines(set_inds{s}).theta];
cand_inds = abs(cand_thetas - next_theta) < 5 | ...
    abs(cand_thetas - next_theta) > 175;
test_angles = [];
test_inds = set_inds{s}(cand_inds);
for icand = test_inds
    pt1 = lines(icand).point1;
    dx_dy = pred_xy(1, :) - pt1;
    test_angles(end+1) = atan(dx_dy(2)/dx_dy(1)) * 180/pi;
end
match_inds = abs(test_angles - (next_theta+90)) < 10 | ...
    abs(test_angles - (next_theta-90)) < 10;
% if angle test did not find any matchees, quit here
found = any(match_inds);
if ~found
    warning([mfilename,':find_neighbor'], 'No matching angle found.');
    % if we didn't find a match, instead of returning a square, return data
    % about where we expected to find the next line
    next = [];
    next.slope = -tan((90 - next_theta)*pi/180);
    next.pred_xy = pred_xy;
    return;
end

% set up array for holding corner location information
next.corners = zeros(4, 2); 
next.corners(new_c, :) = square.corners(old_c, :);

% otherwise, continue testing candidate line
deltas = diff(square.corners([new_c(2), old_c(2)], :));
dist_thresh = 0.2*sqrt(sum(deltas.^2));
cand_lines = test_inds(match_inds);
pt_dist = nan(numel(cand_lines), 2);
all_corners = nan(numel(cand_lines), 2, 2);
for iline = 1:numel(cand_lines)
    ind = cand_lines(iline);
    theta1 = lines(ind).theta * pi/180;
    rho1 = lines(ind).rho;
    h_corners = nan(2, 1);
    for ind2 = 1:2
        theta2 = lines(square.inds(t, ind2)).theta * pi/180;
        rho2 = lines(square.inds(t, ind2)).rho;
        A = [cos(theta1), sin(theta1);
            cos(theta2), sin(theta2)];
        b = [rho1; rho2];
        x = (A \ b)';
        % use point predicted using transform matrix
        pt_delta = pred_xy(ind2, :) - x;
        pt_dist(iline, ind2) = sqrt(sum(pt_delta.^2));
        all_corners(iline, :, ind2) = x;
    end
end
dist_ok = all(pt_dist < dist_thresh, 2);
[~, best_ind] = min(sum(pt_dist, 2));
found = dist_ok(best_ind);
if ~found
    warning([mfilename,':find_neighbor'], 'Intersection too far.');
    % if we didn't find a match, instead of returning a square, return data
    % about where we expected to find the next line
    next = [];
    next.slope = -tan((90 - next_theta)*pi/180);
    next.pred_xy = pred_xy;
    return;
end

% plot corners and add data to the next square
next.inds(nl) = cand_lines(best_ind);
new_corners = squeeze(all_corners(best_ind, :, :))';
next.corners(old_c, :) = new_corners;
h_corners = plot(new_corners(:, 1),new_corners(:, 2), 'yo');
set_inds{s} = setdiff(set_inds{s}, ind);

% get rho and theta values for the lines in this square
next.rhos = [lines(next.inds(1, :)).rho;
    lines(next.inds(2, :)).rho];
next.thetas = [lines(next.inds(1, :)).theta;
    lines(next.inds(2, :)).theta];
% figure; plot([lines(:).theta], [lines(:).rho], '*'); grid on;

function new_lines = no_match_processing(data, img_size, e_img, params)
slope = data.slope;
xy = data.pred_xy;

if slope == 0
    start_pt = [1, xy(1, 2)];
elseif xy(1, 1) - xy(1, 1)/slope < 0
    start_pt = [ceil(xy(1, 1) - xy(1, 2)/slope), 1];
elseif xy(1, 1) - xy(1, 1)/slope > img_size(1)
    start_pt = [ceil(xy(1, 1) + (img_size(1) - xy(1, 2))/slope), img_size(1)];
else
    start_pt = [1, ceil(xy(1, 2) - xy(1,1)*slope)];
end
y = 1:img_size(1);
for icol = 1:img_size(2)
    col_inds = false(1, img_size(2)); col_inds(icol) = true;
    row_inds = slope*(icol - start_pt(1)) + start_pt(2) < y - 50 | ...
        slope*(icol - start_pt(1)) + start_pt(2) > y + 50;
    e_img(row_inds, col_inds) = false;
end
new_lines = find_lines(e_img, img_size, params);


%--- plot lines using format of output from houghlines
function h_lines = plot_lines(lines, color)
h_lines = zeros(numel(lines), 1);
for iline = 1:numel(lines)
    xy = [lines(iline).point1; lines(iline).point2];
    h_lines(iline) = plot(xy(:, 1), xy(:, 2), color);
end

% --- 
function P = get_transform(X, Y)
x = 100*[0, 1, 0, 1];  %  corner format 3  4
y = 100*[0, 0, 1, 1];  %                1  2
n = numel(x);
w = ones(1, n);
xyw = [x; y; w]';
XY = [X; Y];
A = zeros(2*n, 8);
A(2*(1:n)-1, 1:3) = xyw;
A(2*(1:n), 4:6) = xyw;
A(:, 7) = -reshape(repmat(x, 2, 1) .* XY, 2*n, 1);
A(:, 8) = -reshape(repmat(y, 2, 1) .* XY, 2*n, 1);
P = A \ XY(:);
P = reshape([P; 1], 3, 3)';


