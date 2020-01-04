%% Project 1 Point and Neighborhood Processing
% By Christopher W. Fingers
% EE 483
% 3/5/2019

%% Part 1 Uniform Quantization vs. uint8 Division
clc,clear;

vector = uint8(0:255);       % Define our uint vector from 0 to 255
matrix = vec2mat(vector,16); % Reshape vector into 16x16 matrix
k1 = 64; k2=32; k3 = 16;
m1 = (matrix.*k1)/k1;
m2 = (matrix.*k2)/k2;
m3 = (matrix.*k3)/k3;

% The reason for the difference is because of how the uint8 function
% works.  Uint8 had a minimum value of 0 and a maximum value of 255.  
% When multiplying and dividing the selected values this way the 
% values that are supposed to be above 255 are set at a value of 255,
% then when divided by the same number it will divide to the maximum
% number that is possibly divisible.  64 gives us 4, 32 gives us 8, 
% and 16 gives us 16.

% The difference between uniformally quantizing by 64, 32, and 16 is 
% that certain values will lead to a higher quantized level.  An 
% example is with matrix m1, a large majority of the matrix is set 
% to a quantized level of 4, where as the first 4 elements have a 
% value of 0, 1, 2, and, 3.

%% Part 2 Image Blurring

b = imread('buffalo.png');
avg1 = fspecial('average',[3,3]);
avg2 = fspecial('average',[5,5]);
avg3 = fspecial('average',[9,9]);

b1 = imfilter(b,avg1);
b2 = imfilter(b,avg2);
b3 = imfilter(b,avg3);

figure
subplot(1,3,1);
imshow(b1);
title('3x3 average');
subplot(1,3,2);
imshow(b2);
title('5x5 average');
subplot(1,3,3);
imshow(b3);
title('9x9 average');

% The finer detail of the image starts to become harder to view as the
% filter passes a 9x9 averaging filter.  The fur on the buffalo, the
% individual grass blades and the leaves in the tree become a blurred 
% image at this point.

% Gaussian Filters

gauss1 = fspecial('gaussian',[3,3],0.5);
gauss2 = fspecial('gaussian',[3,3],1);
gauss3 = fspecial('gaussian',[3,3],2);

gauss4 = fspecial('gaussian',[7,7],1);
gauss5 = fspecial('gaussian',[7,7],3);
gauss6 = fspecial('gaussian',[7,7],6);

gauss7 = fspecial('gaussian',[11,11],1);
gauss8 = fspecial('gaussian',[11,11],4);
gauss9 = fspecial('gaussian',[11,11],8);

gauss10 = fspecial('gaussian',[21,21],1);
gauss11 = fspecial('gaussian',[21,21],5);
gauss12 = fspecial('gaussian',[21,21],10);

bg1=imfilter(b,gauss1);
bg2=imfilter(b,gauss2);
bg3=imfilter(b,gauss3);
bg4=imfilter(b,gauss4);
bg5=imfilter(b,gauss5);
bg6=imfilter(b,gauss6);
bg7=imfilter(b,gauss7);
bg8=imfilter(b,gauss8);
bg9=imfilter(b,gauss9);
bg10=imfilter(b,gauss10);
bg11=imfilter(b,gauss11);
bg12=imfilter(b,gauss12);

figure
subplot(1,3,1); imshow(bg1); title('3x3');
subplot(1,3,2); imshow(bg2); title('3x3');
subplot(1,3,3); imshow(bg3); title('3x3');

figure
subplot(1,3,1); imshow(bg4); title('7x7');
subplot(1,3,2); imshow(bg5); title('7x7');
subplot(1,3,3); imshow(bg6); title('7x7')

figure
subplot(1,3,1); imshow(bg7); title('11x11');
subplot(1,3,2); imshow(bg8); title('11x11');
subplot(1,3,3); imshow(bg9); title('11x11');

figure
subplot(1,3,1); imshow(bg10); title('21x21');
subplot(1,3,2); imshow(bg11); title('21x21');
subplot(1,3,3); imshow(bg12); title('21x21');

% The finer details for the gaussian image start to mainly disappear 
% around an 11x11 filter with a standard deviation of 4.  Similar to 
%the averaging filter, the detail in terms of the buffalo's hair, 
%the grass, and the trees are blurred out to an extreme extent.

% The averaging filter and Gaussian filter apply a blurring effect 
% that makes the more specific details in the image harder to see.  An 
% averaging filter can be made more blurred by expanding the overall 
% size of the filter, where as the Gaussian filter is effected by an 
% expanded filter  size, but the standard deviation for the filter 
% causes a more intense blurring effect.

%% Part 3 Edge Detection High Pass Filterting

clc,clear;
b = imread('buffalo.png');
c = imread('cameraman.png');

f = fspecial('laplacian');
f1 = fspecial('log');

bf=imfilter(b,f,'symmetric');
cf=imfilter(c,f,'symmetric');

bf1=imfilter(b,f1,'symmetric');
cf1=imfilter(c,f1,'symmetric');

subplot(2,2,1);
imshow(bf)
title('Laplacian filter')
subplot(2,2,2);
imshow(bf1)
title('LOG');
subplot(2,2,3);
imshow(cf)
title('Laplacian Filter')
subplot(2,2,4);
imshow(cf1)
title('LOG')

% The Laplacian of Gaussian seems to have better edge detection than 
% the standard Laplacian.  When comparing the two cameramen, the LOG 
% filter has a much more defined edge than the Laplacian.  The buffalo 
% filter has some problems in terms of the edge detection because of 
% the blades of grass and trees, however similarly to the cameraman,
% any edge is much more well defined.

%% Part 4 Unshapr Masking

clc,clear;
b=imread('buffalo.png');

avgfilt=fspecial('average',[3,3]);
bavgfilt=imfilter(b,avgfilt);

unsharp=fspecial('unsharp');
bnotverysharp=imfilter(b,unsharp);

subplot(1,2,1);
imshow(bavgfilt);
title('Average Filter');
subplot(1,2,2);
imshow(bnotverysharp);
title('Unsharp Filter');

% After the averaging filter was applied to the buffalo the image had 
% a slight blurriness to it, however with the application of the 
% unsharp filter the image almost seems as though no averaging filter 
% was applied. The details of the buffalo image are intense and pop 
% out more effectively when applied.  The unsharp image definitely 
% reversed the effects of the averaging filter and made the 
% surrounding area well-defined.
