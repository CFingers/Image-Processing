%% Project 3 Image Restoration
% By Christopher W. Fingers
% EE 483

%% Part 1 Restoring Images by Salt and Pepper
clc,clear;
% Load in our image, a gray gull.
graygull = imread('gull.jpg');

% Apply the 5% salt and pepper noise for the gull.
somesalt = imnoise(graygull,'salt & pepper',0.05);
% Chapter 8 problem 6
% Part a

% Setting up our 2 filters, a 3x3 and 5x5.
avgfilt3 = fspecial('average');
avgfilt5 = fspecial('average',5);

% Applying the filters to the Salt and Pepper Gull
avggull3 = imfilter(somesalt, avgfilt3);
avggull5 = imfilter(somesalt, avgfilt5);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(avggull3);
title('3ex3 Average Filter');
subplot(2,2,4);
imshow(avggull5);
title('5x5 Average Filter');

% Part b.  This part uses the same gray gull image and salt and pepper
% noise, however a 3x3 and 5x5 median filter will be used instead of the
% average filter.

% 3x3 and 5x5 full mask median filters
median3gull = medfilt2(somesalt);
median5gull = medfilt2(somesalt,[5,5]);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(median3gull);
title('3x3 Median Filter');
subplot(2,2,4);
imshow(median5gull);
title('5x5 Median Filter');

% 3x3 and 5x5 cross mask filters
cross3 = [0 1 0, 1 1 1, 0 1 0];
cross5 = [0 0 1 0 0, 0 0 1 0 0, 1 1 1 1 1, 0 0 1 0 0, 0 0 1 0 0];

crossed3gull = ordfilt2(somesalt,3,cross3);
crossed5gull = ordfilt2(somesalt,5,cross5);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(crossed3gull);
title('3x3 Cross Median Filter');
subplot(2,2,4);
imshow(crossed5gull);
title('5x5 Cross Median Filter');

% 3x3 and 5x5 X mask filters
X3 = [1 0 1, 0 1 0, 1 0 1];
X5 = [1 0 0 0 1, 0 1 0 1 0, 0 0 1 0 0, 0 1 0 1 0, 1 0 0 0 1];

X3gull = ordfilt2(somesalt,3,X3);
X5gull = ordfilt2(somesalt,5,X5);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(X3gull);
title('3x3 X Median Filter');
subplot(2,2,4);
imshow(X5gull);
title('5x5 X Median Filter');

% WIth only 5% salt a standard median filter seems to produce a clearer
% image.  The white specks are gone and the image is relatively intact
% still.  The Average filter has a slight blurr around it and both the
% oross/X median filters effect the image.
%% Problem 7, 10% Salt
clc,clear;
% Apply the 10% salt and pepper noise for the gull.
graygull = imread('gull.jpg');
somesalt = imnoise(graygull,'salt & pepper',0.1);
% Chapter 8 problem 6
% Part a

% Setting up our 2 filters, a 3x3 and 5x5.
avgfilt3 = fspecial('average');
avgfilt5 = fspecial('average',5);

% Applying the filters to the Salt and Pepper Gull
avggull3 = imfilter(somesalt, avgfilt3);
avggull5 = imfilter(somesalt, avgfilt5);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(avggull3);
title('3ex3 Average Filter');
subplot(2,2,4);
imshow(avggull5);
title('5x5 Average Filter');

% Part b.  This part uses the same gray gull image and salt and pepper
% noise, however a 3x3 and 5x5 median filter will be used instead of the
% average filter.

% 3x3 and 5x5 full mask median filters
median3gull = medfilt2(somesalt);
median5gull = medfilt2(somesalt,[5,5]);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(median3gull);
title('3x3 Median Filter');
subplot(2,2,4);
imshow(median5gull);
title('5x5 Median Filter');

% 3x3 and 5x5 cross mask filters
cross3 = [0 1 0, 1 1 1, 0 1 0];
cross5 = [0 0 1 0 0, 0 0 1 0 0, 1 1 1 1 1, 0 0 1 0 0, 0 0 1 0 0];

crossed3gull = ordfilt2(somesalt,3,cross3);
crossed5gull = ordfilt2(somesalt,5,cross5);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(crossed3gull);
title('3x3 Cross Median Filter');
subplot(2,2,4);
imshow(crossed5gull);
title('5x5 Cross Median Filter');

% 3x3 and 5x5 X mask filters
X3 = [1 0 1, 0 1 0, 1 0 1];
X5 = [1 0 0 0 1, 0 1 0 1 0, 0 0 1 0 0, 0 1 0 1 0, 1 0 0 0 1];

X3gull = ordfilt2(somesalt,3,X3);
X5gull = ordfilt2(somesalt,5,X5);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(X3gull);
title('3x3 X Median Filter');
subplot(2,2,4);
imshow(X5gull);
title('5x5 X Median Filter');

% Similar to the 5%, the median filter still reproduces a clearer image
% than the other 3 different filters.  With the increase in salt, the image
% distortion effect caused by the other filters is enhanced.
%% Problem 7 20% Salt
clc,clear;
% Apply the 20% salt and pepper noise for the gull.
graygull = imread('gull.jpg');
somesalt = imnoise(graygull,'salt & pepper',0.2);
% Chapter 8 problem 6
% Part a

% Setting up our 2 filters, a 3x3 and 5x5.
avgfilt3 = fspecial('average');
avgfilt5 = fspecial('average',5);

% Applying the filters to the Salt and Pepper Gull
avggull3 = imfilter(somesalt, avgfilt3);
avggull5 = imfilter(somesalt, avgfilt5);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(avggull3);
title('3ex3 Average Filter');
subplot(2,2,4);
imshow(avggull5);
title('5x5 Average Filter');

% Part b.  This part uses the same gray gull image and salt and pepper
% noise, however a 3x3 and 5x5 median filter will be used instead of the
% average filter.

% 3x3 and 5x5 full mask median filters
median3gull = medfilt2(somesalt);
median5gull = medfilt2(somesalt,[5,5]);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(median3gull);
title('3x3 Median Filter');
subplot(2,2,4);
imshow(median5gull);
title('5x5 Median Filter');

% 3x3 and 5x5 cross mask filters
cross3 = [0 1 0, 1 1 1, 0 1 0];
cross5 = [0 0 1 0 0, 0 0 1 0 0, 1 1 1 1 1, 0 0 1 0 0, 0 0 1 0 0];

crossed3gull = ordfilt2(somesalt,3,cross3);
crossed5gull = ordfilt2(somesalt,5,cross5);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(crossed3gull);
title('3x3 Cross Median Filter');
subplot(2,2,4);
imshow(crossed5gull);
title('5x5 Cross Median Filter');

% 3x3 and 5x5 X mask filters
X3 = [1 0 1, 0 1 0, 1 0 1];
X5 = [1 0 0 0 1, 0 1 0 1 0, 0 0 1 0 0, 0 1 0 1 0, 1 0 0 0 1];

X3gull = ordfilt2(somesalt,3,X3);
X5gull = ordfilt2(somesalt,5,X5);

figure
subplot(2,2,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,2,2);
imshow(somesalt);
title('Salt and Pepper Gull');
subplot(2,2,3);
imshow(X3gull);
title('3x3 X Median Filter');
subplot(2,2,4);
imshow(X5gull);
title('5x5 X Median Filter');

% Finally at 20% the standard median filter still has a clearer image.  The
% Cross filter looks clearer than the average and X filter with the average
% filter coming out before the X filter.  The distortion effect is intense
% for the X filter at this point and the average filter is causing more
% and more blur.

%% Part 2 Problem 9
% This section focuses on using average, Wiener and midpoint filters to
% clean up an image with an increasing amount of Gaussian noise.
%% Average
clc,clear;
graygull = imread('gull.jpg');

gausgull = imnoise(graygull, 'gaussian');
gausgullb = imnoise(graygull, 'gaussian', .02);
gausgullc = imnoise(graygull, 'gaussian', .05);
gausgulld = imnoise(graygull, 'gaussian', .1);

avggull = imfilter(gausgull, fspecial('average'));
avggullb = imfilter(gausgullb, fspecial('average'));
avggullc = imfilter(gausgullc, fspecial('average'));
avggulld = imfilter(gausgulld, fspecial('average'));

figure
subplot(2,1,1);
imshow(graygull);
title('A Gray Gull');
subplot(2,1,2);
imshow(gausgull);
title('Gaussian Gull');

figure
subplot(2,2,1);
imshow(avggull);
title('Gaussian .01 Average Filter Gull');
subplot(2,2,2);
imshow(avggullb);
title('Gaussian .02 Average Filter Gull');
subplot(2,2,3);
imshow(avggullc);
title('Gaussian .05 Average Filter Gull');
subplot(2,2,4);
imshow(avggulld);
title('Gaussian .1 Average Filter Gull');

%% Wiener
clc,clear;
graygull = imread('gull.jpg');

gausgull = imnoise(graygull, 'gaussian');
gausgullb = imnoise(graygull, 'gaussian', .02);
gausgullc = imnoise(graygull, 'gaussian', .05);
gausgulld = imnoise(graygull, 'gaussian', .1);

wienergull = wiener2(gausgull);
wienergullb = wiener2(gausgullb);
wienergullc = wiener2(gausgullc);
wienergulld = wiener2(gausgulld);

figure
subplot(2,2,1);
imshow(wienergull);
title('Gaussian .01 Wiener Filter Gull');
subplot(2,2,2);
imshow(wienergullb);
title('Gaussian .02 Wiener Filter Gull');
subplot(2,2,3);
imshow(wienergullc);
title('Gaussian .05 Wiener Filter Gull');
subplot(2,2,4);
imshow(wienergulld);
title('Gaussian .1 Wiener Filter Gull');

%% Midpoint
clc,clear;
graygull = imread('gull.jpg');

gausgull = imnoise(graygull, 'gaussian');
gausgullb = imnoise(graygull, 'gaussian', .02);
gausgullc = imnoise(graygull, 'gaussian', .05);
gausgulld = imnoise(graygull, 'gaussian', .1);

MaxGull = ordfilt2(gausgull,9,ones(3,3));
MinGull = ordfilt2(gausgull,1,ones(3,3));
midpoint = (MaxGull + MinGull)./2;

MaxGullb = ordfilt2(gausgullb,9,ones(3,3));
MinGullb = ordfilt2(gausgullb,1,ones(3,3));
midpointb = (MaxGullb + MinGullb)./2;

MaxGullc = ordfilt2(gausgullc,9,ones(3,3));
MinGullc = ordfilt2(gausgullc,1,ones(3,3));
midpointc = (MaxGullc + MinGullc)./2;

MaxGulld = ordfilt2(gausgulld,9,ones(3,3));
MinGulld = ordfilt2(gausgulld,1,ones(3,3));
midpointd = (MaxGulld + MinGulld)./2;

figure
subplot(2,2,1);
imshow(midpoint);
title('Gaussian .01 Midpoint Filter');
subplot(2,2,2);
imshow(midpointb);
title('Gaussian .02 Midpoint Filter');
subplot(2,2,3);
imshow(midpointc);
title('Gaussian .05 Midpoint Filter');
subplot(2,2,4);
imshow(midpointd);
title('Gaussian .1 Midpoint Filter');

%% Part 3 Chapter 8 Problem 12
% This part involves Restoring an image that has periodic noise.  
clc,clear;
graygull = imread('gull.jpg');
% Apply the periodic noise to the gray gull image.
[ys, xs] = size(graygull);
[y,x] = meshgrid(1:xs,1:ys);
s=1+sin(x+y/1.5);
periodicgull = (2*im2double(graygull) + s/2)/3;
DFTgull = fftshift(fft2(periodicgull));

%  Taking the DFT shiffted gull image, we area ble to find the maximum
%  value for the gull, which is at position 129,129.  Nulling this point we
%  are able to find the other max areas and the combined distance between
%  those areas from the center.  After which we apply the bandring function
%  to remove the noise of the goa.
Gullmax = im2uint8(mat2gray(abs(DFTgull)));
Gullmax(129,129) = 0;
[i,j] = find(Gullmax==max(Gullmax(:)))
(i-129).^2 + (j-129).^2

% This section defines the bandringed area for the image
z=sqrt((x-129).^2 + (y-129).^2);
d = sqrt(2410);
k = 1;
bandring = (z<floor(d-k) | z>ceil(d+k));

% Applying the Bandring Function to the Gull image
bandringgull = DFTgull .*bandring;
gulled= abs(ifft2(bandringgull));

% Applying and setting up the criss cross function for the gull.
CrossedGull = DFTgull;
CrossedGull(i,:) = 0;
CrossedGull(:,j) = 0;

xgull = abs(ifft2(CrossedGull));

figure
subplot(2,2,1)
imshow(periodicgull)
title('Periodic Gull')
subplot(2,2,2)
imshow(gulled)
title('Bandring Gull k = 1')
subplot(2,2,3)
imshow(xgull)
title('Criss Crossed Gull')

% With a k value of one for the bandring function, the criss cross produces
% an image with less noise.  The higher the K value is increased the gull
% itself is cleared out, but there is noise that starts to appear around
% the gull.  The criss cross function appears to clear up the image better.

%% Part 4 Problems 14,15,16
% This section focuses heavily on using the inverse function to clear up
% blurred images and compare different size filters with increasing values.

%% Problem 14 5x5 filter with constrained division.

clc,clear;
graygull = imread('gull.jpg');

% Set up our low pass butterworth filter values to blur the image.
DFTgull = fftshift(fft2(graygull));
[ys,xs] = size(graygull);
[x,y] = meshgrid(-xs/2:xs/2-1, -ys/2:ys/2-1);

% Define our multiple Divider values for the normal butterworth and the
% different ones for the inverse contrained division.
D=5; d=0.01; d1=0.005; d2=0.002; d3= 0.001;
n=1;
butter = 1./(1+((x.^2+y.^2)./D^2).^n);
buttergull = DFTgull.*butter;
bg = abs(ifft2(buttergull));
blurredgull = im2uint8(mat2gray(bg));
gulldf=fftshift(fft2(blurredgull));

butter1 =butter; butter1(find(butter1<d))=1;
butter2 =butter; butter2(find(butter2<d1))=1;
butter3 =butter; butter3(find(butter3<d2))=1;
butter4 =butter; butter4(find(butter4<d3))=1;

butteredgull1 = fftshift(fft2(blurredgull))./butter1;
butteredgull2 = fftshift(fft2(blurredgull))./butter2;
butteredgull3 = fftshift(fft2(blurredgull))./butter3;
butteredgull4 = fftshift(fft2(blurredgull))./butter4;

bg1 = abs(ifft2(butteredgull1));
bg2 = abs(ifft2(butteredgull2));
bg3 = abs(ifft2(butteredgull3));
bg4 = abs(ifft2(butteredgull4));


% Normal blurred gull with the buttersworth filter.
figure
imshow(blurredgull)
title('Blurred gull')

% Inverse butterworth values.
figure
subplot(2,2,1)
imshow(mat2gray(bg1))
title('Inverse divisor at .01');
subplot(2,2,2)
imshow(mat2gray(bg2))
title('Inverse divisor at .005');
subplot(2,2,3)
imshow(mat2gray(bg3))
title('Inverse divisor at .002');
subplot(2,2,4)
imshow(mat2gray(bg4))
title('Inverse divisor at .001');

% With a 5x5 buttersworth filter, the clearest image comes from a
% constrained divisor of between .01 and .005.  Any lower and the image
% becomes horrible disfigured.

%% Question 15 and 16 7x7 filter with constrained division.

clc,clear;
graygull = imread('gull.jpg');

% Set up our low pass butterworth filter values to blur the image.
DFTgull = fftshift(fft2(graygull));
[ys,xs] = size(graygull);
[x,y] = meshgrid(-xs/2:xs/2-1, -ys/2:ys/2-1);

% Define our multiple Divider values for the normal butterworth and the
% different ones for the inverse contrained division.
D=7; d=0.01; d1=0.005; d2=0.002; d3= 0.001;
n=1;
butter = 1./(1+((x.^2+y.^2)./D^2).^n);
buttergull = DFTgull.*butter;
bg = abs(ifft2(buttergull));
blurredgull = im2uint8(mat2gray(bg));
gulldf=fftshift(fft2(blurredgull));

butter1 =butter; butter1(find(butter1<d))=1;
butter2 =butter; butter2(find(butter2<d1))=1;
butter3 =butter; butter3(find(butter3<d2))=1;
butter4 =butter; butter4(find(butter4<d3))=1;

butteredgull1 = fftshift(fft2(blurredgull))./butter1;
butteredgull2 = fftshift(fft2(blurredgull))./butter2;
butteredgull3 = fftshift(fft2(blurredgull))./butter3;
butteredgull4 = fftshift(fft2(blurredgull))./butter4;

bg1 = abs(ifft2(butteredgull1));
bg2 = abs(ifft2(butteredgull2));
bg3 = abs(ifft2(butteredgull3));
bg4 = abs(ifft2(butteredgull4));


% Normal blurred gull with the buttersworth filter.
figure
imshow(blurredgull)
title('Blurred gull')

% Inverse butterworth values.
figure
subplot(2,2,1)
imshow(mat2gray(bg1))
title('Inverse divisor at .01');
subplot(2,2,2)
imshow(mat2gray(bg2))
title('Inverse divisor at .005');
subplot(2,2,3)
imshow(mat2gray(bg3))
title('Inverse divisor at .002');
subplot(2,2,4)
imshow(mat2gray(bg4))
title('Inverse divisor at .001');

% With a 7x7 filter, the same range from before shows the best threshold
% results.  With .005 starting to show some signs of distortion.  Any lower
% and it becomes very hard to see.

%% Part 5 Question 10/11
clc,clear
%Question 10

astronaut = imread('justchilling.jpg');
noiseyastro = imnoise(astronaut,'gaussian');

% Apply the average filter accross the gaussian noised astronaut
average = fspecial('average',5);
astroback = imfilter(noiseyastro, average);

% Apply the wiener filter to the gaussian astronaut
k2 = zeros(size(noiseyastro));
for i = 1:3, 
    k2(:,:,i) = wiener2(noiseyastro(:,:,i),[5,5]); 
end

figure
subplot(2,2,1)
imshow(astronaut)
title('Normal Astronaut')
subplot(2,2,2)
imshow(noiseyastro)
title('Gaussian Astronaut')
subplot(2,2,3)
imshow(astroback)
title('Average Astronaut')
subplot(2,2,4)
imshow(k2/255)
title('Wiener Astronaut')

% After applying the 2 fiters to the Gaussian astronaut image, they both
% work very well.  There are some artifacts in the Gaussian astronaut image
% on the top left and around the planet, but the image is finer in detail
% then the average filter.  The average filter lost nearly all the stars
% and darkened the relfeciton off the moon, where as the wiener filter has
% some blurred parts.

%% Question 11
% Part a

astronaut = imread('justchilling.jpg');
ay = rgb2ntsc(astronaut);

an = imnoise(ay(:,:,1),'salt & pepper');
ay(:,:,1)=an;

for i = 1:3,
astronautnoise = imnoise(astronaut(:,:,i),'salt & pepper');
astron(:,:,i) = astronautnoise;

end


figure
subplot(1,2,1)
imshow(ntsc2rgb(ay))
title('Intensity')
subplot(1,2,2)
imshow(astron)
title('Normal')

% When apply the salt and pepper noise to the intensisty of the image, the
% noise in the background are represented by varying white specs instead of
% the varying red, green, and blue specs that appear in the image.

% Part B

ab = medfilt2(ay(:,:,1));
ay(:,:,1) = ab;

% Part C
for i = 1:3,
astronautnoise = medfilt2(astronaut(:,:,i));
astron(:,:,i) = astronautnoise;

end

figure
subplot(1,2,1)
imshow(ntsc2rgb(ay))
title('Intensity')
subplot(1,2,2)
imshow(astron)
title('Normal')

% Part D: In terms of effect both worked very well in smoothing ou tthe
% image and getting rid of the salt and pepper noise, however in terms of
% better detail, applying the median filter to intensity work much better.
% There are smaller, individual details like that stars in the sky and
% reflected parts on the moon that shine through.

% Part E

an = imnoise(ay(:,:,1),'salt & pepper',0.05);
ay(:,:,1)=an;

for i = 1:3,
astronautnoise = imnoise(astronaut(:,:,i),'salt & pepper',0.05);
astron(:,:,i) = astronautnoise;

end

ab = medfilt2(ay(:,:,1));
ay(:,:,1) = ab;

for i = 1:3,
astronautnoise = medfilt2(astronaut(:,:,i));
astron(:,:,i) = astronautnoise;

end

figure
subplot(1,2,1)
imshow(ntsc2rgb(ay))
title('Intensity')
subplot(1,2,2)
imshow(astron)
title('Normal')

% With higher noise it seems that the normal image is better to apply the
% median filter on, rather than the intensity.  The light on the surface of
% the moon and stars in the background grow faded with intensity.

% Part F

an = imnoise(ay(:,:,1),'gaussian');
ay(:,:,1)=an;

for i = 1:3,
astronautnoise = imnoise(astronaut(:,:,i),'gaussian');
astron(:,:,i) = astronautnoise;

end

figure
subplot(1,2,1)
imshow(ntsc2rgb(ay))
title('Intensity')
subplot(1,2,2)
imshow(astron)
title('Normal')

ab = medfilt2(ay(:,:,1));
ay(:,:,1) = ab;

for i = 1:3,
astronautnoise = medfilt2(astronaut(:,:,i));
astron(:,:,i) = astronautnoise;

end

figure
subplot(1,2,1)
imshow(ntsc2rgb(ay))
title('Intensity')
subplot(1,2,2)
imshow(astron)
title('Normal')

% With a gaussian noise filter, the median filter is much better at
% cleaning up the overall image on the RGB components, rather than the
% intensity.