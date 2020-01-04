%% Project 4 Chapter 9
% By Christopher W. Fingers

% Part 1 Problem 5


nice = im2uint8(imread('nicework.png'));
camera = imread('cameraman.png');
mix = imlincomb(0.5,camera,0.5,nice);

figure
imshow(mix)

% Because the text is white, we can super impose the text of nice work by
% changing the value up to 1 for the nice work portion and 0 for the camera
% portion.

% Problem 7

circle = imread('circles.png');
[x,y] = meshgrid(1:256,1:256);
circle2 = double(circle).*(x+y)/512;
circle3 = im2uint8(circle2);

figure
imshow(circle3)

% By using the function given in the problem statement, the circle value
% has a much darker tone up to the left of the image.  
%% Part 2 Problem 10

camera = imread('cameraman.png');

robera = edge(camera,'roberts');
prebra = edge(camera,'prewitt');
sobera = edge(camera,'sobel');
laps = fspecial('laplacian',0);
lapera = imfilter(camera,laps);
zero = edge(camera,'zerocross',[],laps);
marrlog = fspecial('log',13,2);
Marera = edge(camera,'zerocross',[],marrlog);
cannera = edge(camera,'canny');

figure
subplot(2,2,1)
imshow(robera)
title('Roberts')
subplot(2,2,2)
imshow(prebra)
title('Prewitt')
subplot(2,2,3)
imshow(sobera)
title('Sobel')
subplot(2,2,4)
imshow(lapera)
title('Laplacian')

figure
subplot(2,2,1)
imshow(zero)
title('zero crossing')
subplot(2,2,2)
imshow(Marera)
title('Marrilog')
subplot(2,2,3)
imshow(cannera)
title('Canny')

% Depending on how overly detailed in terms of edge detection one wants,
% the Canny filter seems to have a better time in detecting edges.  If a
% simple outline of more defined edges is required, then the Sobel feature
% seems to only get the more deeper edges.

% Problem 13

salt = imnoise(camera, 'salt & pepper', 0.05);
gaus = imnoise(camera, 'gaussian');


sr = edge(salt,'roberts');
sp = edge(salt,'prewitt');
ss = edge(salt,'sobel');
laps = fspecial('laplacian',0);
sl = imfilter(salt,laps);
sz = edge(salt,'zerocross',[],laps);
marrlog = fspecial('log',13,2);
sm = edge(salt,'zerocross',[],marrlog);
sc = edge(salt,'canny');

gr = edge(camera,'roberts');
gp = edge(camera,'prewitt');
gs = edge(camera,'sobel');
laps = fspecial('laplacian',0);
gl = imfilter(camera,laps);
gz = edge(camera,'zerocross',[],laps);
marrlog = fspecial('log',13,2);
gm = edge(camera,'zerocross',[],marrlog);
gc = edge(camera,'canny');

figure
subplot(2,2,1)
imshow(sr)
title('Roberts')
subplot(2,2,2)
imshow(sp)
title('Prewitt')
subplot(2,2,3)
imshow(ss)
title('Sobel')
subplot(2,2,4)
imshow(sl)
title('Laplacian')

figure
subplot(2,2,1)
imshow(sz)
title('zero crossing')
subplot(2,2,2)
imshow(sm)
title('Marrilog')
subplot(2,2,3)
imshow(sc)
title('Canny')

figure
subplot(2,2,1)
imshow(gr)
title('Roberts')
subplot(2,2,2)
imshow(gp)
title('Prewitt')
subplot(2,2,3)
imshow(gs)
title('Sobel')
subplot(2,2,4)
imshow(gl)
title('Laplacian')

figure
subplot(2,2,1)
imshow(gz)
title('zero crossing')
subplot(2,2,2)
imshow(gm)
title('Marrilog')
subplot(2,2,3)
imshow(gc)
title('Canny')

% When salt and pepper noise is introducted in the image, the best edge
% detection tool seems to be the Prewitt filter.  There is still some noise
% that is detected, however compared to the others it has a more defined
% outline.  The zero crossing is absolutely the worse detection tool for
% this type of noise.  Causing the cameraman to completely disapear.
% Gaussian noise has more edge detections that work, however the Sobel
% filter has a crisper and less overally detailed edge detection.  Yet
% again,the zero crossing is the worst of the seven.
%% Part 3 Chapter 13 Problem 9

venice = imread('venice.png');
intense = rgb2ntsc(venice);
venice2 = zeros(size(venice));

venice3 = edge(venice(:,:,1)) | venice2(:,:,2) | venice2(:,:,3);
figure 
imshow(venice3)

% By using the standard edge function on the first venice image and logical
% OR combining of blackened out values, an edge value similar to figure ve2
% wa able to be taken.

gv = rgb2gray(venice);

vr = edge(gv,'roberts');
vp = edge(gv,'prewitt');
vs = edge(gv,'sobel');
laps = fspecial('laplacian',0);
vl = imfilter(gv,laps);
vz = edge(gv,'zerocross',[],laps);
marrlog = fspecial('log',13,2);
vm = edge(gv,'zerocross',[],marrlog);
vc = edge(gv,'canny');

figure
subplot(2,2,1)
imshow(vr)
title('Roberts')
subplot(2,2,2)
imshow(vp)
title('Prewitt')
subplot(2,2,3)
imshow(vs)
title('Sobel')
subplot(2,2,4)
imshow(vl)
title('Laplacian')

figure
subplot(2,2,1)
imshow(vz)
title('zero crossing')
subplot(2,2,2)
imshow(vm)
title('Merrilog')
subplot(2,2,3)
imshow(vc)
title('Canny')

% Out of the edge detetion methods, I personally think that merrilog and
% canny are very detailed in the amount of edges they can detect.  The
% other images either detect too few of edges or over detect edges (like
% zero crossing).