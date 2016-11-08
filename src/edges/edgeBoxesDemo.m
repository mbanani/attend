% Demo for Edge Boxes (please see readme.txt first).

%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

%% detect Edge Box bounding box proposals (see edgeBoxes.m)
I = imread('peppers.png');
tic, bbs=edgeBoxes(I,model,opts); toc

%% show evaluation results (using pre-defined or interactive boxes)
gt=[];

gt(:,5)=0; [gtRes,dtRes]=bbGt('evalRes',gt,double(bbs),.7);
figure(1); bbGt('showRes',I,gtRes,dtRes(dtRes(:,6)==1,:));
title('green=matched gt  red=missed gt  dashed-green=matched detect');

%% run and evaluate on entire dataset (see boxesData.m and boxesEval.m)
if(~exist('boxes/VOCdevkit/','dir')), return; end
split='val'; data=boxesData('split',split);
nm='EdgeBoxes70'; opts.name=['boxes/' nm '-' split '.mat'];
edgeBoxes(data.imgs,model,opts); opts.name=[];
