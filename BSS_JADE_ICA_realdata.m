clc; clear all; 
close all;


%Data_path = 'C:\Users\jys\Desktop\BSS and Antenna\Passive sonar\Program_final\Target_BSS_Program\SaveData';
%Prgm_path = 'C:\Users\jys\Desktop\BSS and Antenna\Passive sonar\Program_final\Target_BSS_Program';

%% Load
%cd(Data_path)
disp('Load...')
%% load target T1 T2 assignment1
load('T1'); Target_1 = T1(5,:);
load('T2'); Target_2 = T2(5,:);
%% load Mixed_sig assigment 1
%load('Mixed_sig'); Mixed_1 = Mixed_sig(18,:);
%load('Mixed_sig'); Mixed_2 = Mixed_sig(24,:);
%load('Mixed_sig'); Mixed_3 = Mixed_sig(17,:);
%load('Mixed_sig'); Mixed_4 = Mixed_sig(16,:);
%load('Mixed_sig'); Mixed_5 = Mixed_sig(19,:);
%load('Mixed_sig'); Mixed_6 = Mixed_sig(20,:);
%load('Mixed_sig'); Mixed_7 = Mixed_sig(21,:);
%% load DS_data assignment 2
load('DS_data3');Mixed_1 = BTR(85,:);
load('DS_data3');Mixed_2 = BTR(90,:);
load('DS_data3');Mixed_3 = BTR(95,:);
load('DS_data3');Mixed_4 = BTR(105,:);
load('DS_data3');Mixed_5 = BTR(100,:);
load('DS_data3');Mixed_6 = BTR(110,:);
load('DS_data3');Mixed_7 = BTR(115,:);

Fs = 16000;
time = 180;
t = 0:1/Fs:time; % Time (sec)
%% Set
%cd(Prgm_path)
% 좌표값 (표적 위치)
x = [-50 50 30];
% x2 = [-50 50];
% y = [300 350];
y = [5000 6000 6000];
% y2 = [500 500]; 

% 좌표값 (센서 위치)
x_sen = [5 2 7 8];
y_sen = [8 7 5 2];

p = 44.5; % Surface noise (0~4 stat: 44.5~66.5)

win = 2048; % window (1.024sec)
overlap = win*0.5; % overlap
nfft = 1024;

%% Mixing
%disp('Mixing...')
%[R1, R2,R3,R4,Td] = Mixing3_target(Target_1, Target_2,Target_3, Fs, x, y, x_sen, y_sen, p);
Tar = [Target_1; Target_2];
%R = [R1; R2; R3;R4];

%R = R(:,Td:end);


%% LOFAR
disp('LOFAR...')

dfs =4000; % Down sampling frequency
sc = Fs/dfs; % Down sampling rate
sc=2;
Ds_T = Tar;


Ds_T_deci_1 = decimate(Ds_T(1,:),4); % Down sampling 1
Ds_T_deci_2 = decimate(Ds_T(2,:),4); % Down sampling 2
%Ds_T_deci_3 = decimate(Ds_T(3,:),sc); % Down sampling 3
Ds_T_deci = [Ds_T_deci_1; Ds_T_deci_2]; % lofar (Resampling)

%tt = 0:1/dfs:(size(Ds_T_deci,3)-1)/dfs;

[Ds_T_STFT1,F_LOFAR_T,T_LOFAR_T] = stft(Ds_T_deci(1,:),win,overlap,nfft,dfs); % STFT 1
[Ds_T_STFT2,F_LOFAR_T,T_LOFAR_T] = stft(Ds_T_deci(2,:),win,overlap,nfft,dfs); % STFT 2
%[Ds_T_STFT3,F_LOFAR_T,T_LOFAR_T] = stft(Ds_T_deci(3,:),win,overlap,nfft,dfs); % STFT 3
Ds_T_STFT = cell(1,2); 

Ds_T_STFT{1,1} = Ds_T_STFT1; Ds_T_STFT{1,2} = Ds_T_STFT2;
%Ds_T_STFT{1,3} = Ds_T_STFT3; % LOFAR (STFT)
%% Target signal
figure,
set(gcf,'numbertitle','off','name', 'Target signal');
for ii = 1:size(Ds_T_STFT,2)
    subplot(2,1,ii)
      surf(T_LOFAR_T,F_LOFAR_T,10*log10(abs(Ds_T_STFT{1,ii}./max(abs(Ds_T_STFT{1,ii}))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-40 0])
    xlim([0 20])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
figure,
set(gcf,'numbertitle','off','name', ' frequency  target');
for ii = 1:size(Ds_T_STFT,2)
    subplot(2,1,ii)
      %plot(F_LOFAR_T,10*log10(abs(Ds_T_STFT{1,ii}))./max(abs(Ds_T_STFT{1,ii})));
      plot(F_LOFAR_T,db((abs(Ds_T_STFT{1,ii}(:,1)./max(abs(Ds_T_STFT{1,ii}(:,1)))))),'r-');
    %caxis([-20 0])
    xlim([100 2000])
    ylim([-40 0])
    xlabel('Frequency (Hz)','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R= [Mixed_1;Mixed_2;Mixed_3;Mixed_4;Mixed_5;Mixed_6;Mixed_7];
Ds = R;


Ds_deci_1 = decimate(Ds(1,:),sc); % Down sampling 1
Ds_deci_2 = decimate(Ds(2,:),sc); % Down sampling 2
Ds_deci_3 = decimate(Ds(3,:),sc); % Down sampling 3
Ds_deci_4 = decimate(Ds(4,:),sc); % Down sampling 4
Ds_deci_5 = decimate(Ds(5,:),sc); % Down sampling 5
Ds_deci_6 = decimate(Ds(6,:),sc); % Down sampling 6
Ds_deci_7 = decimate(Ds(7,:),sc); % Down sampling 6
Ds_deci = [Ds_deci_1; Ds_deci_2; Ds_deci_3;Ds_deci_4;Ds_deci_5;Ds_deci_6;Ds_deci_7]; %  (Resampling)
Ds_deci_ICA = [Ds_deci_1; Ds_deci_2; Ds_deci_3;Ds_deci_4];
tt = 0:1/dfs:(size(Ds_deci,4)-1)/dfs;

[Ds_STFT1,F_LOFAR,T_LOFAR] = stft(Ds_deci(1,:),win,overlap,nfft,dfs); % STFT 1
[Ds_STFT2,F_LOFAR,T_LOFAR] = stft(Ds_deci(2,:),win,overlap,nfft,dfs); % STFT 2
[Ds_STFT3,F_LOFAR,T_LOFAR] = stft(Ds_deci(3,:),win,overlap,nfft,dfs); % STFT 3
[Ds_STFT4,F_LOFAR,T_LOFAR] = stft(Ds_deci(4,:),win,overlap,nfft,dfs); % STFT 4
[Ds_STFT5,F_LOFAR,T_LOFAR] = stft(Ds_deci(5,:),win,overlap,nfft,dfs); % STFT 5
[Ds_STFT6,F_LOFAR,T_LOFAR] = stft(Ds_deci(6,:),win,overlap,nfft,dfs); % STFT 6
[Ds_STFT7,F_LOFAR,T_LOFAR] = stft(Ds_deci(7,:),win,overlap,nfft,dfs); % STFT 6
Ds_STFT = cell(1,2,3,4,5,6,7); 

Ds_STFT{1,1} = Ds_STFT1; Ds_STFT{1,2} = Ds_STFT2; Ds_STFT{1,3} = Ds_STFT3;
Ds_STFT{1,4} = Ds_STFT4;Ds_STFT{1,5} = Ds_STFT5;Ds_STFT{1,6} = Ds_STFT6;% LOFAR (STFT)
Ds_STFT{1,7} = Ds_STFT7;
%% Received signal
figure,
set(gcf,'numbertitle','off','name', 'Recieved signal');
surf(T_LOFAR,F_LOFAR,10*log10(abs(Ds_STFT{1,1}./max(abs(Ds_STFT{1,1}))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-40 0])
    %xlim([0 20])
    ylim([100 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
      figure,
set(gcf,'numbertitle','off','name', ' frequency  received');

      plot(F_LOFAR_T,abs(Ds_STFT{1,1}(:,1)),'r-');
    %caxis([-20 0])
    xlim([100 2000])
    %ylim([-40 0])
    xlabel('Frequency (Hz)','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')


%% JADE (Frequency domain)
disp('FD-JADE...')
FD_sep1_F = []; FD_sep2_F = []; FD_sep_F = cell(1,2);
for i = 1:size(Ds_STFT{1,1},2)
    ST = [Ds_STFT{1,1}(:,i)'; Ds_STFT{1,2}(:,i)'];
    [Ae, FD_jade] = jade(ST,2);
    FD_sep1_F = [FD_sep1_F FD_jade(1,:)'];
    FD_sep2_F = [FD_sep2_F FD_jade(2,:)'];
   % FD_sep3_F = [FD_sep3_F FD_jade(3,:)'];
end
clear i

FD_sep_F{1,1} = FD_sep1_F; FD_sep_F{1,2} = FD_sep2_F;  % STFT (Seperated)


%% Figure FD - JADE
figure,
set(gcf,'numbertitle','off','name', 'FD-JADE');
for ii = 1:size(FD_sep_F,2)
    subplot(2,1,ii)
    surf(T_LOFAR,F_LOFAR,10*log10(abs(FD_sep_F{1,ii})./max(abs(FD_sep_F{1,ii})))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-40 0])
    %xlim([0 20])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
figure,
set(gcf,'numbertitle','off','name', '   FDJADE');
for ii = 1:size(FD_sep_F,2)
    subplot(2,1,ii)
      plot(F_LOFAR_T,(db(abs(FD_sep_F{1,ii}(:,4)./max(abs(FD_sep_F{1,ii}(:,4)))))),'r-');
    %caxis([-20 0])
    xlim([100 2000])
    ylim([-40 0])
    xlabel('Frequency (Hz) - FDJADE','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%% TD-JADE
disp('TD-JADE...')
TD_sep1_F = []; TD_sep2_F = [];TD_sep3_F = [];TD_sep4_F = []; TD_sep_F = cell(1,2);

    ST = [Ds_deci(1,:); Ds_deci(2,:); Ds_deci(3,:); Ds_deci(4,:)];
    [Ae, TD_jade] = jade(ST,4);
    TD_sep1_F = [TD_sep1_F TD_jade(1,:)'];
    TD_sep2_F = [TD_sep2_F TD_jade(2,:)'];
    TD_sep3_F = [TD_sep3_F TD_jade(3,:)'];
    TD_sep4_F = [TD_sep4_F TD_jade(4,:)'];
   TD_sep_F{1,1} = TD_sep1_F; TD_sep_F{1,2} = TD_sep2_F;
   TD_sep_F{1,3} = TD_sep3_F; TD_sep_F{1,4} = TD_sep4_F;
   TD_STFT = cell(1,2,3,4); 

[TD_STFT1,F_LOFAR,T_LOFAR] = stft(TD_sep1_F,win,overlap,nfft,dfs); % STFT 1
[TD_STFT2,F_LOFAR,T_LOFAR] = stft(TD_sep2_F,win,overlap,nfft,dfs); % STFT 2
[TD_STFT3,F_LOFAR,T_LOFAR] = stft(TD_sep3_F,win,overlap,nfft,dfs); % STFT 3
[TD_STFT4,F_LOFAR,T_LOFAR] = stft(TD_sep4_F,win,overlap,nfft,dfs); % STFT 2
 TD_STFT{1,1} = TD_STFT1; TD_STFT{1,2} = TD_STFT2; 
  TD_STFT{1,3} = TD_STFT3; TD_STFT{1,4} = TD_STFT4;
 %(STFT)
%% TD-JADE figures
figure,
set(gcf,'numbertitle','off','name', 'TD JADE ');
for ii = 1:size(TD_STFT,4)
    subplot(4,1,ii)
surf(T_LOFAR,F_LOFAR,10*log10(abs(TD_STFT{1,ii}./max(abs(TD_STFT{1,ii}))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-45 -30])
    %xlim([0 20])
    ylim([20 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('TDJADE (hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end 

      figure,
set(gcf,'numbertitle','off','name', ' TD JADE');
for ii = 1:size(TD_STFT,4)
    subplot(4,1,ii)
      plot(F_LOFAR_T,db((abs(TD_STFT{1,ii}(:,1)./max(abs(TD_STFT{1,ii}(:,1)))))),'r-')
    %caxis([-20 0])
    xlim([200 2000])
    ylim([-45 -30])
    xlabel('Frequency (Hz) - TDJADE','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%% ICA 4 ICs
[N,P]=size(Mixed_1); 
%mixes=Ds_deci;
mixes=Ds_deci_ICA; 
mx=mean(mixes'); c=cov(mixes');
%x=x-mx'*ones(1,P);                   % subtract means from mixes
wz=2*inv(sqrtm(c));                  % get decorrelating matrix
%x=wz*x;                              % decorrelate mixes so cov(x')=4*eye(N);

%**** 
%w=[1 1; 1 2];                       % init. unmixing matrix, or w=rand(M,N);
w=eye(N);                            % init. unmixing matrix, or w=rand(M,N);
M=size(w,2);                            % M=N usually
sweep=0; oldw=w; olddelta=ones(1,N*N);
Id=eye(M);


uu=w*wz*mixes;            % make unmixed sources
[ICA_STFT1,F_LOFAR,T_LOFAR] = stft(uu(1,:),win,overlap,nfft,dfs); % STFT 1
[ICA_STFT2,F_LOFAR,T_LOFAR] = stft(uu(2,:),win,overlap,nfft,dfs); % STFT 2
[ICA_STFT3,F_LOFAR,T_LOFAR] = stft(uu(3,:),win,overlap,nfft,dfs); %STFT 3
[ICA_STFT4,F_LOFAR,T_LOFAR] = stft(uu(4,:),win,overlap,nfft,dfs); %STFT 4

  ICA_STFT = cell(1,2,3,4); 
  ICA_STFT{1,1} = ICA_STFT1; ICA_STFT{1,2} = ICA_STFT2; ICA_STFT{1,3} = ICA_STFT3; ICA_STFT{1,4} = ICA_STFT4;%(STFT)
  %%  ICA 4IC figure
figure,
set(gcf,'numbertitle','off','name', ' ICA  4IC');
for ii = 1:size(ICA_STFT,4)
    subplot(4,1,ii)
surf(T_LOFAR,F_LOFAR,db(abs(ICA_STFT{1,ii})./max(abs(ICA_STFT{1,ii})))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-45 -30])
    %xlim([0 20])
    ylim([20 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('  ICA (Hz) ','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
     figure,
set(gcf,'numbertitle','off','name', '  ICA 4IC');
for ii = 1:size(ICA_STFT,4)
    subplot(4,1,ii)
      %plot(F_LOFAR_T,(abs(ICA_STFT{1,ii}(:,1))./max(abs(ICA_STFT{1,ii}(:,1)))),'r-');
      plot(F_LOFAR_T,abs(ICA_STFT{1,ii}(:,1)),'r-');
    %caxis([-20 0])
    xlim([150 2000])
    %ylim([-65 -45])
    xlabel('Frequency (Hz) ICA','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end




%% fast ICA 2IC
%[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_STFT{1,4},2,'kurtosis',1);
[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_deci_ICA,2,'negentropy',0);
[f_ICA_STFT1,F_LOFAR,T_LOFAR] = stft(Zica(2,:),win,overlap,nfft,dfs); % STFT 1
[f_ICA_STFT2,F_LOFAR,T_LOFAR] = stft(Zica(1,:),win,overlap,nfft,dfs); % STFT 2


  f_ICA_STFT = cell(1,2); 
 f_ICA_STFT{1,1} = f_ICA_STFT1; f_ICA_STFT{1,2} = f_ICA_STFT2;%(STFT)


%% fast ICA  2ICfigure
figure,
set(gcf,'numbertitle','off','name', 'fast ICA 2IC ');
for ii = 1:size(f_ICA_STFT,2)
    subplot(2,1,ii)
surf(T_LOFAR,F_LOFAR,abs(f_ICA_STFT{1,ii})...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    %caxis([-40 0])
    %xlim([0 50])
    ylim([20 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)- fast ICA 2IC','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
     figure,
set(gcf,'numbertitle','off','name', ' fast ICA 2IC');
for ii = 1:size(f_ICA_STFT,2)
    subplot(2,1,ii)
      plot(F_LOFAR_T,abs(f_ICA_STFT{1,ii}(:,1))./max(abs(f_ICA_STFT{1,ii}(:,1))),'r-');
    %caxis([-20 0])
    xlim([20 2000])
    %ylim([0 50])
    xlabel('Frequency (Hz) - fast ICA 2IC','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%% fast ICA 3IC
%[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_STFT{1,4},2,'kurtosis',1);
[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_deci_ICA,3,'negentropy',0);
[f_ICA_STFT1,F_LOFAR,T_LOFAR] = stft(Zica(2,:),win,overlap,nfft,dfs); % STFT 1
[f_ICA_STFT2,F_LOFAR,T_LOFAR] = stft(Zica(1,:),win,overlap,nfft,dfs); % STFT 2
[f_ICA_STFT3,F_LOFAR,T_LOFAR] = stft(Zica(3,:),win,overlap,nfft,dfs); %STFT 3


  f_ICA_STFT = cell(1,2,3); 
 f_ICA_STFT{1,1} = f_ICA_STFT1; f_ICA_STFT{1,2} = f_ICA_STFT2; f_ICA_STFT{1,3} = f_ICA_STFT3; %(STFT)


%% fast ICA  3ICfigure
figure,
set(gcf,'numbertitle','off','name', 'fast ICA  3IC');
for ii = 1:size(f_ICA_STFT,3)
    subplot(3,1,ii)
surf(T_LOFAR,F_LOFAR,abs(f_ICA_STFT{1,ii})...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    %caxis([-40 0])
    %xlim([0 20])
    ylim([20 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)- fast ICA 3IC','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
     figure,
set(gcf,'numbertitle','off','name', ' fast ICA 3IC');
for ii = 1:size(f_ICA_STFT,3)
    subplot(3,1,ii)
      plot(F_LOFAR_T,abs(f_ICA_STFT{1,ii}(:,1))./max(abs(f_ICA_STFT{1,ii}(:,1))),'r-');
    %caxis([-20 0])
    xlim([20 2000])
    %ylim([0 50])
    xlabel('Frequency (Hz) - fast ICA','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%% fast ICA 4IC
%[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_STFT{1,4},2,'kurtosis',1);
[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_deci_ICA,4,'negentropy',0);
[f_ICA_STFT1,F_LOFAR,T_LOFAR] = stft(Zica(1,:),win,overlap,nfft,dfs); % STFT 1
[f_ICA_STFT2,F_LOFAR,T_LOFAR] = stft(Zica(2,:),win,overlap,nfft,dfs); % STFT 2
[f_ICA_STFT3,F_LOFAR,T_LOFAR] = stft(Zica(3,:),win,overlap,nfft,dfs); %STFT 3
[f_ICA_STFT4,F_LOFAR,T_LOFAR] = stft(Zica(4,:),win,overlap,nfft,dfs); %STFT 4

  f_ICA_STFT = cell(1,2,3,4); 
 f_ICA_STFT{1,1} = f_ICA_STFT1; f_ICA_STFT{1,2} = f_ICA_STFT2; f_ICA_STFT{1,3} = f_ICA_STFT3; f_ICA_STFT{1,4} = f_ICA_STFT4;%(STFT)


%% fast ICA 4IC figure
figure,
set(gcf,'numbertitle','off','name', 'fast ICA  4IC');
for ii = 1:size(f_ICA_STFT,4)
    subplot(4,1,ii)
surf(T_LOFAR,F_LOFAR,db(abs(f_ICA_STFT{1,ii})./max(abs(f_ICA_STFT{1,ii})))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-45 -30])
    %xlim([0 20])
    ylim([20 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel(' fast ICA (Hz) ','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
     figure,
set(gcf,'numbertitle','off','name', ' fast ICA 4IC');
for ii = 1:size(f_ICA_STFT,4)
    subplot(4,1,ii)
      plot(F_LOFAR_T,db(abs(f_ICA_STFT{1,ii}(:,1))./max(abs(f_ICA_STFT{1,ii}(:,1)))),'r-');
    %caxis([-20 0])
    xlim([100 2000])
    ylim([-45 -30])
    xlabel('Frequency (Hz) - fast ICA','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%% fast ICA 5IC
%[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_STFT{1,4},2,'kurtosis',1);
[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_deci_ICA,5,'negentropy',0);
[f_ICA_STFT1,F_LOFAR,T_LOFAR] = stft(Zica(2,:),win,overlap,nfft,dfs); % STFT 1
[f_ICA_STFT2,F_LOFAR,T_LOFAR] = stft(Zica(1,:),win,overlap,nfft,dfs); % STFT 2
[f_ICA_STFT3,F_LOFAR,T_LOFAR] = stft(Zica(3,:),win,overlap,nfft,dfs); %STFT 3
[f_ICA_STFT4,F_LOFAR,T_LOFAR] = stft(Zica(4,:),win,overlap,nfft,dfs); %STFT 4
[f_ICA_STFT5,F_LOFAR,T_LOFAR] = stft(Zica(5,:),win,overlap,nfft,dfs); %STFT 5
  f_ICA_STFT = cell(1,2,3,4,5); 
 f_ICA_STFT{1,1} = f_ICA_STFT1; f_ICA_STFT{1,2} = f_ICA_STFT2; f_ICA_STFT{1,3} = f_ICA_STFT3; f_ICA_STFT{1,4} = f_ICA_STFT4;
 f_ICA_STFT{1,5} = f_ICA_STFT5;%(STFT)


%% fast ICA 5IC figure
figure,
set(gcf,'numbertitle','off','name', 'fast ICA  5IC');
for ii = 1:size(f_ICA_STFT,5)
    subplot(5,1,ii)
surf(T_LOFAR,F_LOFAR,abs(f_ICA_STFT{1,ii})...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    %caxis([-40 0])
    %xlim([0 20])
    ylim([20 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)- fast ICA 5IC','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
     figure,
set(gcf,'numbertitle','off','name', ' fast ICA 5IC');
for ii = 1:size(f_ICA_STFT,5)
    subplot(5,1,ii)
      plot(F_LOFAR_T,abs(f_ICA_STFT{1,ii}(:,1))./max(abs(f_ICA_STFT{1,ii}(:,1))),'r-');
    %caxis([-20 0])
    xlim([20 2000])
    %ylim([0 50])
    xlabel('Frequency (Hz) - fast ICA','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end

%% JADE (Frequency domain 2 IC)
disp('FD-JADE...2IC')
FD_sep1_F_2IC = []; FD_sep2_F_2IC = [];  FD_sep_F_2IC = cell(1,2);
for i = 1:size(Ds_STFT{1,1},2)
    ST = [Ds_STFT{1,1}(:,i)'; Ds_STFT{1,2}(:,i)'];
    [Ae, FD_jade] = jade(ST,2);
    FD_sep1_F_2IC = [FD_sep1_F_2IC FD_jade(1,:)'];
    FD_sep2_F_2IC = [FD_sep2_F_2IC FD_jade(2,:)'];
   
end
clear i

FD_sep_F_2IC{1,1} = FD_sep1_F_2IC; FD_sep_F_2IC{1,2} = FD_sep2_F_2IC; % STFT (Seperated)

%% JADE (Frequency domain 3IC)
disp('FD-JADE...3IC')
FD_sep1_F_3IC = []; FD_sep2_F_3IC = [];FD_sep3_F_3IC = [];  FD_sep_F_3IC = cell(1,2,3);
for i = 1:size(Ds_STFT{1,1},2)
    ST = [Ds_STFT{1,1}(:,i)'; Ds_STFT{1,2}(:,i)';Ds_STFT{1,3}(:,i)'];
    [Ae, FD_jade] = jade(ST,3);
    FD_sep1_F_3IC = [FD_sep1_F_3IC FD_jade(1,:)'];
    FD_sep2_F_3IC = [FD_sep2_F_3IC FD_jade(2,:)'];
    FD_sep3_F_3IC = [FD_sep3_F_3IC FD_jade(3,:)'];
   
end
clear i

FD_sep_F_3IC{1,1} = FD_sep1_F_3IC; FD_sep_F_3IC{1,2} = FD_sep2_F_3IC;FD_sep_F_3IC{1,3} = FD_sep3_F_3IC; % STFT (Seperated)

%% JADE (Frequency domain 4 IC)
disp('FD-JADE...4IC')
FD_sep1_F_4IC = []; FD_sep2_F_4IC = [];FD_sep3_F_4IC = []; FD_sep4_F_4IC = []; FD_sep_F_4IC = cell(1,2,3,4);
for i = 1:size(Ds_STFT{1,1},2)
    ST = [Ds_STFT{1,1}(:,i)'; Ds_STFT{1,2}(:,i)';Ds_STFT{1,3}(:,i)';Ds_STFT{1,4}(:,i)'];
    [Ae, FD_jade] = jade(ST,4);
    FD_sep1_F_4IC = [FD_sep1_F_4IC FD_jade(1,:)'];
    FD_sep2_F_4IC = [FD_sep2_F_4IC FD_jade(2,:)'];
    FD_sep3_F_4IC = [FD_sep3_F_4IC FD_jade(3,:)'];
    FD_sep4_F_4IC = [FD_sep4_F_4IC FD_jade(4,:)'];
   
end
clear i

FD_sep_F_4IC{1,1} = FD_sep1_F_4IC; FD_sep_F_4IC{1,2} = FD_sep2_F_4IC;FD_sep_F_4IC{1,3} = FD_sep3_F_4IC; FD_sep_F_4IC{1,4} = FD_sep4_F_4IC;% STFT (Seperated)


%% JADE (Frequency domain 5 IC)
disp('FD-JADE...5IC')
FD_sep1_F_5IC = []; FD_sep2_F_5IC = [];FD_sep3_F_5IC = []; FD_sep4_F_5IC = [];  FD_sep5_F_5IC = [];FD_sep_F_5IC = cell(1,2,3,4,5);
for i = 1:size(Ds_STFT{1,1},2)
    ST = [Ds_STFT{1,1}(:,i)'; Ds_STFT{1,2}(:,i)';Ds_STFT{1,3}(:,i)';Ds_STFT{1,4}(:,i)';Ds_STFT{1,5}(:,i)'];
    [Ae, FD_jade] = jade(ST,5);
    FD_sep1_F_5IC = [FD_sep1_F_5IC FD_jade(1,:)'];
    FD_sep2_F_5IC = [FD_sep2_F_5IC FD_jade(2,:)'];
    FD_sep3_F_5IC = [FD_sep3_F_5IC FD_jade(3,:)'];
    FD_sep4_F_5IC = [FD_sep4_F_5IC FD_jade(4,:)'];
    FD_sep5_F_5IC = [FD_sep5_F_5IC FD_jade(5,:)'];
   
end
clear i

FD_sep_F_5IC{1,1} = FD_sep1_F_5IC; FD_sep_F_5IC{1,2} = FD_sep2_F_5IC;FD_sep_F_5IC{1,3} = FD_sep3_F_5IC; FD_sep_F_5IC{1,4} = FD_sep4_F_5IC;FD_sep_F_5IC{1,5} = FD_sep5_F_5IC;% STFT (Seperated)

%% JADE (Frequency domain 6 IC)
disp('FD-JADE...6IC')
FD_sep1_F_6IC = []; FD_sep2_F_6IC = [];FD_sep3_F_6IC = []; FD_sep4_F_6IC = [];  FD_sep5_F_6IC = [];FD_sep6_F_6IC = [];FD_sep_F_6IC = cell(1,2,3,4,5,6);
for i = 1:size(Ds_STFT{1,1},2)
    ST = [Ds_STFT{1,1}(:,i)'; Ds_STFT{1,2}(:,i)';Ds_STFT{1,3}(:,i)';Ds_STFT{1,4}(:,i)';Ds_STFT{1,5}(:,i)';Ds_STFT{1,6}(:,i)'];
    [Ae, FD_jade] = jade(ST,6);
    FD_sep1_F_6IC = [FD_sep1_F_6IC FD_jade(1,:)'];
    FD_sep2_F_6IC = [FD_sep2_F_6IC FD_jade(2,:)'];
    FD_sep3_F_6IC = [FD_sep3_F_6IC FD_jade(3,:)'];
    FD_sep4_F_6IC = [FD_sep4_F_6IC FD_jade(4,:)'];
    FD_sep5_F_6IC = [FD_sep5_F_6IC FD_jade(5,:)'];
    FD_sep6_F_6IC = [FD_sep6_F_6IC FD_jade(6,:)'];
   
end
clear i

FD_sep_F_6IC{1,1} = FD_sep1_F_6IC; FD_sep_F_6IC{1,2} = FD_sep2_F_6IC;
FD_sep_F_6IC{1,3} = FD_sep3_F_6IC; FD_sep_F_6IC{1,4} = FD_sep4_F_6IC;
FD_sep_F_6IC{1,5} = FD_sep5_F_6IC; FD_sep_F_6IC{1,6} = FD_sep6_F_6IC;% STFT (Seperated)
%% JADE (Frequency domain 7 IC)
disp('FD-JADE...7IC')
FD_sep1_F_7IC = []; FD_sep2_F_7IC = [];FD_sep3_F_7IC = []; 
FD_sep4_F_7IC = []; FD_sep5_F_7IC = [];FD_sep6_F_7IC = [];
FD_sep7_F_7IC = [];
FD_sep_F_7IC = cell(1,2,3,4,5,6,7);
for i = 1:size(Ds_STFT{1,1},2)
    ST = [Ds_STFT{1,1}(:,i)'; Ds_STFT{1,2}(:,i)';Ds_STFT{1,3}(:,i)';Ds_STFT{1,4}(:,i)';Ds_STFT{1,5}(:,i)';Ds_STFT{1,6}(:,i)';Ds_STFT{1,7}(:,i)'];
    [Ae, FD_jade] = jade(ST,7);
    FD_sep1_F_7IC = [FD_sep1_F_7IC FD_jade(1,:)'];
    FD_sep2_F_7IC = [FD_sep2_F_7IC FD_jade(2,:)'];
    FD_sep3_F_7IC = [FD_sep3_F_7IC FD_jade(3,:)'];
    FD_sep4_F_7IC = [FD_sep4_F_7IC FD_jade(4,:)'];
    FD_sep5_F_7IC = [FD_sep5_F_7IC FD_jade(5,:)'];
    FD_sep6_F_7IC = [FD_sep6_F_7IC FD_jade(6,:)'];
    FD_sep7_F_7IC = [FD_sep6_F_7IC FD_jade(7,:)'];
   
end
clear i

FD_sep_F_7IC{1,1} = FD_sep1_F_7IC; FD_sep_F_7IC{1,2} = FD_sep2_F_7IC;
FD_sep_F_7IC{1,3} = FD_sep3_F_7IC; FD_sep_F_7IC{1,4} = FD_sep4_F_7IC;
FD_sep_F_7IC{1,5} = FD_sep5_F_7IC; FD_sep_F_7IC{1,6} = FD_sep6_F_7IC;% STFT (Seperated)


%% corrcoef calculation
%% correlation coefficient of JADE 1 output and 2 outputs
% 1x2 = 2 coefficients
 r1_21=corrcoef(Ds_STFT1,FD_sep1_F_2IC)
 r1_22=corrcoef(Ds_STFT1,FD_sep2_F_2IC)
 %% correlation coefficient of JADE 2 outputs and JADE 3 outputs: 
 % 2x3 = 6 coefficients
 r21_31= corrcoef(FD_sep1_F_2IC, FD_sep1_F_3IC)
 r21_32= corrcoef(FD_sep1_F_2IC, FD_sep2_F_3IC)
 r21_33= corrcoef(FD_sep1_F_2IC, FD_sep3_F_3IC)
 r22_31= corrcoef(FD_sep2_F_2IC, FD_sep1_F_3IC)
 r22_32= corrcoef(FD_sep2_F_2IC, FD_sep2_F_3IC)
 r22_33= corrcoef(FD_sep2_F_2IC, FD_sep3_F_3IC)
  %% correlation coefficient of JADE 3 outputs and JADE 4 outputs: 
 % 3x4 = 12 coefficients
  r31_41= corrcoef(FD_sep1_F_3IC, FD_sep1_F_4IC)
  r31_42= corrcoef(FD_sep1_F_3IC, FD_sep2_F_4IC)
  r31_43= corrcoef(FD_sep1_F_3IC, FD_sep3_F_4IC)
  r31_44= corrcoef(FD_sep1_F_3IC, FD_sep4_F_4IC)
  r32_41= corrcoef(FD_sep2_F_3IC, FD_sep1_F_4IC)
  r32_42= corrcoef(FD_sep2_F_3IC, FD_sep2_F_4IC)
  r32_43= corrcoef(FD_sep2_F_3IC, FD_sep3_F_4IC)
  r32_44= corrcoef(FD_sep2_F_3IC, FD_sep4_F_4IC)
  r33_41= corrcoef(FD_sep3_F_3IC, FD_sep1_F_4IC)
  r33_42= corrcoef(FD_sep3_F_3IC, FD_sep2_F_4IC)
  r33_43= corrcoef(FD_sep3_F_3IC, FD_sep3_F_4IC)
  r33_44= corrcoef(FD_sep3_F_3IC, FD_sep4_F_4IC)
  %% correlation coefficient of JADE 4 outputs and JADE 5 outputs: 
 % 4x5 = 20 coefficients
  r41_51= corrcoef(FD_sep1_F_4IC, FD_sep1_F_5IC)
  r41_52= corrcoef(FD_sep1_F_4IC, FD_sep2_F_5IC)
  r41_53= corrcoef(FD_sep1_F_4IC, FD_sep3_F_5IC)
  r41_54= corrcoef(FD_sep1_F_4IC, FD_sep4_F_5IC)
  r41_55= corrcoef(FD_sep1_F_4IC, FD_sep5_F_5IC)
  r42_51= corrcoef(FD_sep2_F_4IC, FD_sep1_F_5IC)
  r42_52= corrcoef(FD_sep2_F_4IC, FD_sep2_F_5IC)
  r42_53= corrcoef(FD_sep2_F_4IC, FD_sep3_F_5IC)
  r42_54= corrcoef(FD_sep2_F_4IC, FD_sep4_F_5IC)
  r42_55= corrcoef(FD_sep2_F_4IC, FD_sep5_F_5IC)
  r43_51= corrcoef(FD_sep3_F_4IC, FD_sep1_F_5IC)
  r43_52= corrcoef(FD_sep3_F_4IC, FD_sep2_F_5IC)
  r43_53= corrcoef(FD_sep3_F_4IC, FD_sep3_F_5IC)
  r43_54= corrcoef(FD_sep3_F_4IC, FD_sep4_F_5IC)
  r43_55= corrcoef(FD_sep3_F_4IC, FD_sep5_F_5IC)
  r44_51= corrcoef(FD_sep4_F_4IC, FD_sep1_F_5IC)
  r44_52= corrcoef(FD_sep4_F_4IC, FD_sep2_F_5IC)
  r44_53= corrcoef(FD_sep4_F_4IC, FD_sep3_F_5IC)
  r44_54= corrcoef(FD_sep4_F_4IC, FD_sep4_F_5IC)
  r44_55= corrcoef(FD_sep4_F_4IC, FD_sep5_F_5IC)
    %% correlation coefficient of JADE 5 outputs and JADE 6 outputs: 
 % 5x6 = 30 coefficients
  r51_61= corrcoef(FD_sep1_F_5IC, FD_sep1_F_6IC)
  r51_62= corrcoef(FD_sep1_F_5IC, FD_sep2_F_6IC)
  r51_63= corrcoef(FD_sep1_F_5IC, FD_sep3_F_6IC)
  r51_64= corrcoef(FD_sep1_F_5IC, FD_sep4_F_6IC)
  r51_65= corrcoef(FD_sep1_F_5IC, FD_sep5_F_6IC)
  r51_66= corrcoef(FD_sep1_F_5IC, FD_sep6_F_6IC)
  r52_61= corrcoef(FD_sep2_F_5IC, FD_sep1_F_6IC)
  r52_62= corrcoef(FD_sep2_F_5IC, FD_sep2_F_6IC)
  r52_63= corrcoef(FD_sep2_F_5IC, FD_sep3_F_6IC)
  r52_64= corrcoef(FD_sep2_F_5IC, FD_sep4_F_6IC)
  r52_65= corrcoef(FD_sep2_F_5IC, FD_sep5_F_6IC)
  r52_66= corrcoef(FD_sep2_F_5IC, FD_sep6_F_6IC)
  r53_61= corrcoef(FD_sep3_F_5IC, FD_sep1_F_6IC)
  r53_62= corrcoef(FD_sep3_F_5IC, FD_sep2_F_6IC)
  r53_63= corrcoef(FD_sep3_F_5IC, FD_sep3_F_6IC)
  r53_64= corrcoef(FD_sep3_F_5IC, FD_sep4_F_6IC)
  r53_65= corrcoef(FD_sep3_F_5IC, FD_sep5_F_6IC)
  r53_66= corrcoef(FD_sep3_F_5IC, FD_sep6_F_6IC)
  r54_61= corrcoef(FD_sep4_F_5IC, FD_sep1_F_6IC)
  r54_62= corrcoef(FD_sep4_F_5IC, FD_sep2_F_6IC)
  r54_63= corrcoef(FD_sep4_F_5IC, FD_sep3_F_6IC)
  r54_64= corrcoef(FD_sep4_F_5IC, FD_sep4_F_6IC)
  r54_65= corrcoef(FD_sep4_F_5IC, FD_sep5_F_6IC)
  r54_66= corrcoef(FD_sep4_F_5IC, FD_sep6_F_6IC)
  r55_61= corrcoef(FD_sep5_F_5IC, FD_sep1_F_6IC)
  r55_62= corrcoef(FD_sep5_F_5IC, FD_sep2_F_6IC)
  r55_63= corrcoef(FD_sep5_F_5IC, FD_sep3_F_6IC)
  r55_64= corrcoef(FD_sep5_F_5IC, FD_sep4_F_6IC)
  r55_65= corrcoef(FD_sep5_F_5IC, FD_sep5_F_6IC)
  r55_66= corrcoef(FD_sep5_F_5IC, FD_sep6_F_6IC)
     %% correlation coefficient of JADE 6 outputs and JADE 7 outputs: 
 % 6x7 = 42 coefficients
  r61_71= corrcoef(FD_sep1_F_6IC, FD_sep1_F_7IC)
  r61_72= corrcoef(FD_sep1_F_6IC, FD_sep2_F_7IC)
  r61_73= corrcoef(FD_sep1_F_6IC, FD_sep3_F_7IC)
  r61_74= corrcoef(FD_sep1_F_6IC, FD_sep4_F_7IC)
  r61_75= corrcoef(FD_sep1_F_6IC, FD_sep5_F_7IC)
  r61_76= corrcoef(FD_sep1_F_6IC, FD_sep6_F_7IC)
  %r61_77= corrcoef(FD_sep1_F_6IC, FD_sep7_F_7IC)
  r62_71= corrcoef(FD_sep2_F_6IC, FD_sep1_F_7IC)
  r62_72= corrcoef(FD_sep2_F_6IC, FD_sep2_F_7IC)
  r62_73= corrcoef(FD_sep2_F_6IC, FD_sep3_F_7IC)
  r62_74= corrcoef(FD_sep2_F_6IC, FD_sep4_F_7IC)
  r62_75= corrcoef(FD_sep2_F_6IC, FD_sep5_F_7IC)
  r62_76= corrcoef(FD_sep2_F_6IC, FD_sep6_F_7IC)
  %r62_77= corrcoef(FD_sep2_F_6IC, FD_sep7_F_7IC)
  r63_71= corrcoef(FD_sep3_F_6IC, FD_sep1_F_7IC)
  r63_72= corrcoef(FD_sep3_F_6IC, FD_sep2_F_7IC)
  r63_73= corrcoef(FD_sep3_F_6IC, FD_sep3_F_7IC)
  r63_74= corrcoef(FD_sep3_F_6IC, FD_sep4_F_7IC)
  r63_75= corrcoef(FD_sep3_F_6IC, FD_sep5_F_7IC)
  r63_76= corrcoef(FD_sep3_F_6IC, FD_sep6_F_7IC)
  %r63_77= corrcoef(FD_sep3_F_6IC, FD_sep7_F_7IC)
  r64_71= corrcoef(FD_sep4_F_6IC, FD_sep1_F_7IC)
  r64_72= corrcoef(FD_sep4_F_6IC, FD_sep2_F_7IC)
  r64_73= corrcoef(FD_sep4_F_6IC, FD_sep3_F_7IC)
  r64_74= corrcoef(FD_sep4_F_6IC, FD_sep4_F_7IC)
  r64_75= corrcoef(FD_sep4_F_6IC, FD_sep5_F_7IC)
  r64_76= corrcoef(FD_sep4_F_6IC, FD_sep6_F_7IC)
  %r64_77= corrcoef(FD_sep4_F_6IC, FD_sep7_F_7IC)
  r65_71= corrcoef(FD_sep5_F_6IC, FD_sep1_F_7IC)
  r65_72= corrcoef(FD_sep5_F_6IC, FD_sep2_F_7IC)
  r65_73= corrcoef(FD_sep5_F_6IC, FD_sep3_F_7IC)
  r65_74= corrcoef(FD_sep5_F_6IC, FD_sep4_F_7IC)
  r65_75= corrcoef(FD_sep5_F_6IC, FD_sep5_F_7IC)
  r65_76= corrcoef(FD_sep5_F_6IC, FD_sep6_F_7IC)
  %r65_77= corrcoef(FD_sep5_F_6IC, FD_sep7_F_7IC)
  r66_71= corrcoef(FD_sep6_F_6IC, FD_sep1_F_7IC)
  r66_72= corrcoef(FD_sep6_F_6IC, FD_sep2_F_7IC)
  r66_73= corrcoef(FD_sep6_F_6IC, FD_sep3_F_7IC)
  r66_74= corrcoef(FD_sep6_F_6IC, FD_sep4_F_7IC)
  r66_75= corrcoef(FD_sep6_F_6IC, FD_sep5_F_7IC)
  r66_76= corrcoef(FD_sep6_F_6IC, FD_sep6_F_7IC)
  %r66_77= corrcoef(FD_sep6_F_6IC, FD_sep7_F_7IC)
  
%% Figure FD - JADE ICs 


figure,
set(gcf,'numbertitle','off','name', ' frequency FD-JADE 2IC');
for ii = 1:size(FD_sep_F_2IC,2)
    subplot(2,1,ii)
    %plot(F_LOFAR,abs(FD_sep_F_2IC{1,ii}(:,4)));
    plot(F_LOFAR_T,db(abs(FD_sep_F_2IC{1,ii}(:,1))./max(abs(FD_sep_F_2IC{1,ii}(:,1)))),'r-');
    xlabel('Frequency (Hz) FDJADE','fontsize',6);
    ylabel('Amplitude','fontsize',6);
    xlim([20 2000])
    ylim([-5 0])
    set(gca,'fontsize',12)
    set(gcf,'color','w')
    grid on
end
figure,
set(gcf,'numbertitle','off','name', ' frequency FD-JADE 3IC');
for ii = 1:size(FD_sep_F_3IC,3)
    subplot(3,1,ii)
    %plot(F_LOFAR,abs(FD_sep_F_3IC{1,ii}(:,1)));
    plot(F_LOFAR_T,db(abs(FD_sep_F_3IC{1,ii}(:,1))./max(abs(FD_sep_F_3IC{1,ii}(:,1)))),'r-');
    xlabel('Frequency (Hz) FDJADE','fontsize',6);
    ylabel('Amplitude','fontsize',6);
    xlim([20 2000])
    ylim([-5 0])
    set(gca,'fontsize',12)
    set(gcf,'color','w')
    grid on
end
figure,
set(gcf,'numbertitle','off','name', 'FDJADE  4IC');
for ii = 1:size(FD_sep_F_4IC,4)
    subplot(4,1,ii)
surf(T_LOFAR,F_LOFAR,db(abs(FD_sep_F_4IC{1,ii})./max(abs(FD_sep_F_4IC{1,ii})))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-15 0])
    %xlim([0 20])
    ylim([20 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('FDJADE (Hz) ','fontsize',6);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
figure,
set(gcf,'numbertitle','off','name', ' frequency FD-JADE 4IC');
for ii = 1:size(FD_sep_F_4IC,4)
    subplot(4,1,ii)
    %plot(F_LOFAR,abs(FD_sep_F_4IC{1,ii}(:,1)));
    plot(F_LOFAR_T,db(abs(FD_sep_F_4IC{1,ii}(:,1))./max(abs(FD_sep_F_4IC{1,ii}(:,1)))),'r-');
    xlabel('Frequency (Hz) FDJADE','fontsize',6);
    ylabel('Amplitude','fontsize',6);
    xlim([20 2000])
    ylim([-15 0])
    set(gca,'fontsize',12)
    set(gcf,'color','w')
    grid on
end
figure,
set(gcf,'numbertitle','off','name', ' frequency FD-JADE 5IC');
for ii = 1:size(FD_sep_F_5IC,5)
    subplot(5,1,ii)
    %plot(F_LOFAR,abs(FD_sep_F_4IC{1,ii}(:,1)));
    plot(F_LOFAR_T,db(abs(FD_sep_F_5IC{1,ii}(:,1))./max(abs(FD_sep_F_5IC{1,ii}(:,1)))),'r-');
    xlabel('Frequency (Hz) FDJADE','fontsize',6);
    ylabel('Amplitude','fontsize',6);
    xlim([20 2000])
    ylim([-5 0])
    set(gca,'fontsize',12)
    set(gcf,'color','w')
    grid on
end
figure,
set(gcf,'numbertitle','off','name', ' frequency FD-JADE 6IC');
for ii = 1:size(FD_sep_F_6IC,6)
    subplot(6,1,ii)
    %plot(F_LOFAR,abs(FD_sep_F_4IC{1,ii}(:,1)));
    plot(F_LOFAR_T,db(abs(FD_sep_F_6IC{1,ii}(:,1))./max(abs(FD_sep_F_6IC{1,ii}(:,1)))),'r-');
    xlabel('Frequency (Hz) FDJADE','fontsize',6);
    ylabel('Amplitude','fontsize',6);
    xlim([20 2000])
    ylim([-5 0])
    set(gca,'fontsize',12)
    set(gcf,'color','w')
    grid on
end
figure,
set(gcf,'numbertitle','off','name', ' frequency FD-JADE 7IC');
for ii = 1:size(FD_sep_F_7IC,7)
    subplot(7,1,ii)
    %plot(F_LOFAR,abs(FD_sep_F_4IC{1,ii}(:,1)));
    plot(F_LOFAR_T,((abs(FD_sep_F_7IC{1,ii}(:,1)))),'r-');
    xlabel('Frequency (Hz) FDJADE','fontsize',6);
    ylabel('Amplitude','fontsize',6);
    xlim([20 2000])
    ylim([0 1])
    set(gca,'fontsize',12)
    set(gcf,'color','w')
    grid on
end


