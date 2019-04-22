clear all
close all
fRange = [50, 8000];
%x1=detrend(audioread('FDJADE1_sound.wav'));
%x2=detrend(audioread('FDjADE2_sound.wav'));
%load('Mixed_sig'); x1 = detrend(Mixed_sig(18,:));
%load('Mixed_sig'); x2 = detrend(Mixed_sig(24,:));
%load('Target_T1'); x1 = Target_sig;
%load('Target_T2'); x2 = Target_sig;
load('T1'); x1 = detrend(T1(5,:));
load('T2'); x2 = detrend(T2(5,:));
Fs=10e3;
time = (length(x1)-1)/Fs;


%%
win = 1024; % window (1.024sec)
overlap = win*0.5; % overlap
nfft = 1024;
dfs =4000;
sc = Fs/dfs; % Down sampling rate

x_ICA = [x1; x2];
%%
[x1_STFT,F_LOFAR_T,T_LOFAR_T] = stft(x1,win,overlap,nfft,dfs);
figure,
set(gcf,'numbertitle','off','name', 'received result 1');
surf(T_LOFAR_T,F_LOFAR_T,10*log10(abs(x1_STFT./max(abs(x1_STFT))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
%%
[x2_STFT,F_LOFAR_T,T_LOFAR_T] = stft(x2,win,overlap,nfft,dfs);
figure,
set(gcf,'numbertitle','off','name', 'received result 2');
surf(T_LOFAR_T,F_LOFAR_T,10*log10(abs(x2_STFT./max(abs(x2_STFT))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
    %
        x_STFT{1,1} = x1_STFT; x_STFT{1,2} = x2_STFT;
 


%% gamma tone
gf1 = gammatoneFast(x1,128,fRange);%Construct the cochleagram use Gammatone filterbank
gf2 = gammatoneFast(x2,128,fRange);%Construct the cochleagram use Gammatone filterbank

 
%%

cg1 = cochleagram(gf1);

cg2 = cochleagram(gf2);

 %% Using NNMF as tool to create zerocrossing
d = 1;

Tau=0:7;           % Defines tau shifts zerocrossing step 7
Phi=0:10;          % Defines phi shifts
maxiter = 50;     % Defines number of iterations
[W, H, cost] = is_nmf2D_mu(gf1,maxiter,d,Tau,Phi);% the MU algorithm
for idx = 1:d
    Rec(:,:,idx) = isp_nmf2d_rec(W, H, Tau,Phi, idx);
end
for k = 1:size(Rec,3)
    mask = logical(Rec(:,:,k)==max(Rec,[],3));%//  decision zerocrossing
    r1(k,:) = synthesisFast(x1,mask,fRange); % logarit to linear
    r1(k,:) = r(k,:)./max(abs(r1(k,:)));
%     wavwrite(r,Fs,sprintf([wavfile, '%d.wav'], k));
end

[r1_STFT,F_LOFAR_T,T_LOFAR_T] = stft(r1(1,:),win,overlap,nfft,dfs);
figure,
set(gcf,'numbertitle','off','name', 'output');
surf(T_LOFAR_T,F_LOFAR_T,10*log10(abs(r1_STFT./max(abs(r1_STFT))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-40 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
    
%%
d = 1;

Tau=0:7;           % Defines tau shifts zerocrossing step 7
Phi=0:10;          % Defines phi shifts
maxiter = 50;     % Defines number of iterations
[W, H, cost] = is_nmf2D_mu(gf1,maxiter,d,Tau,Phi);% the MU algorithm
for idx = 1:d
    Rec(:,:,idx) = isp_nmf2d_rec(W, H, Tau,Phi, idx);
end
for k = 1:size(Rec,3)
    mask = logical(Rec(:,:,k)==max(Rec,[],3));%//  decision zerocrossing
    r2(k,:) = synthesisFast(x1,mask,fRange); % logarit to linear
    r2(k,:) = r2(k,:)./max(abs(r2(k,:)));
%     wavwrite(r,Fs,sprintf([wavfile, '%d.wav'], k));
end
[r2_STFT,F_LOFAR_T,T_LOFAR_T] = stft(r2(1,:),win,overlap,nfft,dfs);
figure,
set(gcf,'numbertitle','off','name', 'output 2');
surf(T_LOFAR_T,F_LOFAR_T,10*log10(abs(r2_STFT./max(abs(r2_STFT))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
 %%
 %% JADE (Frequency domain)
disp('FD-JADE...')
FD_sep1_F = []; FD_sep2_F = []; FD_sep_F = cell(1,2);
for i = 1:size(x_STFT{1,1},2)
    ST = [x_STFT{1,1}(:,i)'; x_STFT{1,2}(:,i)'];
    [Ae, FD_jade] = jade(ST,2);
    FD_sep1_F = [FD_sep1_F FD_jade(1,:)'];
    FD_sep2_F = [FD_sep2_F FD_jade(2,:)'];
   % FD_sep3_F = [FD_sep3_F FD_jade(3,:)'];
end
clear i

FD_sep_F{1,1} = FD_sep1_F; FD_sep_F{1,2} = FD_sep2_F;  % STFT (Seperated)
[FD_sep1, T_ifft1] = istft(FD_sep1_F, win, overlap, nfft, dfs);
[FD_sep2, T_ifft2] = istft(FD_sep2_F, win, overlap, nfft, dfs);
Noise_Power_FDJADE1=4*bandpower(FD_sep1,4000,[1500 2000]);
Noise_Power_FDJADE2=4*bandpower(FD_sep2,4000,[1500 2000]);

%% Figure FD - JADE
figure,
set(gcf,'numbertitle','off','name', 'FDJADE');
for ii = 1:size(FD_sep_F,2)
    subplot(2,1,ii)
    surf(T_LOFAR_T,F_LOFAR_T,10*log10(abs(FD_sep_F{1,ii})./max(abs(FD_sep_F{1,ii})))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
figure,
set(gcf,'numbertitle','off','name', '   FDJADE');
for ii = 1:size(FD_sep_F,2)
    subplot(2,1,ii)
      %plot(F_LOFAR_T,(db(abs(FD_sep_F{1,ii}(:,4)./max(abs(FD_sep_F{1,ii}(:,4)))))),'r-');
      plot(F_LOFAR_T,abs(FD_sep_F{1,ii}(:,4)),'r-');
    %caxis([-20 0])
    xlim([0 2000])
    %ylim([-20 0])
    xlabel('Frequency (Hz) - FDJADE','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%% TD-JADE
disp('TD-JADE...')
TD_sep1_F = []; TD_sep2_F = []; TD_sep_F = cell(1,2);

    ST = [x_1;x_2];
    [Ae, TD_jade] = jade(ST,2);
    TD_sep1_F = [TD_sep1_F TD_jade(1,:)'];
    TD_sep2_F = [TD_sep2_F TD_jade(2,:)'];

   TD_sep_F{1,1} = TD_sep1_F; TD_sep_F{1,2} = TD_sep2_F;
   TD_STFT = cell(1,2); 

[TD_STFT1,F_LOFAR,T_LOFAR] = stft(TD_sep1_F,win,overlap,nfft,dfs); % STFT 1
[TD_STFT2,F_LOFAR,T_LOFAR] = stft(TD_sep2_F,win,overlap,nfft,dfs); % STFT 2

 TD_STFT{1,1} = TD_STFT1; TD_STFT{1,2} = TD_STFT2; 
 Noise_Power_TDJADE1=4*bandpower(TD_sep1_F,4000,[1500 2000]);
 Noise_Power_TDJADE2=4*bandpower(TD_sep2_F,4000,[1500 2000]);
 %(STFT)
%% TD-JADE figures
figure,
set(gcf,'numbertitle','off','name', 'TD JADE ');
for ii = 1:size(TD_STFT,2)
    subplot(2,1,ii)
surf(T_LOFAR,F_LOFAR,10*log10(abs(TD_STFT{1,ii}./max(abs(TD_STFT{1,ii}))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('TDJADE (hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end 

      figure,
set(gcf,'numbertitle','off','name', ' TD JADE');
for ii = 1:size(TD_STFT,2)
    subplot(2,1,ii)
      %plot(F_LOFAR_T,db((abs(TD_STFT{1,ii}(:,1)./max(abs(TD_STFT{1,ii}(:,1)))))),'r-')
       plot(F_LOFAR_T,abs(TD_STFT{1,ii}(:,1)),'r-');
    %caxis([-20 0])
    xlim([0 2000])
    %ylim([-20 0])
    xlabel('Frequency (Hz) - TDJADE','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%% ICA 
disp('ICA...')
[N,P]=size(x_1); 
%mixes=Ds_deci;
mixes=x_ICA; 
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


  ICA_STFT = cell(1,2); 
  ICA_STFT{1,1} = ICA_STFT1; ICA_STFT{1,2} = ICA_STFT2; ;%(STFT)
  Noise_Power_ICA1=4*bandpower(uu(1,:),4000,[1500 2000]);
  Noise_Power_ICA2=4*bandpower(uu(2,:),4000,[1500 2000]);
  
  %%  ICA  figure
figure,
set(gcf,'numbertitle','off','name', ' ICA ');
for ii = 1:size(ICA_STFT,2)
    subplot(2,1,ii)
surf(T_LOFAR,F_LOFAR,db(abs(ICA_STFT{1,ii})./max(abs(ICA_STFT{1,ii})))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('  ICA (Hz) ','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
     figure,
set(gcf,'numbertitle','off','name', '  ICA ');
for ii = 1:size(ICA_STFT,2)
    subplot(2,1,ii)
      %plot(F_LOFAR_T,db((abs(ICA_STFT{1,ii}(:,1))./max(abs(ICA_STFT{1,ii}(:,1))))),'r-');
      plot(F_LOFAR_T,abs(ICA_STFT{1,ii}(:,1)),'r-');
    %caxis([-20 0])
    xlim([0 2000])
    %ylim([-20 0])
    xlabel('Frequency (Hz) ICA','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end

%% fast ICA 2IC
disp('fast ICA...')
%[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_STFT{1,4},2,'kurtosis',1);
[Zica, W, T, mu, Zcw, normRows] = fastICA(x_ICA,2,'negentropy',0);
[f_ICA_STFT1,F_LOFAR,T_LOFAR] = stft(Zica(2,:),win,overlap,nfft,dfs); % STFT 1
[f_ICA_STFT2,F_LOFAR,T_LOFAR] = stft(Zica(1,:),win,overlap,nfft,dfs); % STFT 2


  f_ICA_STFT = cell(1,2); 
 f_ICA_STFT{1,1} = f_ICA_STFT1; f_ICA_STFT{1,2} = f_ICA_STFT2;%(STFT)
 Noise_Power_f_ICA1=4*bandpower(Zica(1,:),4000,[1500 2000]);
 Noise_Power_f_ICA2=4*bandpower(Zica(2,:),4000,[1500 2000]);


%% fast ICA  2ICfigure
figure,
set(gcf,'numbertitle','off','name', 'fast ICA  ');
for ii = 1:size(f_ICA_STFT,2)
    subplot(2,1,ii)
surf(T_LOFAR,F_LOFAR,10*log10(abs(f_ICA_STFT{1,ii}./max(abs(f_ICA_STFT{1,ii}))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)- fast ICA ','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
     figure,
set(gcf,'numbertitle','off','name', ' fast ICA 2IC');
for ii = 1:size(f_ICA_STFT,2)
    subplot(2,1,ii)
      %plot(F_LOFAR_T,db(abs(f_ICA_STFT{1,ii}(:,1))./max(abs(f_ICA_STFT{1,ii}(:,1)))),'r-');
       plot(F_LOFAR_T,abs(f_ICA_STFT{1,ii}(:,1)),'r-');
    %caxis([-20 0])
    xlim([0 2000])
    %ylim([-20 0])
    xlabel('Frequency (Hz) - fast ICA ','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
    %% Mixed signal after EIH
        %cd(Prgm_path)
% 좌표값 (표적 위치)
x = [-50 50];
y = [500 500];

% 좌표값 (센서 위치)
x_sen = [-5 5];
y_sen = [0 0];

p = 44.5; % Surface noise (0~4 stat: 44.5~66.5)
[R1, R2,Td] = Mixing(r(1,:),r(2,:), Fs, x, y, x_sen, y_sen, p);
R = [R1; R2];
R = R(:,Td:end);
[R1_STFT,F_LOFAR_T,T_LOFAR_T] = stft(R(1,:),win,overlap,nfft,dfs);
[R2_STFT,F_LOFAR_T,T_LOFAR_T] = stft(R(2,:),win,overlap,nfft,dfs);
 R_STFT = cell(1,2); 
 R_STFT{1,1} = R1_STFT; R_STFT{1,2} = R2_STFT;

%% JADE (Frequency domain)
disp('FD-JADE EIH..')
FD_sep1_EIH = []; FD_sep2_EIH = []; FD_sep_EIH = cell(1,2);
for i = 1:size(R_STFT{1,1},2)
    ST = [R_STFT{1,1}(:,i)'; R_STFT{1,2}(:,i)'];
    [Ae, FD_jade_EIH] = jade(ST,2);
    FD_sep1_EIH = [FD_sep1_EIH FD_jade_EIH(1,:)'];
    FD_sep2_EIH = [FD_sep2_EIH FD_jade_EIH(2,:)'];
   % FD_sep3_F = [FD_sep3_F FD_jade(3,:)'];
end
clear i

FD_sep_EIH{1,1} = FD_sep1_EIH; FD_sep_EIH{1,2} = FD_sep2_EIH;  % STFT (Seperated)

[FD_sep1_EIH, T_ifft1] = istft(FD_sep1_EIH, win, overlap, nfft, dfs);
[FD_sep2_EIH, T_ifft2] = istft(FD_sep2_EIH, win, overlap, nfft, dfs);
%Noise_Power_FDJADE1_EIH=4*bandpower(FD_sep1_EIH,2000,[1500 2000]);
%Noise_Power_FDJADE2_EIH=4*bandpower(FD_sep2_EIH,2000,[1500 2000]);
%% Figure FD - JADE
figure,
set(gcf,'numbertitle','off','name', 'FDJADE EIH');
for ii = 1:size(FD_sep_EIH,2)
    subplot(2,1,ii)
    surf(T_LOFAR_T,F_LOFAR_T,10*log10(abs(FD_sep_EIH{1,ii})./max(abs(FD_sep_EIH{1,ii})))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency(Hz) FDJADE EIH','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end

figure,
set(gcf,'numbertitle','off','name', '   FDJADE EIH ');
for ii = 1:size(FD_sep_EIH,2)
    subplot(2,1,ii)
      %plot(F_LOFAR_T,(db(abs(FD_sep_EIH{1,ii}(:,4)./max(abs(FD_sep_EIH{1,ii}(:,4)))))),'r-');
       plot(F_LOFAR_T,abs(FD_sep_EIH{1,ii}(:,4)),'r-');
    %caxis([-20 0])
    xlim([0 2000])
    %ylim([-30 0])
    xlabel('Frequency (Hz) - FDJADE EIH','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%%
disp('TD-JADE EIH...')
TD_sep1_EIH = []; TD_sep2_EIH = []; TD_sep_EIH = cell(1,2);

    ST = [R(1,:);R(2,:)];
    [Ae, TD_jade] = jade(ST,2);
    TD_sep1_EIH = [TD_sep1_EIH TD_jade(1,:)'];
    TD_sep2_EIH = [TD_sep2_EIH TD_jade(2,:)'];

   TD_sep_EIH{1,1} = TD_sep1_EIH; TD_sep_EIH{1,2} = TD_sep2_EIH;
   TD_STFT_EIH = cell(1,2); 

[TD_STFT1_EIH,F_LOFAR,T_LOFAR] = stft(TD_sep1_EIH,win,overlap,nfft,dfs); % STFT 1
[TD_STFT2_EIH,F_LOFAR,T_LOFAR] = stft(TD_sep2_EIH,win,overlap,nfft,dfs); % STFT 2

 TD_STFT_EIH{1,1} = TD_STFT1_EIH; TD_STFT_EIH{1,2} = TD_STFT2_EIH; 
  %Noise_Power_TDJADE_EIH1=4*bandpower(TD_sep1_EIH,2000,[1500 2000]);
  %Noise_Power_TDJADE_EIH2=4*bandpower(TD_sep2_EIH,2000,[1500 2000]);
 %(STFT)
%% TD-JADE EIH figures
figure,
set(gcf,'numbertitle','off','name', 'TDJADE EIH ');
for ii = 1:size(TD_STFT_EIH,2)
    subplot(2,1,ii)
surf(T_LOFAR,F_LOFAR,10*log10(abs(TD_STFT_EIH{1,ii}./max(abs(TD_STFT_EIH{1,ii}))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('TDJADE EIH (hz)','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end 

      figure,
set(gcf,'numbertitle','off','name', ' TDJADE EIH');
for ii = 1:size(TD_STFT_EIH,2)
    subplot(2,1,ii)
      %plot(F_LOFAR_T,db((abs(TD_STFT_EIH{1,ii}(:,1)./max(abs(TD_STFT_EIH{1,ii}(:,1)))))),'r-')
       plot(F_LOFAR,abs(TD_STFT_EIH{1,ii}(:,1)),'r-');
    %caxis([-20 0])
    xlim([0 2000])
    %ylim([-30 0])
    %xlabel('Frequency (Hz) - TDJADE EIH','fontsize',12); ylabel('Amplitude','fontsize',12);
    xlabel('Frequency (Hz)','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%% fast ICA EIH
disp('fast ICA EIH..')
%[Zica, W, T, mu, Zcw, normRows] = fastICA(Ds_STFT{1,4},2,'kurtosis',1);
[Zica, W, T, mu, Zcw, normRows] = fastICA(R,2,'negentropy',0);
[f_ICA_STFT1,F_LOFAR,T_LOFAR] = stft(Zica(2,:),win,overlap,nfft,dfs); % STFT 1
[f_ICA_STFT2,F_LOFAR,T_LOFAR] = stft(Zica(1,:),win,overlap,nfft,dfs); % STFT 2


  f_ICA_STFT = cell(1,2); 
 f_ICA_STFT{1,1} = f_ICA_STFT1; f_ICA_STFT{1,2} = f_ICA_STFT2;%(STFT)
 %Noise_Power_f_ICA_EIH1=4*bandpower(Zica(1,:),2000,[1500 2000]);
 %Noise_Power_f_ICA_EIH2=4*bandpower(Zica(2,:),2000,[1500 2000]);
%% fast ICA EIH figure
figure,
set(gcf,'numbertitle','off','name', 'fast ICA EIH  ');
for ii = 1:size(f_ICA_STFT,2)
    subplot(2,1,ii)
surf(T_LOFAR,F_LOFAR,10*log10(abs(f_ICA_STFT{1,ii}./max(abs(f_ICA_STFT{1,ii}))))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('Frequency (Hz)- fast ICA EIH','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
     figure,
set(gcf,'numbertitle','off','name', ' fast ICA EIH');
for ii = 1:size(f_ICA_STFT,2)
    subplot(2,1,ii)
      %plot(F_LOFAR_T,db(abs(f_ICA_STFT{1,ii}(:,1))./max(abs(f_ICA_STFT{1,ii}(:,1)))),'r-');
       plot(F_LOFAR_T,abs(f_ICA_STFT{1,ii}(:,1)),'r-');
    %caxis([-20 0])
    xlim([0 2000])
    %ylim([-20 0])
    xlabel('Frequency (Hz) - fast ICA EIH ','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
%% ICA EIH
disp('ICA EIH...')
[N,P]=size(R1); 
%mixes=Ds_deci;
mixes=R; 
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
  ICA_STFT = cell(1,2); 
  ICA_STFT{1,1} = ICA_STFT1; ICA_STFT{1,2} = ICA_STFT2; ;%(STFT)
  %Noise_Power_ICA_EIH1=4*bandpower(uu(1,:),2000,[1500 2000]);
  %Noise_Power_ICA_EIH2=4*bandpower(uu(2,:),2000,[1500 2000]);
  
  %%  ICA EIH figure
figure,
set(gcf,'numbertitle','off','name', ' ICA EIH ');
for ii = 1:size(ICA_STFT,2)
    subplot(2,1,ii)
surf(T_LOFAR,F_LOFAR,db(abs(ICA_STFT{1,ii})./max(abs(ICA_STFT{1,ii})))...
        ,'edgecolor','none'); axis tight;
    shading interp;
    view(0,90);
    colorbar;
    caxis([-20 0])
    xlim([1 5])
    ylim([0 2000])
    xlabel('Time (Sec)','fontsize',12); ylabel('  ICA EIH (Hz) ','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end
    
     figure,
set(gcf,'numbertitle','off','name', '  ICA EIH  ');
for ii = 1:size(ICA_STFT,2)
    subplot(2,1,ii)
     % plot(F_LOFAR_T,db((abs(ICA_STFT{1,ii}(:,1))./max(abs(ICA_STFT{1,ii}(:,1))))),'r-');
      plot(F_LOFAR_T,abs(ICA_STFT{1,ii}(:,1)),'r-');
    %caxis([-20 0])
    xlim([0 2000])
    %ylim([-20 0])
    xlabel('Frequency (Hz) ICA EIH','fontsize',12); ylabel('Amplitude','fontsize',12);
    set(gca,'fontsize',12)
    set(gcf,'color','w')
end