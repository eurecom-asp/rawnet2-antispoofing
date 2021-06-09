
function [stat,delta,double_delta]=extract_lfcc_high_res_baseline(speech,Fs,Window_Length,Window_Shift,NFFT,No_Filter,fmin,fmax)

% Input: file_path=Path of the speech file
%        Fs=Sampling frequency in Hz
%        Window_Length=Window length in ms (default: 30ms)
%        Window_Shift=hop length in ms (default: 15ms)
%        NFFT=No of FFT bins (default:1024)
%        No_Filter=No of filter (default:70)
%
%Output: stat=Static LFCC (Size: NxNo_Filter where N is the number of frames)
%        delta=Delta LFCC (Size: NxNo_Filter where N is the number of frames)
%        double_delta=Double Delta LFCC (Size: NxNo_Filter where N is the number of frames)




frame_length_inSample=round((Fs/1000)*Window_Length);
frame_shift_inSample=round((Fs/1000)*Window_Shift);
framedspeech=buffer(speech,frame_length_inSample,frame_shift_inSample,'nodelay')';

w=hamming(frame_length_inSample);
y_framed=framedspeech.*repmat(w',size(framedspeech,1),1);
%--------------------------------------------------------------------------

f=linspace(Fs/NFFT,Fs/2,NFFT/2+1); 
filbandwidthsf=linspace(fmin,fmax,No_Filter+2);
fr_all=(abs(fft(y_framed',NFFT))).^2;
fa_all=fr_all(1:(NFFT/2)+1,:)';


[~,indmin] = min(abs(f-fmin));
[~,indmax] = min(abs(f-fmax));

fa_all = fa_all(:,indmin:indmax);
no_coeff=20;
filterbank=zeros(size(fa_all,2),No_Filter);
for i=1:No_Filter
    filterbank(:,i)=trimf(f(indmin:indmax),[filbandwidthsf(i),filbandwidthsf(i+1),...
        filbandwidthsf(i+2)]);
end

filbanksum=fa_all*filterbank(1:end,:);

%-------------------------Calculate Static Cepstral------------------------
t=dct(log10(filbanksum'+eps));
t=(t(1:no_coeff,:));
stat=t'; 
delta=deltas(stat',3)';
double_delta=deltas(delta',3)';

%--------------------------------------------------------------------------
