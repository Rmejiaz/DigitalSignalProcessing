# Funciones para el preprocesamiento para señales de voz y clase dummy


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scipy.io.wavfile import read
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import stft, welch
from python_speech_features import mfcc


class dummy_speech(BaseEstimator, TransformerMixin):
    def __init__(self, fs, mfcc_winlen,psd_winlen,stft_winlen):
        self.fs = fs
        self.mfcc_winlen=mfcc_winlen
        self.psd_winlen = psd_winlen
        self.stft_winlen = stft_winlen

    def fit(self,X, *_):
        Xi = X.copy()
        if (len(Xi.shape)==2):
          trials = Xi.shape[0]
        elif(len(Xi.shape)==1):
          trials = 1
        
        self.Xsampled = Xi
        # self.Xsampled = downsample(Xi,self.fs)
        self.fs = 48000

        self.psd_nperseg = self.psd_winlen*16
        self.stft_nperseg = self.stft_winlen*16
        # Fourier
        self.Xrfft = abs(np.fft.rfft(self.Xsampled,axis=-1)) # matriz 1 de atributos segun fft
        
        # STFT

        vf_stft,_,self.Xstft = stft(self.Xsampled,fs=self.fs,nperseg=self.stft_nperseg,axis=-1)
        self.Xstft = abs(self.Xstft)

        # PSD

        vf_psd, self.Xpsd = welch(self.Xsampled, fs=self.fs, nperseg=self.psd_nperseg,axis=-1)


        # Mfcc
        self.mfcc_winlens = self.mfcc_winlen/1000
        self.mfcc = mel_coefficients(self.Xsampled,fs=self.fs,winlen=self.mfcc_winlens)
      

        Xdata = np.c_[(cal_momentos(self.Xrfft).reshape(trials,-1),
                      cal_momentos(self.Xstft.reshape(trials,-1)).reshape(trials,-1),
                      cal_momentos(self.Xpsd).reshape(trials,-1),
                      self.mfcc.reshape(trials,-1))]

        
        return self

    
    def transform(self, X, *_):
        Xi = X.copy()
        if (len(Xi.shape)==2):
          trials = Xi.shape[0]
        elif(len(Xi.shape)==1):
          trials = 1
          
        self.Xsampled = Xi
        # self.Xsampled = downsample(Xi,self.fs)
        self.fs = 48000

        self.psd_nperseg = self.psd_winlen*16
        self.stft_nperseg = self.stft_winlen*16
        # Fourier
        self.Xrfft = abs(np.fft.rfft(self.Xsampled,axis=-1)) # matriz 1 de atributos segun fft
        
        # STFT

        vf_stft,_,self.Xstft = stft(self.Xsampled,fs=self.fs,nperseg=self.stft_nperseg,axis=-1)
        self.Xstft = abs(self.Xstft)

        # PSD

        vf_psd, self.Xpsd = welch(self.Xsampled, fs=self.fs, nperseg=self.psd_nperseg,axis=-1)


        # Mfcc
        self.mfcc_winlens = self.mfcc_winlen/1000
        self.mfcc = mel_coefficients(self.Xsampled,fs=self.fs,winlen=self.mfcc_winlens)
      

        Xdata = np.c_[(cal_momentos(self.Xrfft).reshape(trials,-1),
                      cal_momentos(self.Xstft.reshape(trials,-1)).reshape(trials,-1),
                      cal_momentos(self.Xpsd).reshape(trials,-1),
                      self.mfcc.reshape(trials,-1))]
        
        return Xdata

    def fit_transform(self,X,*_):
        self.fit(X)
        return self.transform(X)


def cal_momentos(Xf): #se calcula momentos sobre ultimo eje
  #media, mediana, var, max, min
  m =np.c_[(Xf.mean(axis=-1),np.median(Xf,axis=-1),Xf.var(axis=-1),Xf.max(axis=-1),Xf.min(axis=-1))]
  return m

def downsample(X,fs):
    Xi = X.copy()
    if (len(Xi.shape)==2):
        if (fs == 48000):
            Xi = Xi[:,::3]
    elif(len(Xi.shape)==1):
        if (fs == 48000):
            Xi = Xi[::3]
    return Xi


def mel_coefficients(X,fs,winlen):
  mel = []
  if (len(X.shape)==1):
    mel = mfcc(X,samplerate=fs,nfft=(int(2*fs*winlen)), winlen=winlen)
  elif(len(X.shape)==2):
    for i in range(X.shape[0]):
      mel.append(mfcc(X[i],samplerate=fs,nfft=(int(2*fs*winlen)),winlen=winlen)) 
    mel = np.array(mel)
  return mel




def load_database(labels,sujetos):
  x = []
  y = []
  for sujeto in sujetos:
    for palabra in range(len(labels)):
      i = 1
      while(True):
        path = f"Database/{sujeto}/{labels[palabra]}/{i}.wav"
        try:
          a = np.array(read(path)[1])
        except:
          break
        x.append(a)
        i+=1
        y.append(palabra)
  fs = read(f"Database/Rafael/Arriba/1.wav")[0]
  x = np.array(x)
  y = np.array(y)

  return x,y,fs



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



# from IPython.display import HTML, Audio
# from google.colab.output import eval_js
# from base64 import b64decode
# import numpy as np
# from scipy.io.wavfile import read as wav_read
# import io
# import ffmpeg
# import time
 
# AUDIO_HTML = """
# <script>
# var my_div = document.createElement("DIV");
# var my_p = document.createElement("P");
# var my_btn = document.createElement("BUTTON");
# var t = document.createTextNode("Press to start recording");
 
# my_btn.appendChild(t);
# //my_p.appendChild(my_btn);
# my_div.appendChild(my_btn);
# document.body.appendChild(my_div);
 
# var base64data = 0;
# var reader;
# var recorder, gumStream;
# var recordButton = my_btn;
 
# var handleSuccess = function(stream) {
#   gumStream = stream;
#   var options = {
#     //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
#     mimeType : 'audio/webm;codecs=opus'
#     //mimeType : 'audio/webm;codecs=pcm'
#   };            
#   //recorder = new MediaRecorder(stream, options);
#   recorder = new MediaRecorder(stream);
#   recorder.ondataavailable = function(e) {            
#     var url = URL.createObjectURL(e.data);
#     var preview = document.createElement('audio');
#     preview.controls = true;
#     preview.src = url;
#     document.body.appendChild(preview);
 
#     reader = new FileReader();
#     reader.readAsDataURL(e.data); 
#     reader.onloadend = function() {
#     base64data = reader.result;
#     }
#   };
#   recorder.start();
#   };
 
# navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);
# recordButton.innerText = "Recording...";
 
# function toggleRecording() {
#   if (recorder && recorder.state == "recording") {
#       recorder.stop();
#       gumStream.getAudioTracks()[0].stop();
#       recordButton.innerText = "Saving the recording... pls wait!"
#       var data = new Promise(resolve => {
#       resolve(base64data.toString());
#   });
#   }
# }
 
# // https://stackoverflow.com/a/951057
# function sleep(ms) {
#   return new Promise(resolve => setTimeout(resolve, ms));
# }
 
# sleep(2000).then(() => {
#   // wait 2000ms for the data to be available...
#   toggleRecording();
# });
 
# var data = new Promise(resolve => {
#   sleep(2500).then(() => {
#   resolve(base64data.toString());
#   });
# });   
# </script>
# """
 
# def get_audio():
#   display(HTML(AUDIO_HTML))
#   data = eval_js("data")
#   binary = b64decode(data.split(',')[1])
  
#   process = (ffmpeg
#     .input('pipe:0')
#     .output('pipe:1', format='wav')
#     .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
#   )
#   output, err = process.communicate(input=binary)
  
#   riff_chunk_size = len(output) - 8
#   # Break up the chunk size into four bytes, held in b.
#   q = riff_chunk_size
#   b = []
#   for i in range(4):
#       q, r = divmod(q, 256)
#       b.append(r)
 
#   # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
#   riff = output[:4] + bytes(b) + output[8:]
 
#   sr, audio = wav_read(io.BytesIO(riff))
#   return audio, sr





