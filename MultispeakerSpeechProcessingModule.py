from __future__ import division

import re
import sys
from google.cloud import speech_v1p1beta1 as speech
import datetime
import time
import wave
# from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from collections import defaultdict

from google.cloud import language
from google.cloud.language import enums as e2
from google.cloud.language import types as t2
import pyaudio
import requests
from six.moves import queue
import six
import collections
import re
import sys
import socket
import os
import scipy.io.wavfile
import numpy as np
sys.path.append("C:/Users/gokhalea/HackathonCodes/OpenVokaturi-3-0a/OpenVokaturi-3-0a/api")
import Vokaturi

FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK1=1024
iterator = 0
frames = []

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\gokhalea\\HackathonCodes\\key.json"
# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
color_matrix = []
#host = '127.0.0.1'
host = '127.0.0.1'
#host = '172.20.10.3'
port = 2007
BUFFER_SIZE = 2000
tcpClientA = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpClientA.connect((host, port))

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    

    def generator(self):
        global frames
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            frames.append(b''.join(data))
            yield b''.join(data)

def matrix_creation():

    color_matrix = defaultdict(dict)
    x = "joy"
    # Colors for JOY
    color_matrix["joy"]     = {}
    color_matrix[x][1]      = "A++"
    color_matrix[x][0.5]    = "A+"
    color_matrix[x][0]      = "A"
    color_matrix[x][-0.5]   = "B"
    color_matrix[x][-1]     = "B-"

    #Colors for confident
    x = "confident"
    color_matrix["confident"]   = {}
    color_matrix[x][1]          = "A++"
    color_matrix[x][0.5]        = "A+"
    color_matrix[x][0]          = "A"
    color_matrix[x][-0.5]       = "B"
    color_matrix[x][-1]         = "B-"

    #Colors for analytical
    x = "analytical"
    color_matrix["analytical"]   = {}
    color_matrix[x][1]          = "A+"
    color_matrix[x][0.5]        = "A-"
    color_matrix[x][0]          = "A"
    color_matrix[x][-0.5]       = "B"
    color_matrix[x][-1]         = "B-"

    #Colors for tentative
    x = "tentative"
    color_matrix["tentative"]   = {}
    color_matrix[x][1]          = "A+"
    color_matrix[x][0.5]        = "A-"
    color_matrix[x][0]          = "A"
    color_matrix[x][-0.5]       = "B"
    color_matrix[x][-1]         = "B-"


    #Colors for sadness
    x = "sadness"
    color_matrix["sadness"]   = {}
    color_matrix[x][1]          = "A-"
    color_matrix[x][0.5]        = "A"
    color_matrix[x][0]          = "B+"
    color_matrix[x][-0.5]       = "B-"
    color_matrix[x][-1]         = "B--"


    #Colors for fear
    x = "fear"
    color_matrix["fear"]   = {}
    color_matrix[x][1]          = "A-"
    color_matrix[x][0.5]        = "A"
    color_matrix[x][0]          = "B+"
    color_matrix[x][-0.5]       = "B-"
    color_matrix[x][-1]         = "B--"


    #Colors for anger
    x = "anger"
    color_matrix["anger"]   = {}
    color_matrix[x][1]          = "A-"
    color_matrix[x][0.5]        = "A"
    color_matrix[x][0]          = "B+"
    color_matrix[x][-0.5]       = "B-"
    color_matrix[x][-1]         = "B--"

    return color_matrix

def color_selector(tone, sentiment):
    global color_matrix

    if(sentiment >= 0.5):
        sentiment = 1
    elif (sentiment > 0):
        sentiment = 0.5
    elif (sentiment < -0.5):
        sentiment = -1
    elif (sentiment < 0):
        sentiment = -0.5
    if(tone == ""):
        tone = 'tentative'

    # print (color_matrix)
    # print (sentiment)
    return color_matrix[str(tone)][(sentiment)]

def sample_analyze_sentiment(content):
        print("In google sentiment code")

        client = language.LanguageServiceClient()

        # content = 'Your text to analyze, e.g. Hello, world!'

        if isinstance(content, six.binary_type):
            content = content.decode('utf-8')

        type_ = e2.Document.Type.PLAIN_TEXT
        document = {'type': type_, 'content': content}
        print("Before call to analyze sentiment")
        response = client.analyze_sentiment(document)
        print("After call to analyze sentiment")
        sentiment = response.document_sentiment
        print("Leaving google sentiment")
        # print('Score: {}'.format(sentiment.score))
        # print('Magnitude: {}'.format(sentiment.magnitude)) 
        # print ("this is the sentiment score ", sentiment.score)
        return (sentiment.score)



def emotion_analysis(text_for_tone):
    print("Entering IBM code")
    params = (
        ('version', '2017-09-21\n'),
        ('text', text_for_tone)
    )

    # ----- TONE ANALYSIS
    # print("does this work?")
    response = requests.get('https://gateway-lon.watsonplatform.net/tone-analyzer/api/v3/tone', verify=False, params=params, auth=('apikey', 'xDQQ4SE2vF414qGVGGIiFhLhWApRpAUtuq5iLzz_v7pK'))
    var = response.json()
    tone = ""
    #print(var)
    for x in var['document_tone']['tones']:
        tone = x['tone_id']
        break
        

    # print ( var['document_tone']['tones'])
    # print response.json()

    #------ SENTIMENT ANALYSIS
    print("Before call to google seentiment")
    sentiment_score = sample_analyze_sentiment(text_for_tone)
    #sentiment_score=0
    print("Leaving IBM code")
    return tone, sentiment_score


def listen_print_loop(responses):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    global frames
    global iterator
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript
        result = response.results[-1]
        words_info = result.alternatives[0].words

        # print(" sentence: '{}', speaker_tag: {}".format(words_info, words_info.speaker_tag))
        # for word_info in words_info:
        #     if(word_info.speaker_tag == 1):
        #         print("word: '{}', speaker_tag: {}".format(word_info.word, word_info.speaker_tag))

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)
            #print(transcript + overwrite_chars)

        else:
            # print("This is OPPPPOO the final line?")
            print(transcript + overwrite_chars)
            final_stt = transcript + overwrite_chars
            #byte_str=final_stt.encode('utf-8')

            file_name = "file" + str(iterator) + ".wav"
            iterator = iterator + 1
            waveFile = wave.open(file_name, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(pyaudio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
            frames = []
            
            voka_result_probabilities = vokatori_fun(file_name)
            

            print("Before call to IBM")
            tone, sentiment = emotion_analysis(final_stt)


            grade = 'A'
            print ( "listing for loop ", tone, sentiment)
            print (color_selector(tone, sentiment))
            if voka_result_probabilities is not None:
                x = color_selector(tone, sentiment)
                if (x=="A++"):
                    if(voka_result_probabilities.happiness > 0.5):
                            grade = 'C'
                    elif(voka_result_probabilities.happiness <= 0.5 and voka_result_probabilities.happiness > 0.2):
                            grade = 'B'
                    else:
                            grade = 'A'

                if (x == "A+"):
                    if (voka_result_probabilities.happiness > 0.5):
                        grade = 'C'
                    elif (voka_result_probabilities.happiness <= 0.5 and voka_result_probabilities.happiness > 0.2):
                        grade = 'B'
                    else:
                        grade = 'A'

                if (x == "B-"):
                    if (voka_result_probabilities.sadness > 0.5 or voka_result_probabilities.anger > 0.5 ):
                        grade = 'C'
                    elif ((voka_result_probabilities.sadness <= 0.5 and voka_result_probabilities.sadness > 0.2) or (voka_result_probabilities.sadness <= 0.5 and voka_result_probabilities.sadness > 0.2)):
                        grade = 'B'
                    elif (voka_result_probabilities.sadness <= 0.2 or voka_result_probabilities.anger <= 0.2 ):
                        grade = 'A'

                if (x == "B--"):
                    if (voka_result_probabilities.sadness > 0.5 or voka_result_probabilities.anger > 0.5):
                        grade = 'C'
                    elif ((voka_result_probabilities.sadness <= 0.5 and voka_result_probabilities.sadness > 0.2) or (
                            voka_result_probabilities.sadness <= 0.5 and voka_result_probabilities.sadness > 0.2)):
                        grade = 'B'
                    elif (voka_result_probabilities.sadness <= 0.2 or voka_result_probabilities.anger <= 0.2):
                        grade = 'A'

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break
            # time_date    
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            #print {st, final_stt, color_selector(tone, sentiment)}
            #byte_str={st, final_stt, color_selector(tone, sentiment)}.encode('utf-8')
            final = "Asmita"+ ";;" + st + ";;" + final_stt + ";;" + color_selector(tone, sentiment) + ";;" + grade
            print (final)
            byte_str=final.encode('utf-8')	
            tcpClientA.sendall(byte_str)

            num_chars_printed = 0
        # print ("Is this the final line of sentence")

def vokatori_fun(file_name):
    #print("Loading library...")
    Vokaturi.load("C:/Users/gokhalea/HackathonCodes/OpenVokaturi-3-0a/OpenVokaturi-3-0a/lib/open/win/OpenVokaturi-3-0-win64.dll")
    #print("Analyzed by: %s" % Vokaturi.versionAndLicense())

    #print("Reading sound file...")
    #file_name = sys.argv[1]
    #(sample_rate, samples) = scipy.io.wavfile.read(file_name)
    (sample_rate, samples) = scipy.io.wavfile.read(file_name)
    #print("   sample rate %.3f Hz" % sample_rate)
    #print("Samples:" % samples)
    #print("Allocating Vokaturi sample array...")
    buffer_length = len(samples)
    #print("   %d samples, %d channels" % (buffer_length, samples.ndim))
    c_buffer = Vokaturi.SampleArrayC(buffer_length)
    if samples.ndim == 1:  # mono
        c_buffer[:] = samples[:] / 32768.0
    else:  # stereo
        c_buffer[:] = 0.5 * (samples[:, 0] + 0.0 + samples[:, 1]) / 32768.0

    #print("Creating VokaturiVoice...")
    voice = Vokaturi.Voice(sample_rate, buffer_length)

    #print("Filling VokaturiVoice with samples...")
    voice.fill(buffer_length, c_buffer)

    print("Extracting emotions from VokaturiVoice...")
    quality = Vokaturi.Quality()
    emotionProbabilities = Vokaturi.EmotionProbabilities()
    voice.extract(quality, emotionProbabilities)



    if quality.valid:

        print("Neutral: %.3f" % emotionProbabilities.neutrality)
        print("Happy: %.3f" % emotionProbabilities.happiness)
        print("Sad: %.3f" % emotionProbabilities.sadness)
        print("Angry: %.3f" % emotionProbabilities.anger)
        print("Fear: %.3f" % emotionProbabilities.fear)
        print("______________________________________________________________________________________________")
        return emotionProbabilities
    else:
        print("Not enough sonorancy to determine emotions")

    voice.destroy()

    return None


def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'en'  # a BCP-47 language tag
    name = "Nikita"
    keyipo = "IPO"    
    listipo = ["IPO" , "IP Office" , "COM" , "Cloud Operation Manager", "Cloud Media Manager", "CMM" , "Web Manager", "Certificate Agent", "CAS", "CDA", "Auto Attendant", "Self Admin", "Jira", "BitBucket"]
    keydes = "DES"    
    listdes = ['Jira', 'confluence', 'DES', 'the Jira Id', 'code review', 'this code', 'the code', 'Purva', 'Poorva', 'Avaya', 'the DES team', 'DES team', 'DES people', 'Jira Id', ]
    teamdict = {'IPO' : listipo, 'DES' : listdes}
    list = teamdict[keyipo]
    #list = teamdict[teamname]	
    byname = "by " + name
    withname = "with " + name 
    fromname = "from " + name
    list.extend([byname, withname, fromname])    
    #list.extend("by Nikita" , "with Nikita" , "from Nikita")	
    global frames
    global color_matrix 
    color_matrix = matrix_creation()
    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        speech_contexts = [{'phrases':list}]
 )
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    p = pyaudio.PyAudio()
    exception = 0

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)


        # try:
        responses = client.streaming_recognize(streaming_config, requests)
        # except:
            # exception = 1
        # Now, put the transcription responses to use.
        #try:    
        listen_print_loop(responses)
        #except:
        #    print ("Excption handle : Exceeded maximum allowed stream duration of 65 seconds")
        

            # print ("Lets hope this worked")

if __name__ == '__main__':
    while True:
        main()
