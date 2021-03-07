# Emotion-based-subtitling
This project can be used with multiple video conferencing platforms to provide color based subtitles.

**Methodology:**
1. Google speech to text api is used to convert audio into text
2. Audio transcripts are sent to 2 text processing engines:
   a. IBM watson API for tone analyisis
   b. Google NLP API for sentiment analysis
3. Raw audio is sent to speech sentiment processing API: Vokaturi API
4. Output of all the 3 APIs is combined to determine the final color to be assigned to each sentence in the Transcript in real-time.


**The algorithm for the system to determine color is associated with a US patent application and cannot be made public** 
