# AudioSync AI v2

## Deploy
1. Upload all files and the `static` folder to your new GitHub repository.
2. In Render choose New + > Web Service and connect this repository.
3. Runtime: Python
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable `GROQ_API_KEY` with your Groq API key.
7. Deploy. Open the service URL to use the app.

`render.yaml` contains the same deployment settings if Render detects Blueprint configuration.

## How it works
The app extracts and transcribes English audio from the video, transcribes the supplied Hindi narration, groups Whisper segments into sentences, pairs sentences by order, retimes corresponding video sections, removes the source audio, and adds the Hindi track.

## Important
- This is an initial beta. Whisper sentence boundaries are approximate.
- The number of detected English and Hindi sentences must match. If they do not, processing stops rather than pairing potentially incorrect sentences.
- Test with a short video first. Render free instances may sleep, run out of memory, or time out on long videos.
- Jobs and output files are held temporarily on the server. A service restart can clear them.
- Keep `GROQ_API_KEY` in Render environment variables; never put it in Blogger JavaScript.
- This first deployment serves its own upload interface. Blogger integration can be done after backend testing.
