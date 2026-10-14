// Captures the microphone via an AudioWorklet, streams it to the server over
// a WebSocket, and renders the live transcript and fact-check results as they
// come back.

const CHUNK_MS = 200; // how much audio we buffer client-side before sending

const startButton = document.getElementById("start-button");
const transcriptEl = document.getElementById("transcript");
const resultsEl = document.getElementById("results");

let audioContext = null;
let ws = null; // the current session, or null when stopped

startButton.addEventListener("click", toggleRecording);

async function toggleRecording() {
  if (ws) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  // Set up microphone capture and stream audio chunks to the server.
  startButton.disabled = true;

  // The AudioContext (and its worklet module) only needs setting up once;
  // it's reused across every later start/stop cycle.
  if (!audioContext) {
    audioContext = new AudioContext();
    await audioContext.audioWorklet.addModule("audio-processor.js");
  }
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioContext.createMediaStreamSource(stream);
  const captureNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
  source.connect(captureNode);

  // Buffer incoming audio quanta until we have CHUNK_MS worth, then send as
  // one binary frame - sending every ~128-sample quantum individually would
  // be hundreds of tiny websocket messages per second.
  const samplesPerChunk = Math.floor((audioContext.sampleRate * CHUNK_MS) / 1000);
  let buffered = [];
  let bufferedLength = 0;

  const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  ws = socket;

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ sample_rate: audioContext.sampleRate }));
    startButton.textContent = "Stop";
    startButton.disabled = false;
  });

  socket.addEventListener("message", (event) => {
    renderMessage(JSON.parse(event.data));
  });

  socket.addEventListener("close", () => {
    stream.getTracks().forEach((t) => t.stop());
    captureNode.disconnect();
    source.disconnect();
    if (ws === socket) ws = null;
    startButton.textContent = "Start";
    startButton.disabled = false;
  });

  // audio-processor.js posts a message here every audio "quantum" (a fixed
  // batch of ~128 samples, delivered many times per second) with the
  // microphone's raw audio for that instant. Each piece gets appended
  // to buffered below. Once there's CHUNK_MS worth, they're stitched into
  // one array, converted to the PCM format the server expects, and sent -
  // then the buffer resets to start collecting the next chunk.
  captureNode.port.onmessage = (event) => {
    buffered.push(event.data);
    bufferedLength += event.data.length;
    if (bufferedLength >= samplesPerChunk) {
      const chunk = mergeFloat32(buffered, bufferedLength);
      buffered = [];
      bufferedLength = 0;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(floatTo16BitPCM(chunk).buffer);
      }
    }
  };
}

function stopRecording() {
  if (!ws) return;
  ws.close(); // triggers the "close" listener above, which does the actual cleanup
}

function renderMessage(message) {
  // Update the transcript or add a fact-check result card to the page.
  if (message.type === "transcript") {
    transcriptEl.textContent = message.text;
    return;
  }

  const card = document.createElement("div");
  card.className = "card";
  if (!message.matched) {
    card.innerHTML = `
      <div class="claim">${escapeHtml(message.text)}</div>
      <div class="badge">No match</div>
    `;
  } else {
    card.innerHTML = `
      <div class="claim">${escapeHtml(message.text)}</div>
      <div class="badge badge-${message.verdict}">${message.verdict.replace("_", " ")}</div>
      <div class="reasoning">${escapeHtml(message.reasoning)}</div>
    `;
  }
  resultsEl.prepend(card);
}

// Web Audio delivers audio in small fixed-size pieces as the microphone
// captures it, not as one continuous stream. This stitches a batch of
// those pieces back together into a single contiguous array right before
// we send it over the network.
function mergeFloat32(chunks, totalLength) {
  // Combine buffered Float32 audio chunks into one contiguous array.
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

// Web Audio represents each audio sample internally as a 32-bit float
// between -1 and 1, but Mistral's transcription API expects raw 16-bit
// integer PCM samples (the standard format most audio APIs use) - this
// rescales and converts each sample into that format.
function floatTo16BitPCM(float32) {
  // Convert normalized Float32 samples into 16-bit PCM samples for transport.
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
