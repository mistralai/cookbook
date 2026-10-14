// Runs on the AudioWorklet's own thread, separate from the page's main thread.
// Web Audio calls process() once per "render quantum" (128 samples) and hands
// us the microphone's raw audio; we just forward each quantum to the main
// thread, which buffers a bit before sending it over the network (see app.js).
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0][0];
    if (channel) {
      // The underlying buffer gets reused next quantum, so copy it with slice().
      this.port.postMessage(channel.slice());
    }
    return true; // keep the processor alive
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
