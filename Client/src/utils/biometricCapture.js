// Keystroke Dynamics Capture
export class KeystrokeCapture {
  constructor() {
    this.events = [];
    this.isRecording = false;
  }

  start() {
    this.events = [];
    this.isRecording = true;
  }

  stop() {
    this.isRecording = false;
    return this.getFeatures();
  }

  handleKeyDown(event) {
    if (!this.isRecording) return;
    
    this.events.push({
      key: event.key,
      type: 'keydown',
      timestamp: Date.now()
    });
  }

  handleKeyUp(event) {
    if (!this.isRecording) return;
    
    this.events.push({
      key: event.key,
      type: 'keyup',
      timestamp: Date.now()
    });
  }

  getFeatures() {
    const holdTimes = [];
    const ddTimes = [];
    const udTimes = [];

    // Calculate timing features
    for (let i = 0; i < this.events.length - 1; i++) {
      const current = this.events[i];
      const next = this.events[i + 1];

      if (current.type === 'keydown' && next.type === 'keyup' && current.key === next.key) {
        // Hold time (H) - convert to seconds
        holdTimes.push((next.timestamp - current.timestamp) / 1000);
      }

      if (current.type === 'keydown' && next.type === 'keydown') {
        // Down-Down time (DD) - convert to seconds
        ddTimes.push((next.timestamp - current.timestamp) / 1000);
      }

      if (current.type === 'keyup' && next.type === 'keydown') {
        // Up-Down time (UD) - convert to seconds
        udTimes.push((next.timestamp - current.timestamp) / 1000);
      }
    }

    // Combine timing features into a single vector
    const timingFeatures = [...holdTimes, ...ddTimes, ...udTimes];

    // Pad or truncate to fixed length (31 timing features)
    while (timingFeatures.length < 31) {
      timingFeatures.push(0);
    }
    const paddedTimingFeatures = timingFeatures.slice(0, 31);

    // Calculate statistical features (7 features: mean, std, median, min, max, q25, q75)
    const stats = this.calculateStatistics(paddedTimingFeatures);

    // Combine timing features (31) + statistical features (7) = 38 total
    return [...paddedTimingFeatures, ...stats];
  }

  calculateStatistics(features) {
    if (features.length === 0) {
      return [0, 0, 0, 0, 0, 0, 0];
    }

    const validFeatures = features.filter(f => f > 0);
    if (validFeatures.length === 0) {
      return [0, 0, 0, 0, 0, 0, 0];
    }

    const sorted = [...validFeatures].sort((a, b) => a - b);
    const sum = validFeatures.reduce((a, b) => a + b, 0);
    const mean = sum / validFeatures.length;

    // Standard deviation
    const variance = validFeatures.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / validFeatures.length;
    const std = Math.sqrt(variance);

    // Median
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

    // Min and Max
    const min = sorted[0];
    const max = sorted[sorted.length - 1];

    // Quartiles
    const q1Index = Math.floor(sorted.length * 0.25);
    const q3Index = Math.floor(sorted.length * 0.75);
    const q25 = sorted[q1Index];
    const q75 = sorted[q3Index];

    return [mean, std, median, min, max, q25, q75];
  }
}

// Mouse Movement Capture
export class MouseCapture {
  constructor() {
    this.events = [];
    this.isRecording = false;
  }

  start() {
    this.events = [];
    this.isRecording = true;
  }

  stop() {
    this.isRecording = false;
    return this.events;
  }

  handleMouseMove(event) {
    if (!this.isRecording) return;
    
    this.events.push({
      timestamp: Date.now() / 1000, // Convert to seconds
      x: event.clientX,
      y: event.clientY,
      button: 'NoButton',
      state: 'Move'
    });
  }

  handleMouseClick(event) {
    if (!this.isRecording) return;
    
    this.events.push({
      timestamp: Date.now() / 1000,
      x: event.clientX,
      y: event.clientY,
      button: event.button === 0 ? 'Left' : event.button === 2 ? 'Right' : 'Middle',
      state: 'Pressed'
    });
  }

  getEvents() {
    return this.events;
  }
}

// Voice Recording with WAV conversion
export class VoiceCapture {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;
    this.audioContext = null;
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      // Try to use audio/wav if supported, otherwise use default
      let mimeType = 'audio/webm';
      if (MediaRecorder.isTypeSupported('audio/wav')) {
        mimeType = 'audio/wav';
      } else if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
      }

      this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.start();
      return true;
    } catch (error) {
      console.error('Failed to start voice recording:', error);
      return false;
    }
  }

  async stop() {
    return new Promise((resolve) => {
      if (!this.mediaRecorder) {
        resolve(null);
        return;
      }

      this.mediaRecorder.onstop = async () => {
        // Create blob from recorded chunks
        const mimeType = this.mediaRecorder.mimeType;
        const audioBlob = new Blob(this.audioChunks, { type: mimeType });

        // Convert to WAV format
        const wavBlob = await this.convertToWav(audioBlob);

        // Stop all tracks
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
        }

        resolve(wavBlob);
      };

      this.mediaRecorder.stop();
    });
  }

  async convertToWav(blob) {
    try {
      // Create audio context
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });

      // Read blob as array buffer
      const arrayBuffer = await blob.arrayBuffer();

      // Decode audio data
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      // Convert to WAV
      const wavBuffer = this.audioBufferToWav(audioBuffer);
      const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });

      audioContext.close();
      return wavBlob;
    } catch (error) {
      console.error('Error converting to WAV:', error);
      // Return original blob if conversion fails
      return blob;
    }
  }

  audioBufferToWav(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    // Interleave channels
    const length = audioBuffer.length * numChannels * 2;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);

    // Write WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bitDepth / 8, true);
    view.setUint16(32, numChannels * bitDepth / 8, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);

    // Write audio data
    const channels = [];
    for (let i = 0; i < numChannels; i++) {
      channels.push(audioBuffer.getChannelData(i));
    }

    let offset = 44;
    for (let i = 0; i < audioBuffer.length; i++) {
      for (let channel = 0; channel < numChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, channels[channel][i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
    }

    return buffer;
  }
}

// Face Capture with webcam
export class FaceCapture {
  constructor() {
    this.stream = null;
    this.faceImages = [];
  }

  async startCamera(videoElement) {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });

      if (videoElement) {
        videoElement.srcObject = this.stream;
        // Explicitly play the video
        try {
          await videoElement.play();
        } catch (playError) {
          console.warn('Video play was prevented:', playError);
          // Some browsers require user interaction before playing
        }
      }

      return true;
    } catch (error) {
      console.error('Failed to access camera:', error);
      return false;
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }

  captureFrame(videoElement, width = 224, height = 224) {
    return new Promise((resolve, reject) => {
      try {
        if (!videoElement || !videoElement.srcObject) {
          reject(new Error('Video element not ready'));
          return;
        }

        // Create canvas to capture frame
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        // Draw current video frame
        ctx.drawImage(videoElement, 0, 0, width, height);

        // Convert to blob
        canvas.toBlob((blob) => {
          if (blob) {
            // Create File object
            const file = new File([blob], `face_${Date.now()}.jpg`, { type: 'image/jpeg' });
            this.faceImages.push(file);
            resolve(file);
          } else {
            reject(new Error('Failed to capture frame'));
          }
        }, 'image/jpeg', 0.95);
      } catch (error) {
        reject(error);
      }
    });
  }

  clearImages() {
    this.faceImages = [];
  }

  getImages() {
    return this.faceImages;
  }

  getImageCount() {
    return this.faceImages.length;
  }
}

