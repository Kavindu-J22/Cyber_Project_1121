import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';

class MLService {
  constructor() {
    this.voiceApiUrl = process.env.VOICE_API_URL || 'http://localhost:8001';
    this.keystrokeApiUrl = process.env.KEYSTROKE_API_URL || 'http://localhost:8002';
    this.mouseApiUrl = process.env.MOUSE_API_URL || 'http://localhost:8003';
    this.faceApiUrl = process.env.FACE_API_URL || 'http://localhost:8004';
  }

  // Voice Recognition Services
  async enrollVoice(userId, audioFilePath) {
    try {
      const formData = new FormData();
      formData.append('audio_file', fs.createReadStream(audioFilePath));
      formData.append('speaker_id', userId);

      const response = await axios.post(
        `${this.voiceApiUrl}/api/v1/enroll/upload`,
        formData,
        {
          headers: formData.getHeaders(),
          timeout: 30000
        }
      );

      return response.data;
    } catch (error) {
      console.error('Voice enrollment error:', error.message);
      throw new Error(`Voice enrollment failed: ${error.message}`);
    }
  }

  async enrollVoiceMultiple(userId, audioFilePaths) {
    try {
      // Use the /api/v1/enroll endpoint which expects multiple audio files
      const response = await axios.post(
        `${this.voiceApiUrl}/api/v1/enroll`,
        {
          speaker_id: userId,
          audio_files: audioFilePaths
        },
        {
          headers: { 'Content-Type': 'application/json' },
          timeout: 60000 // Longer timeout for multiple files
        }
      );

      return response.data;
    } catch (error) {
      console.error('Voice enrollment (multiple) error:', error.message);
      throw new Error(`Voice enrollment failed: ${error.message}`);
    }
  }

  async verifyVoice(userId, audioFilePath) {
    try {
      const formData = new FormData();
      formData.append('audio_file', fs.createReadStream(audioFilePath));
      formData.append('speaker_id', userId);

      const response = await axios.post(
        `${this.voiceApiUrl}/api/v1/verify`,
        formData,
        {
          headers: formData.getHeaders(),
          timeout: 30000
        }
      );

      return response.data;
    } catch (error) {
      console.error('Voice verification error:', error.message);
      throw new Error(`Voice verification failed: ${error.message}`);
    }
  }

  // Keystroke Dynamics Services
  async enrollKeystroke(userId, keystrokeSamples) {
    try {
      const response = await axios.post(
        `${this.keystrokeApiUrl}/enroll`,
        {
          user_id: userId,
          keystroke_samples: keystrokeSamples
        },
        { timeout: 30000 }
      );

      return response.data;
    } catch (error) {
      console.error('Keystroke enrollment error:', error.message);
      throw new Error(`Keystroke enrollment failed: ${error.message}`);
    }
  }

  async verifyKeystroke(userId, keystrokeSample) {
    try {
      const response = await axios.post(
        `${this.keystrokeApiUrl}/verify`,
        {
          user_id: userId,
          keystroke_sample: keystrokeSample
        },
        { timeout: 10000 }
      );

      return response.data;
    } catch (error) {
      console.error('Keystroke verification error:', error.message);
      throw new Error(`Keystroke verification failed: ${error.message}`);
    }
  }

  // Mouse Movement Services
  async enrollMouse(userId, mouseEvents) {
    try {
      const response = await axios.post(
        `${this.mouseApiUrl}/enroll`,
        {
          user_id: userId,
          events: mouseEvents
        },
        { timeout: 30000 }
      );

      return response.data;
    } catch (error) {
      console.error('Mouse enrollment error:', error.message);
      throw new Error(`Mouse enrollment failed: ${error.message}`);
    }
  }

  async verifyMouse(userId, mouseEvents) {
    try {
      const response = await axios.post(
        `${this.mouseApiUrl}/verify`,
        {
          user_id: userId,
          events: mouseEvents
        },
        { timeout: 10000 }
      );

      return response.data;
    } catch (error) {
      console.error('Mouse verification error:', error.message);
      throw new Error(`Mouse verification failed: ${error.message}`);
    }
  }

  // Face Verification Services
  async enrollFace(userId, faceImagePaths) {
    try {
      const formData = new FormData();
      formData.append('user_id', userId);
      
      // Add all face images
      for (const imagePath of faceImagePaths) {
        formData.append('face_samples', fs.createReadStream(imagePath));
      }

      const response = await axios.post(
        `${this.faceApiUrl}/api/v1/enroll`,
        formData,
        {
          headers: formData.getHeaders(),
          timeout: 30000
        }
      );

      return response.data;
    } catch (error) {
      console.error('Face enrollment error:', error.message);
      throw new Error(`Face enrollment failed: ${error.message}`);
    }
  }

  async verifyFace(userId, faceImagePath) {
    try {
      const formData = new FormData();
      formData.append('user_id', userId);
      formData.append('face_sample', fs.createReadStream(faceImagePath));

      const response = await axios.post(
        `${this.faceApiUrl}/api/v1/verify`,
        formData,
        {
          headers: formData.getHeaders(),
          timeout: 10000
        }
      );

      return response.data;
    } catch (error) {
      console.error('Face verification error:', error.message);
      throw new Error(`Face verification failed: ${error.message}`);
    }
  }

  // Health check for all ML services
  async checkHealth() {
    const results = {
      voice: false,
      keystroke: false,
      mouse: false,
      face: false
    };

    try {
      await axios.get(`${this.voiceApiUrl}/health`, { timeout: 5000 });
      results.voice = true;
    } catch (error) {
      console.error('Voice API health check failed');
    }

    try {
      await axios.get(`${this.keystrokeApiUrl}/health`, { timeout: 5000 });
      results.keystroke = true;
    } catch (error) {
      console.error('Keystroke API health check failed');
    }

    try {
      await axios.get(`${this.mouseApiUrl}/health`, { timeout: 5000 });
      results.mouse = true;
    } catch (error) {
      console.error('Mouse API health check failed');
    }

    try {
      await axios.get(`${this.faceApiUrl}/health`, { timeout: 5000 });
      results.face = true;
    } catch (error) {
      console.error('Face API health check failed');
    }

    return results;
  }
}

export default new MLService();

