import { Link } from 'react-router-dom';
import { Shield, Video, Activity, Lock, CheckCircle, Fingerprint, Mic, Keyboard, Mouse as MouseIcon, ArrowRight } from 'lucide-react';

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-blue-50">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <div className="flex justify-center mb-6">
            <div className="bg-primary-600 p-4 rounded-full">
              <Shield className="h-16 w-16 text-white" />
            </div>
          </div>
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Zero Trust Telehealth Platform
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Secure, verified telemedicine consultations with continuous biometric authentication.
            Experience healthcare with military-grade security.
          </p>
          <div className="flex gap-4 justify-center">
            <Link
              to="/login"
              className="px-8 py-4 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors flex items-center gap-2 shadow-lg"
            >
              Login
              <ArrowRight className="h-5 w-5" />
            </Link>
            <Link
              to="/register"
              className="px-8 py-4 bg-white text-primary-600 border-2 border-primary-600 rounded-lg font-semibold hover:bg-primary-50 transition-colors flex items-center gap-2 shadow-lg"
            >
              Register as Doctor
              <ArrowRight className="h-5 w-5" />
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          <div className="bg-white p-6 rounded-xl shadow-lg border-t-4 border-primary-600">
            <div className="bg-primary-100 p-3 rounded-lg w-fit mb-4">
              <Video className="h-8 w-8 text-primary-600" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">HD Video Consultations</h3>
            <p className="text-gray-600">
              Real-time, peer-to-peer video and audio consultations with crystal-clear quality.
            </p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg border-t-4 border-green-600">
            <div className="bg-green-100 p-3 rounded-lg w-fit mb-4">
              <Shield className="h-8 w-8 text-green-600" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">Zero Trust Security</h3>
            <p className="text-gray-600">
              Continuous verification ensures only authorized doctors can conduct consultations.
            </p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg border-t-4 border-blue-600">
            <div className="bg-blue-100 p-3 rounded-lg w-fit mb-4">
              <Activity className="h-8 w-8 text-blue-600" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">Real-Time Monitoring</h3>
            <p className="text-gray-600">
              Live biometric verification during consultations with instant trust scores.
            </p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg border-t-4 border-purple-600">
            <div className="bg-purple-100 p-3 rounded-lg w-fit mb-4">
              <Lock className="h-8 w-8 text-purple-600" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">HIPAA Compliant</h3>
            <p className="text-gray-600">
              End-to-end encryption and secure data handling for patient privacy.
            </p>
          </div>
        </div>

        {/* Biometric Authentication Section */}
        <div className="bg-white rounded-2xl shadow-xl p-12 mb-16">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Multi-Modal Biometric Authentication
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our platform uses four independent biometric verification methods to ensure
              the highest level of security and trust.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center p-6 bg-gradient-to-br from-red-50 to-red-100 rounded-xl">
              <div className="bg-red-600 p-4 rounded-full w-fit mx-auto mb-4">
                <Fingerprint className="h-10 w-10 text-white" />
              </div>
              <h4 className="text-lg font-bold text-gray-900 mb-2">Face Recognition</h4>
              <p className="text-sm text-gray-600">
                Advanced facial biometrics with liveness detection
              </p>
            </div>

            <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl">
              <div className="bg-blue-600 p-4 rounded-full w-fit mx-auto mb-4">
                <Mic className="h-10 w-10 text-white" />
              </div>
              <h4 className="text-lg font-bold text-gray-900 mb-2">Voice Analysis</h4>
              <p className="text-sm text-gray-600">
                Voiceprint verification using deep learning models
              </p>
            </div>

            <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-xl">
              <div className="bg-green-600 p-4 rounded-full w-fit mx-auto mb-4">
                <Keyboard className="h-10 w-10 text-white" />
              </div>
              <h4 className="text-lg font-bold text-gray-900 mb-2">Keystroke Dynamics</h4>
              <p className="text-sm text-gray-600">
                Unique typing patterns for behavioral authentication
              </p>
            </div>

            <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl">
              <div className="bg-purple-600 p-4 rounded-full w-fit mx-auto mb-4">
                <MouseIcon className="h-10 w-10 text-white" />
              </div>
              <h4 className="text-lg font-bold text-gray-900 mb-2">Mouse Movement</h4>
              <p className="text-sm text-gray-600">
                Movement pattern analysis for continuous verification
              </p>
            </div>
          </div>
        </div>

        {/* How It Works Section */}
        <div className="bg-gradient-to-r from-primary-600 to-blue-600 rounded-2xl shadow-xl p-12 text-white mb-16">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-lg text-primary-100 max-w-2xl mx-auto">
              Simple, secure, and seamless telemedicine experience
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-white bg-opacity-20 backdrop-blur-sm p-6 rounded-xl mb-4">
                <div className="bg-white p-4 rounded-full w-fit mx-auto mb-4">
                  <CheckCircle className="h-12 w-12 text-primary-600" />
                </div>
                <h3 className="text-2xl font-bold mb-2">1. Register & Enroll</h3>
                <p className="text-primary-100">
                  Doctors register and enroll their biometric data (face, voice, keystroke, mouse patterns)
                </p>
              </div>
            </div>

            <div className="text-center">
              <div className="bg-white bg-opacity-20 backdrop-blur-sm p-6 rounded-xl mb-4">
                <div className="bg-white p-4 rounded-full w-fit mx-auto mb-4">
                  <Video className="h-12 w-12 text-primary-600" />
                </div>
                <h3 className="text-2xl font-bold mb-2">2. Start Consultation</h3>
                <p className="text-primary-100">
                  Patients book appointments and join secure video consultations at scheduled times
                </p>
              </div>
            </div>

            <div className="text-center">
              <div className="bg-white bg-opacity-20 backdrop-blur-sm p-6 rounded-xl mb-4">
                <div className="bg-white p-4 rounded-full w-fit mx-auto mb-4">
                  <Shield className="h-12 w-12 text-primary-600" />
                </div>
                <h3 className="text-2xl font-bold mb-2">3. Continuous Verification</h3>
                <p className="text-primary-100">
                  Real-time biometric verification ensures the doctor's identity throughout the consultation
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center bg-white rounded-2xl shadow-xl p-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Ready to Experience Secure Telemedicine?
          </h2>
          <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
            Join our platform today and provide your patients with the most secure
            telehealth experience available.
          </p>
          <div className="flex gap-4 justify-center">
            <Link
              to="/register"
              className="px-8 py-4 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors flex items-center gap-2 shadow-lg text-lg"
            >
              Get Started Now
              <ArrowRight className="h-6 w-6" />
            </Link>
            <Link
              to="/login"
              className="px-8 py-4 bg-gray-100 text-gray-700 rounded-lg font-semibold hover:bg-gray-200 transition-colors text-lg"
            >
              Already have an account?
            </Link>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8 mt-16">
        <div className="container mx-auto px-4 text-center">
          <div className="flex justify-center mb-4">
            <Shield className="h-8 w-8 text-primary-400" />
          </div>
          <p className="text-gray-400">
            © 2026 Zero Trust Telehealth Platform. All rights reserved.
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Secured with multi-modal biometric authentication
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Home;

