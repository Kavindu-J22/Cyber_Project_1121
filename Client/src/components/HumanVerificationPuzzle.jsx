import { useState, useEffect, useRef } from 'react';
import { CheckCircle, Shield, ArrowRight } from 'lucide-react';

const HumanVerificationPuzzle = ({ onVerified }) => {
  const [step, setStep] = useState(1); // 1 = image selection, 2 = slider
  const [selectedImages, setSelectedImages] = useState([]);
  const [images, setImages] = useState([]);
  const [sliderPosition, setSliderPosition] = useState(0);
  const [isSliding, setIsSliding] = useState(false);
  const [isVerified, setIsVerified] = useState(false);
  const [attempts, setAttempts] = useState(0);
  const sliderRef = useRef(null);

  // Image categories for selection
  const imageCategories = [
    { type: 'medical', emoji: '🏥', label: 'Hospital' },
    { type: 'medical', emoji: '💊', label: 'Medicine' },
    { type: 'medical', emoji: '🩺', label: 'Stethoscope' },
    { type: 'medical', emoji: '💉', label: 'Syringe' },
    { type: 'other', emoji: '🚗', label: 'Car' },
    { type: 'other', emoji: '🍕', label: 'Pizza' },
    { type: 'other', emoji: '⚽', label: 'Ball' },
    { type: 'other', emoji: '🎮', label: 'Game' },
    { type: 'medical', emoji: '🏥', label: 'Clinic' },
  ];

  // Initialize puzzle on mount
  useEffect(() => {
    initializePuzzle();
  }, []);

  const initializePuzzle = () => {
    // Shuffle images
    const shuffled = [...imageCategories].sort(() => Math.random() - 0.5);
    setImages(shuffled);
    setSelectedImages([]);
    setSliderPosition(0);
    setIsVerified(false);
    setAttempts(0);
    setStep(1);
  };

  const handleImageClick = (index) => {
    if (selectedImages.includes(index)) {
      setSelectedImages(selectedImages.filter(i => i !== index));
    } else {
      setSelectedImages([...selectedImages, index]);
    }
  };

  const handleVerifyImages = () => {
    // Check if all medical images are selected and no other images
    const correctSelections = images
      .map((img, idx) => ({ ...img, idx }))
      .filter(img => img.type === 'medical')
      .map(img => img.idx);

    const isCorrect =
      selectedImages.length === correctSelections.length &&
      selectedImages.every(idx => correctSelections.includes(idx)) &&
      correctSelections.every(idx => selectedImages.includes(idx));

    if (isCorrect) {
      setStep(2); // Move to slider step
    } else {
      setAttempts(prev => prev + 1);
      // Shake effect
      const container = document.getElementById('image-grid');
      if (container) {
        container.classList.add('animate-shake');
        setTimeout(() => {
          container.classList.remove('animate-shake');
        }, 500);
      }
    }
  };

  const handleMouseDown = (e) => {
    setIsSliding(true);
  };

  const handleMouseMove = (e) => {
    if (!isSliding || !sliderRef.current) return;

    const rect = sliderRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  };

  const handleMouseUp = () => {
    if (!isSliding) return;
    setIsSliding(false);

    // Check if slider is at the end (>95%)
    if (sliderPosition > 95) {
      setIsVerified(true);
      setSliderPosition(100);
      onVerified(true);
    } else {
      // Reset slider
      setSliderPosition(0);
      setAttempts(prev => prev + 1);
    }
  };

  const handleTouchStart = () => {
    setIsSliding(true);
  };

  const handleTouchMove = (e) => {
    if (!isSliding || !sliderRef.current) return;

    const touch = e.touches[0];
    const rect = sliderRef.current.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  };

  const handleTouchEnd = () => {
    handleMouseUp();
  };

  useEffect(() => {
    if (isSliding) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isSliding, sliderPosition]);

  return (
    <div className="w-full">
      <div className="mb-4 text-center">
        <div className="flex items-center justify-center mb-2">
          <Shield className="h-6 w-6 text-primary-600 mr-2" />
          <h4 className="text-lg font-semibold text-gray-900">Human Verification</h4>
        </div>
        <p className="text-sm text-gray-600">
          {step === 1 ? 'Select all medical-related images' : 'Slide to verify you are human'}
        </p>
        {attempts > 0 && !isVerified && (
          <p className="text-sm text-orange-600 mt-2">
            ❌ Incorrect! Try again. (Attempts: {attempts})
          </p>
        )}
      </div>

      {/* Step 1: Image Selection */}
      {step === 1 && (
        <div className="mb-6">
          <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm font-medium text-blue-900 text-center">
              🏥 Click on all images that are related to <strong>medical/healthcare</strong>
            </p>
          </div>

          <div
            id="image-grid"
            className="grid grid-cols-3 gap-3 max-w-md mx-auto mb-4"
          >
            {images.map((img, index) => (
              <button
                key={index}
                type="button"
                onClick={() => handleImageClick(index)}
                className={`
                  aspect-square border-4 rounded-lg flex flex-col items-center justify-center
                  transition-all duration-200 cursor-pointer
                  ${selectedImages.includes(index)
                    ? 'border-primary-500 bg-primary-50 shadow-lg scale-95'
                    : 'border-gray-300 bg-white hover:border-primary-300 hover:shadow-md'
                  }
                `}
              >
                <span className="text-4xl mb-1">{img.emoji}</span>
                <span className="text-xs font-medium text-gray-700">{img.label}</span>
                {selectedImages.includes(index) && (
                  <CheckCircle className="absolute top-1 right-1 h-5 w-5 text-primary-600" />
                )}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={handleVerifyImages}
            disabled={selectedImages.length === 0}
            className="w-full py-3 px-4 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            Verify Selection
          </button>
        </div>
      )}

      {/* Step 2: Slider Verification */}
      {step === 2 && !isVerified && (
        <div className="mb-6">
          <div className="mb-3 p-3 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-sm font-medium text-green-900 text-center">
              ✅ Images verified! Now slide to complete verification
            </p>
          </div>

          <div
            ref={sliderRef}
            className="relative w-full h-14 bg-gray-200 rounded-full overflow-hidden cursor-pointer select-none"
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
          >
            {/* Progress Bar */}
            <div
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-primary-400 to-primary-600 transition-all"
              style={{ width: `${sliderPosition}%` }}
            />

            {/* Slider Button */}
            <div
              className="absolute top-1 left-1 h-12 w-12 bg-white rounded-full shadow-lg flex items-center justify-center transition-all"
              style={{
                left: `calc(${sliderPosition}% - 24px)`,
                maxWidth: 'calc(100% - 8px)'
              }}
            >
              <ArrowRight className={`h-6 w-6 ${sliderPosition > 95 ? 'text-green-600' : 'text-primary-600'}`} />
            </div>

            {/* Text */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <span className="text-sm font-medium text-gray-700">
                {sliderPosition < 10 ? 'Slide to verify →' : sliderPosition < 95 ? 'Keep sliding →' : 'Release!'}
              </span>
            </div>
          </div>

          <p className="text-xs text-gray-500 text-center mt-2">
            Drag the slider all the way to the right
          </p>
        </div>
      )}

      {/* Verification Success */}
      {isVerified && (
        <div className="p-4 bg-green-50 border-2 border-green-500 rounded-lg flex items-center justify-center">
          <CheckCircle className="h-6 w-6 text-green-600 mr-2" />
          <span className="text-green-800 font-semibold">✅ Verified! You can now complete registration.</span>
        </div>
      )}
    </div>
  );
};

export default HumanVerificationPuzzle;

