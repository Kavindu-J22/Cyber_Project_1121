import { useState, useEffect } from 'react';
import { CheckCircle, XCircle } from 'lucide-react';

const HumanVerificationPuzzle = ({ onVerified }) => {
  const [boxes, setBoxes] = useState([]);
  const [puzzlePiece, setPuzzlePiece] = useState(null);
  const [draggedPiece, setDraggedPiece] = useState(null);
  const [isVerified, setIsVerified] = useState(false);
  const [attempts, setAttempts] = useState(0);

  // Initialize puzzle on mount
  useEffect(() => {
    initializePuzzle();
  }, []);

  const initializePuzzle = () => {
    // Create 6 boxes in random positions
    const newBoxes = [];
    const gridSize = 3; // 3x3 grid
    const positions = [];
    
    // Generate all possible positions
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        positions.push({ row: i, col: j });
      }
    }
    
    // Shuffle positions
    const shuffled = positions.sort(() => Math.random() - 0.5);
    
    // Take first 6 positions for boxes
    for (let i = 0; i < 6; i++) {
      newBoxes.push({
        id: i,
        position: shuffled[i],
        filled: false,
        isCorrect: i === 0 // First box is the correct one
      });
    }
    
    setBoxes(newBoxes);
    
    // Create puzzle piece
    setPuzzlePiece({
      id: 'puzzle',
      shape: 'puzzle'
    });
    
    setIsVerified(false);
    setAttempts(0);
  };

  const handleDragStart = (e) => {
    setDraggedPiece(puzzlePiece);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e, boxId) => {
    e.preventDefault();
    
    if (!draggedPiece) return;
    
    const box = boxes.find(b => b.id === boxId);
    
    if (box.isCorrect) {
      // Correct box!
      const updatedBoxes = boxes.map(b => 
        b.id === boxId ? { ...b, filled: true } : b
      );
      setBoxes(updatedBoxes);
      setIsVerified(true);
      setPuzzlePiece(null);
      onVerified(true);
    } else {
      // Wrong box
      setAttempts(prev => prev + 1);
      // Shake animation or feedback
      const updatedBoxes = boxes.map(b => 
        b.id === boxId ? { ...b, shake: true } : b
      );
      setBoxes(updatedBoxes);
      
      setTimeout(() => {
        setBoxes(boxes.map(b => ({ ...b, shake: false })));
      }, 500);
    }
    
    setDraggedPiece(null);
  };

  const handleDragEnd = () => {
    setDraggedPiece(null);
  };

  return (
    <div className="w-full">
      <div className="mb-4 text-center">
        <h4 className="text-lg font-semibold text-gray-900 mb-2">Human Verification</h4>
        <p className="text-sm text-gray-600">
          Drag the puzzle piece to the correct box to verify you're human
        </p>
        {attempts > 0 && !isVerified && (
          <p className="text-sm text-orange-600 mt-2">
            Incorrect! Try again. (Attempts: {attempts})
          </p>
        )}
      </div>

      {/* Puzzle Grid */}
      <div className="mb-6 bg-gray-50 p-6 rounded-lg border-2 border-gray-200">
        <div className="grid grid-cols-3 gap-4 max-w-md mx-auto">
          {boxes.map((box) => (
            <div
              key={box.id}
              onDragOver={handleDragOver}
              onDrop={(e) => handleDrop(e, box.id)}
              className={`
                aspect-square border-4 border-dashed rounded-lg flex items-center justify-center
                transition-all duration-200
                ${box.filled ? 'border-green-500 bg-green-100' : 'border-gray-300 bg-white'}
                ${box.shake ? 'animate-shake' : ''}
                ${draggedPiece ? 'hover:border-primary-500 hover:bg-primary-50' : ''}
              `}
              style={{
                gridRow: box.position.row + 1,
                gridColumn: box.position.col + 1
              }}
            >
              {box.filled && (
                <div className="w-16 h-16 bg-gradient-to-br from-primary-400 to-primary-600 rounded-lg flex items-center justify-center shadow-lg">
                  <svg className="w-12 h-12 text-white" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"/>
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Draggable Puzzle Piece */}
      {puzzlePiece && !isVerified && (
        <div className="flex justify-center">
          <div
            draggable
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
            className="w-20 h-20 bg-gradient-to-br from-primary-400 to-primary-600 rounded-lg flex items-center justify-center cursor-move shadow-xl hover:shadow-2xl transition-all transform hover:scale-105"
          >
            <svg className="w-14 h-14 text-white" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"/>
            </svg>
          </div>
        </div>
      )}

      {/* Verification Status */}
      {isVerified && (
        <div className="mt-4 p-4 bg-green-50 border-2 border-green-500 rounded-lg flex items-center justify-center">
          <CheckCircle className="h-6 w-6 text-green-600 mr-2" />
          <span className="text-green-800 font-semibold">Verified! You can now complete registration.</span>
        </div>
      )}
    </div>
  );
};

export default HumanVerificationPuzzle;

