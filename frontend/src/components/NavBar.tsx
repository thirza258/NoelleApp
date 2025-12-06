import React from 'react';
import { useNavigate } from 'react-router-dom';

// Updated interface to include the onLogout function
interface NavBarProps {
  selectedTool: string;
  onLogout: () => void; // Callback function for logging out
  onClearDocument: () => void;
}

const NavBar: React.FC<NavBarProps> = ({ selectedTool, onLogout, onClearDocument }) => {
  const navigate = useNavigate();

  return (
    <nav className="w-full h-16 bg-gray-800 text-white flex items-center justify-between px-4 border-b-2 border-white z-10">
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-bold">Noelle</h1>
        {/* This button can be kept or removed depending on your routing setup */}
        <button
          className="px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded-md text-sm"
          onClick={() => navigate("/about")}
        >
          About Us
        </button>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex gap-4">
          <span className="font-bold text-blue-400">
            {selectedTool} Page
          </span>
        </div>

        {/* Button to clear the document when in Document AI tool */}
        {selectedTool === "Document AI" && (
          <button
            className="px-3 py-1 bg-blue-500 hover:bg-blue-600 rounded-md text-sm"
            onClick={onClearDocument}
          >
            Clear Document
          </button>
        )}

        {/* Logout Button */}
        <button
          className="px-3 py-1 bg-red-500 hover:bg-red-600 rounded-md text-sm"
          onClick={onLogout} // Calls the onLogout function passed from App.js
        >
          Logout
        </button>
      </div>
    </nav>
  );
};

export default NavBar;