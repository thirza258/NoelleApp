import React, { useState } from 'react';
import services from '../../services/services';
import InputField from '../../components/InputField';

// Interface defining the props for the LoginPage component
interface LoginPageProps {
  onLoginSuccess: () => void;
  switchToSignup?: () => void;
  error?: string;
}

const LoginPage: React.FC<LoginPageProps> = ({ onLoginSuccess, switchToSignup, error }) => {
  // State variables with explicit string types
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [localError, setLocalError] = useState<string>('');

  const handleLogin = async (): Promise<void> => {
    if (username.trim() === '' || password.trim() === '') {
      setLocalError('Username and password cannot be empty.');
      return;
    }
    setLocalError('');
    try {
      // Assuming services.loginRequest returns a response with a status and optional data/message
      const response = await services.loginRequest(username, password);
      
      if (response.status === 200 && response.data?.token) {
        // Store the token upon successful login
        localStorage.setItem('authToken', response.data.token);
        onLoginSuccess();
      } else {
        setLocalError(response.message || 'Login failed. Please check your credentials.');
      }
    } catch (err: any) { // Catching the error with 'any' type for flexibility
      console.error("Login error:", err);
      setLocalError(err.message || 'An unexpected error occurred.');
    }
  };

  // Event handler with explicit keyboard event type
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === 'Enter') {
      handleLogin();
    }
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-xl w-full max-w-md">
        <h1 className="text-2xl font-bold text-center text-gray-800 mb-6">
          Welcome Back!
        </h1>
        <p className="text-center text-gray-600 mb-6">
          Please enter your credentials to log in.
        </p>
        
        <InputField
          id="username"
          label="Username"
          type="text"
          value={username}
          // Event handler with explicit change event type
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUsername(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="your_username"
        />
        
        <InputField
          id="password"
          label="Password"
          type="password"
          value={password}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="••••••••"
        />

        {(error || localError) && <p className="text-red-500 text-sm text-center mb-4">{error || localError}</p>}
        
        <button
          className="w-full bg-blue-600 text-white py-2 mt-2 rounded-md text-lg font-semibold hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-colors"
          onClick={handleLogin}
        >
          Login
        </button>

        {/* Conditionally render the signup link if the switchToSignup prop is provided */}
        {switchToSignup && (
          <p className="text-center text-sm text-gray-600 mt-6">
            Don't have an account?{' '}
            <span
              onClick={switchToSignup}
              className="text-blue-600 hover:underline cursor-pointer font-semibold"
            >
              Sign Up
            </span>
          </p>
        )}
      </div>
    </div>
  );
};

export default LoginPage;