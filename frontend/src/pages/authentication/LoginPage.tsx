
import React, { useState } from 'react';
import InputField from '../../components/InputField';

interface LoginPageProps {
  onLogin: (username: string, password: string) => void; // Callback for form submission
  switchToSignup: () => void; // Callback to switch to the sign-up view
  error?: string; // Prop to display an error message from the parent
}

const LoginPage: React.FC<LoginPageProps> = ({ onLogin, switchToSignup, error }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [localError, setLocalError] = useState('');

  const handleLogin = () => {
    if (username.trim() === '' || password.trim() === '') {
      setLocalError('Username and password cannot be empty.');
      return;
    }
    setLocalError('');
    onLogin(username, password);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleLogin();
    }
  };

  return (
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
        onChange={(e) => setUsername(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="your_username"
      />
      
      <InputField
        id="password"
        label="Password"
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
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

      <p className="text-center text-sm text-gray-600 mt-6">
        Don't have an account?{' '}
        <span
          onClick={switchToSignup}
          className="text-blue-600 hover:underline cursor-pointer font-semibold"
        >
          Sign Up
        </span>
      </p>
    </div>
  );
};

export default LoginPage;