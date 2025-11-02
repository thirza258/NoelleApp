
import React, { useState } from 'react';
import InputField from '../../components/InputField';

interface SignupPageProps {
  onSignup: (username: string, email: string, password: string) => void;
  switchToLogin: () => void;
  error?: string;
}

const SignupPage: React.FC<SignupPageProps> = ({ onSignup, switchToLogin, error }) => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [localError, setLocalError] = useState('');

  const handleSignup = () => {
    if (!username || !email || !password || !confirmPassword) {
      setLocalError('All fields are required.');
      return;
    }
    if (password !== confirmPassword) {
      setLocalError('Passwords do not match.');
      return;
    }
    // Simple email validation
    if (!/\S+@\S+\.\S+/.test(email)) {
        setLocalError('Please enter a valid email address.');
        return;
    }

    setLocalError('');
    onSignup(username, email, password);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSignup();
    }
  };

  return (
    <div className="bg-white p-8 rounded-lg shadow-xl w-full max-w-md">
      <h1 className="text-2xl font-bold text-center text-gray-800 mb-6">
        Create an Account
      </h1>
      <p className="text-center text-gray-600 mb-6">
        Join us by filling out the information below.
      </p>
      
      <InputField
        id="username"
        label="Username"
        type="text"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        placeholder="Choose a username"
      />
      <InputField
        id="email"
        label="Email"
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="you@example.com"
      />
      <InputField
        id="password"
        label="Password"
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="••••••••"
      />
      <InputField
        id="confirmPassword"
        label="Confirm Password"
        type="password"
        value={confirmPassword}
        onChange={(e) => setConfirmPassword(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="••••••••"
      />

      {(error || localError) && <p className="text-red-500 text-sm text-center mb-4">{error || localError}</p>}

      <button
        className="w-full bg-green-600 text-white py-2 mt-2 rounded-md text-lg font-semibold hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 transition-colors"
        onClick={handleSignup}
      >
        Sign Up
      </button>

      <p className="text-center text-sm text-gray-600 mt-6">
        Already have an account?{' '}
        <span
          onClick={switchToLogin}
          className="text-blue-600 hover:underline cursor-pointer font-semibold"
        >
          Login
        </span>
      </p>
    </div>
  );
};

export default SignupPage;