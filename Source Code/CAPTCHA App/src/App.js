import React, { useContext } from 'react';
import logo from './logo.svg';
import './App.css';
import { ThemeContext } from './ThemeContext';

function App() {
  const { darkMode, toggleTheme } = useContext(ThemeContext);

  return (
    <div className="App" data-theme={darkMode ? 'dark' : 'light'}>
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
        <button className="theme-toggle" onClick={toggleTheme}>
          {darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        </button>
      </header>
    </div>
  );
}

export default App;
