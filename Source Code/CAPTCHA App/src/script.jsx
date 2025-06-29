import * as React from "react";
import { Box, Card, Flex, Text, Heading, IconButton } from '@radix-ui/themes';
import { ShieldCheck, Warning, Trophy, Moon, Sun } from "@phosphor-icons/react";
import { useState, useCallback } from 'react';
import logo from './logo.svg';
import { getRandomSentence, createSession, saveResponse } from './dataService';

// Custom flat button component that doesn't rely on Radix UI's Button
const FlatButton = ({ children, onClick, isDarkMode }) => {
  const [isHovered, setIsHovered] = React.useState(false);
  
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        backgroundColor: isDarkMode 
          ? isHovered ? '#1a5fd0' : '#1f6feb'  // Dark mode: darker when hovered
          : isHovered ? '#1a73e8' : '#0550ae', // Light mode: lighter when hovered
        color: 'white',
        fontWeight: 600,
        padding: '8px 16px',
        borderRadius: '6px',
        cursor: 'pointer',
        border: 'none',
        outline: 'none',
        transition: 'background-color 0.2s ease',
        display: 'inline-block',
        textAlign: 'center',
        userSelect: 'none',
      }}
    >
      {children}
    </div>
  );
};

function useKV(key, initialValue) {
  // Get from localStorage on init
  const [value, setValue] = React.useState(() => {
    try {
      const storedValue = localStorage.getItem(key);
      // Check if the value exists and is not "undefined" string
      return storedValue && storedValue !== "undefined" 
        ? JSON.parse(storedValue) 
        : initialValue;
    } catch (error) {
      console.error("Error reading from localStorage:", error);
      return initialValue;
    }
  });
  
  // Return a wrapped version of useState's setter function
  const setStoredValue = (newValue) => {
    try {
      // Save state
      setValue(newValue);
      // Save to localStorage
      localStorage.setItem(key, JSON.stringify(newValue));
    } catch (error) {
      console.error("Error writing to localStorage:", error);
    }
  };
  
  return [value, setStoredValue];
}

// Define SDG image URLs, their numbers, and descriptions
const sdgBaseUrl = "https://www.un.org/sustainabledevelopment/wp-content/uploads/2019/08/E-Goal-";
const sdgs = Array.from({ length: 16 }, (_, i) => ({
  url: `${sdgBaseUrl}${String(i + 1).padStart(2, '0')}.png`,
  number: i + 1,
  goalNumber: `Goal ${i + 1}:`,
  description: [
    "End poverty in all its forms everywhere",
    "End hunger, achieve food security and improved nutrition and promote sustainable agriculture",
    "Ensure healthy lives and promote well-being for all at all ages",
    "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all",
    "Achieve gender equality and empower all women and girls",
    "Ensure availability and sustainable management of water and sanitation for all",
    "Ensure access to affordable, reliable, sustainable and modern energy for all",
    "Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all",
    "Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation",
    "Reduce inequality within and among countries",
    "Make cities and human settlements inclusive, safe, resilient and sustainable",
    "Ensure sustainable consumption and production patterns",
    "Take urgent action to combat climate change and its impacts",
    "Conserve and sustainably use the oceans, seas and marine resources for sustainable development",
    "Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss",
    "Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels"
  ][i]
}));

// We're no longer using hard-coded prompts
// Everything will be dynamic from the server

function Captcha({ onToggleTheme, isDarkMode }) {
  // State management
  const [selectedIndex, setSelectedIndex] = React.useState(null);
  const [shuffledIndices, setShuffledIndices] = React.useState([]);
  const [currentPrompt, setCurrentPrompt] = React.useState("");
  const [verificationStatus, setVerificationStatus] = React.useState(null);
  const [hoveredIndex, setHoveredIndex] = React.useState(null);
  const [lastVerificationStatus, setLastVerificationStatus] = React.useState(null);
  const [showNotSure, setShowNotSure] = React.useState(false); // Changed from showWarning to showNotSure
  const [streak, setStreak] = useKV("captcha-streak", 0);
  const [showWelcome, setShowWelcome] = React.useState(true);
  
  // New state for storing dataframe row information and relevant SDGs
  const [currentRowData, setCurrentRowData] = React.useState(null);
  const [relevantSDGIndices, setRelevantSDGIndices] = React.useState([]);
  const [sdgTotals, setSdgTotals] = React.useState([]); // Added sdgTotals state
  const [maxTotal, setMaxTotal] = React.useState(null); // Added maxTotal state
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  
  // Session management
  const [sessionId, setSessionId] = useKV("captcha-session-id", null);
  const [startTime, setStartTime] = React.useState(null); // To track response time
  const [rowIndex, setRowIndex] = React.useState(null); // To track current row index
  
  // Ref for the container to measure available height
  const containerRef = React.useRef(null);
  const [scale, setScale] = React.useState(1);

  // Create a session if one doesn't exist
  const initializeSession = useCallback(async () => {
    if (!sessionId) {
      try {
        const response = await createSession();
        if (response.success) {
          setSessionId(response.session_id);
          console.log("Session created:", response.session_id);
        } else {
          console.error("Failed to create session:", response.error);
        }
      } catch (err) {
        console.error("Error creating session:", err);
      }
    }
  }, [sessionId, setSessionId]);
  
  // Initialize the session when the app starts
  React.useEffect(() => {
    initializeSession();
  }, [initializeSession]);
  
  // Initialize the CAPTCHA
  React.useEffect(() => {
    if (!showWelcome) {
      initializeCaptcha();
    }
  }, [showWelcome]);

  // Handle window resizing and initial scaling
  React.useEffect(() => {
    const updateScale = () => {
      if (!containerRef.current) return;

      const viewportHeight = window.innerHeight;
      const viewportWidth = window.innerWidth;
      const contentHeight = containerRef.current.scrollHeight;
      const contentWidth = containerRef.current.scrollWidth;

      // Calculate scale needed to fit in viewport for both dimensions
      const scaleHeight = viewportHeight / contentHeight;
      const scaleWidth = viewportWidth / contentWidth;
      const newScale = Math.min(1, scaleHeight, scaleWidth);
      setScale(newScale);
    };

    updateScale();
    window.addEventListener('resize', updateScale);

    return () => window.removeEventListener('resize', updateScale);
  }, [currentPrompt, showWelcome]);

  // Function to shuffle the grid and set a new prompt
  const initializeCaptcha = async () => {
    // Fetch a random sentence from the dataframe
    try {
      setLoading(true);
      setError(null);
      const response = await getRandomSentence();
      
      if (response.success) {
        // Store the full row data for future use
        setCurrentRowData(response.row_data);
        
        // Use the sentence as the prompt
        setCurrentPrompt(response.sentence || "Please select an SDG related to this invention");
        
        // Store SDG totals and max total if available
        setSdgTotals(response.sdg_totals || []);
        setMaxTotal(response.max_total || 0);
        
        // Store the row index for tracking responses
        setRowIndex(response.row_index);
        
        // Set start time to track response time
        setStartTime(Date.now());
        
        // Check if we have the SDG vector
        const sdgVector = response.sdg_vector;
        
        // Array to store selected indices
        let selectedIndices = [];
        
        if (sdgVector && Array.isArray(sdgVector)) {
          // The sdgVector contains the numbers of the relevant SDGs (1-based)
          // Convert them to 0-based indices for our array
          const relevantIndices = sdgVector.map(sdgNumber => sdgNumber - 1).filter(index => index >= 0 && index < 16);
          
          // Store the relevant SDG indices for verification later
          setRelevantSDGIndices(relevantIndices);
          
          if (relevantIndices.length > 0) {
            // If we have fewer than 9 relevant SDGs, we'll fill the rest with random ones
            if (relevantIndices.length < 9) {
              // Create array of indices not in relevantIndices
              const remainingIndices = Array.from({ length: 16 }, (_, i) => i)
                .filter(i => !relevantIndices.includes(i));
              
              // Shuffle the remaining indices
              const shuffledRemaining = [...remainingIndices].sort(() => Math.random() - 0.5);
              
              // Take enough to make up 9 total
              const additionalIndices = shuffledRemaining.slice(0, 9 - relevantIndices.length);
              
              // Combine relevant and additional indices
              selectedIndices = [...relevantIndices, ...additionalIndices];
            } else {
              // If we have more than 9, shuffle and take 9
              selectedIndices = [...relevantIndices].sort(() => Math.random() - 0.5).slice(0, 9);
            }
          } else {
            // No relevant SDGs, select 9 random ones
            selectedIndices = Array.from({ length: 9 }, () => 
              Math.floor(Math.random() * 16)
            );
          }
        } else {
          // If no valid SDG vector, select 9 random ones
          selectedIndices = Array.from({ length: 9 }, () => 
            Math.floor(Math.random() * 16)
          );
        }
        
        // Sort the indices to ensure SDGs are displayed in increasing order
        const sortedIndices = [...new Set(selectedIndices)].sort((a, b) => a - b);
        
        // If we somehow have less than 9 (e.g., due to duplicates), add more
        while (sortedIndices.length < 9) {
          const randomIndex = Math.floor(Math.random() * 16);
          if (!sortedIndices.includes(randomIndex)) {
            sortedIndices.push(randomIndex);
          }
        }
        
        // Final sorted list of exactly 9 unique indices in increasing order
        sortedIndices.sort((a, b) => a - b);
        setShuffledIndices(sortedIndices);
      } else {
        console.error("Failed to fetch sentence:", response.error);
        setError("Failed to fetch data from server. Please check if the server is running.");
        
        // Clear any existing data since we want to rely only on the server
        setShuffledIndices([]);
        setCurrentPrompt("Server error. Please try again later.");
        setRelevantSDGIndices([]);
      }
    } catch (err) {
      console.error("Error fetching sentence:", err);
      setError("Cannot connect to server. Please check if the server is running.");
      
      // Clear any existing data since we want to rely only on the server
      setShuffledIndices([]);
      setCurrentPrompt("Server connection failed. Please try again later.");
      setRelevantSDGIndices([]);
    } finally {
      setLoading(false);
    }

    setSelectedIndex(null);
    setVerificationStatus(null);
    setHoveredIndex(null);
  };

  // Handle image selection
  const handleImageClick = (index) => {
    setSelectedIndex(prevIndex => prevIndex === index ? null : index);
    setVerificationStatus(null);
    setShowNotSure(false);
  };

  // Record the response to the database
  const recordResponse = async (responseType, isCorrect, selectedSDGNumber = null) => {
    if (!sessionId) {
      // Try to initialize session if not already done
      await initializeSession();
    }
    
    // Calculate response time
    const responseTimeMs = Date.now() - startTime;
    
    // Prepare response data
    const responseData = {
      session_id: sessionId,
      row_index: rowIndex,
      sentence: currentPrompt,
      selected_sdg: selectedSDGNumber,
      relevant_sdgs: relevantSDGIndices.map(i => i + 1), // Convert to 1-based SDG numbers
      sdg_totals: sdgTotals,
      max_total: maxTotal,
      is_correct: isCorrect,
      response_type: responseType,
      streak: responseType === "correct" ? streak + 1 : responseType === "incorrect" ? 0 : streak,
      response_time_ms: responseTimeMs
    };
    
    try {
      // Save response to the database
      await saveResponse(responseData);
    } catch (err) {
      console.error("Failed to save response:", err);
      // Continue with the app flow even if saving fails
    }
  };

  // Verify the selection
  const handleVerify = () => {
    let responseType = "not_sure";
    let isCorrect = null;
    let selectedSDGNumber = null;
    let newStreak = streak;
    
    if (selectedIndex === null) {
      // User clicked "None" - check max_total value
      if (maxTotal !== null) {
        if (maxTotal < 3) {
          // Correct for max_total < 3
          responseType = "correct";
          isCorrect = true;
          newStreak = streak + 1;
          setVerificationStatus(true);
          setLastVerificationStatus(true);
          setShowNotSure(false);
        } else if (maxTotal === 3) {
          // Not sure for max_total = 3
          responseType = "not_sure";
          isCorrect = null;
          // Keep streak the same
          setVerificationStatus(null);
          setLastVerificationStatus(null);
          setShowNotSure(true);
        } else {
          // Incorrect for max_total > 3
          responseType = "incorrect";
          isCorrect = false;
          newStreak = 0;
          setVerificationStatus(false);
          setLastVerificationStatus(false);
          setShowNotSure(false);
        }
      } else {
        // Default to "Not sure" if maxTotal is not available
        responseType = "not_sure";
        isCorrect = null;
        setVerificationStatus(null);
        setLastVerificationStatus(null);
        setShowNotSure(true);
      }
      
      // Record the response
      recordResponse(responseType, isCorrect);
      
      // Update streak state
      setStreak(newStreak);
      
      initializeCaptcha();
      return;
    }

    // Check if the selected SDG is in our list of relevant SDGs
    const selectedSDG = shuffledIndices[selectedIndex];
    selectedSDGNumber = selectedSDG + 1;  // Convert to 1-based SDG number
    const relevantSDGIndex = relevantSDGIndices.indexOf(selectedSDG);
    
    if (relevantSDGIndex !== -1 && sdgTotals && sdgTotals.length > relevantSDGIndex) {
      // Get the total value for this SDG
      const totalValue = sdgTotals[relevantSDGIndex];
      
      if (totalValue > 1) {
        // Correct for total > 1
        responseType = "correct";
        isCorrect = true;
        newStreak = streak + 1;
        setVerificationStatus(true);
        setLastVerificationStatus(true);
        setShowNotSure(false);
      } else if (totalValue === 1) {
        // Not sure for total = 1
        responseType = "not_sure";
        isCorrect = null;
        // Keep streak the same - newStreak is already set to streak
        setVerificationStatus(null);
        setLastVerificationStatus(null);
        setShowNotSure(true);
      } else {
        // Incorrect for total < 1
        responseType = "incorrect";
        isCorrect = false;
        newStreak = 0;
        setVerificationStatus(false);
        setLastVerificationStatus(false);
        setShowNotSure(false);
      }
    } else {
      // Selected SDG is not in our list of relevant SDGs - incorrect
      responseType = "incorrect";
      isCorrect = false;
      newStreak = 0;
      setVerificationStatus(false);
      setLastVerificationStatus(false);
      setShowNotSure(false);
    }
    
    // Record the response
    recordResponse(responseType, isCorrect, selectedSDGNumber);
    
    // Update streak state
    setStreak(newStreak);

    initializeCaptcha();
  };

  if (showWelcome) {
    return (
      <Flex align="center" justify="center" style={{ height: '100vh', padding: '0 16px' }}>
        <IconButton
          variant="ghost"
          style={{ position: 'absolute', top: '16px', right: '16px' }}
          onClick={onToggleTheme}
          aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDarkMode ? <Sun weight="fill" /> : <Moon weight="fill" />}
        </IconButton>
        
        <Card style={{ maxWidth: '500px', width: '100%' }}>
          <Box p="4">
            <Heading as="h2" size="5" mb="2" style={{ 
              color: isDarkMode ? '#ffffff' : 'var(--color-fg-default)',
              fontWeight: 600
            }}>Welcome!</Heading>
            <Text as="p" mb="4" style={{ 
              color: isDarkMode ? '#ffffff' : 'var(--color-fg-muted)',
              fontWeight: 500
            }}>
              With your responses to the following CAPTCHAs, you are helping us to classify innovations for a sustainable future. Please select your answers carefully. Enjoy!
            </Text>
            <FlatButton 
              onClick={() => setShowWelcome(false)}
              isDarkMode={isDarkMode}
            >
              Start
            </FlatButton>
          </Box>
        </Card>
      </Flex>
    );
  }

  return (
    <Flex align="center" justify="center" style={{ height: '100vh', padding: '0 16px' }}>
      <IconButton
        variant="ghost"
        style={{ position: 'absolute', top: '16px', right: '16px' }}
        onClick={onToggleTheme}
        aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
      >
        {isDarkMode ? <Sun weight="fill" /> : <Moon weight="fill" />}
      </IconButton>
      
      {error && (
        <Card style={{ maxWidth: '500px', width: '100%' }}>
          <Box p="4">
            <Heading as="h2" size="5" mb="2" style={{ 
              color: isDarkMode ? '#ff7b72' : 'var(--color-danger-fg)',
              fontWeight: 600
            }}>Server Connection Error</Heading>
            <Text as="p" mb="4" style={{ 
              color: isDarkMode ? '#ffffff' : 'var(--color-fg-muted)',
              fontWeight: 500
            }}>
              {error}
            </Text>
            <FlatButton 
              onClick={() => {
                setError(null);
                initializeCaptcha();
              }}
              isDarkMode={isDarkMode}
            >
              Retry
            </FlatButton>
          </Box>
        </Card>
      )}
      
      {!error && (
        <div
          ref={containerRef}
          className="scale-container"
          style={{ transform: `scale(${scale})` }}
        >
        <Card>
          <Box p="4" style={{ borderBottom: '1px solid var(--color-border-default)' }}>
            <Heading as="h2" size="5" mb="2" style={{ 
              color: isDarkMode ? '#ffffff' : 'var(--color-fg-default)',
              fontWeight: 600
            }}>{currentPrompt}</Heading>
          </Box>

          <Box p="2">
            <div className="grid-cols-3">
              {shuffledIndices.map((sdgIndex, currentIndex) => {
                // Determine the description text color based on selection state
                const descColor = selectedIndex === currentIndex 
                  ? isDarkMode ? '#24292f' : '#24292f' // Nearly black for both modes when clicked
                  : isDarkMode ? '#ffffff' : '#24292f';
                  
                return (
                  <button
                    key={currentIndex}
                    onClick={() => handleImageClick(currentIndex)}
                    onMouseEnter={() => setHoveredIndex(currentIndex)}
                    onMouseLeave={() => setHoveredIndex(null)}
                    className={selectedIndex === currentIndex ? 'selected' : ''}
                  >
                    {hoveredIndex === currentIndex ? (
                      <Text size="1" style={{ 
                        textAlign: 'center', 
                        overflow: 'auto', 
                        maxHeight: '100%', 
                        padding: '4px',
                        fontWeight: 600, 
                        color: descColor, // Use the determined color
                      }}>
                        <>
                          {sdgs[sdgIndex].goalNumber}
                          <br />
                          {sdgs[sdgIndex].description}
                        </>
                      </Text>
                    ) : (
                      // Always show image when not hovering, regardless of selection state
                      <img
                        src={sdgs[sdgIndex].url}
                        alt={`UN SDG ${sdgs[sdgIndex].number}`}
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      />
                    )}
                  </button>
                );
              })}
            </div>
          </Box>

          <Flex justify="between" align="center" p="4" style={{ borderTop: '1px solid var(--color-border-default)' }}>
            <Flex align="center" gap="4">
              <a href="https://sdgs.un.org/goals" target="_blank" rel="noopener noreferrer">
                <img
                  src="https://cdn.shopify.com/s/files/1/0326/7189/t/67/assets/pf-25eba8d9-aff6-46ff-a351-b9f145af530c--logosdgsnip.png"
                  alt="UN SDG Logo"
                  style={{ 
                    height: '36px', /* Match FlatButton height (includes the padding) */
                    filter: isDarkMode ? 'none' : 'invert(1)'
                  }}
                />
              </a>
              <Flex align="center" gap="2" className="streak-counter">
                <Trophy weight="fill" style={{ color: isDarkMode ? '#a8d4ff' : 'var(--color-accent-fg)' }} />
                <Text style={{ 
                  color: isDarkMode ? '#a8d4ff' : 'var(--color-accent-fg)',
                  fontWeight: 600
                }}>Streak: {streak}</Text>
              </Flex>
            </Flex>

            <Flex align="center" gap="4">
              {(showNotSure || lastVerificationStatus !== null) && (
                <Flex align="center" gap="2">
                  {showNotSure ? (
                    <>
                      <Warning weight="fill" style={{ color: isDarkMode ? '#f0cc6d' : 'var(--color-attention-fg)' }} />
                      <Text style={{ 
                        color: isDarkMode ? '#f0cc6d' : 'var(--color-attention-fg)',
                        fontWeight: 600
                      }}>Not sure</Text>
                    </>
                  ) : lastVerificationStatus ? (
                    <>
                      <ShieldCheck weight="fill" style={{ color: isDarkMode ? '#6ae57f' : 'var(--color-success-fg)' }} />
                      <Text style={{ 
                        color: isDarkMode ? '#6ae57f' : 'var(--color-success-fg)',
                        fontWeight: 600
                      }}>Correct!</Text>
                    </>
                  ) : (
                    <>
                      <Warning weight="fill" style={{ color: isDarkMode ? '#ff7b72' : 'var(--color-danger-fg)' }} />
                      <Text style={{ 
                        color: isDarkMode ? '#ff7b72' : 'var(--color-danger-fg)',
                        fontWeight: 600
                      }}>Incorrect</Text>
                    </>
                  )}
                </Flex>
              )}
              <FlatButton 
                onClick={handleVerify}
                isDarkMode={isDarkMode}
              >
                {selectedIndex === null ? 'None' : 'Verify'}
              </FlatButton>
            </Flex>
          </Flex>
        </Card>
      </div>
      )}
    </Flex>
  );
}

export default Captcha;