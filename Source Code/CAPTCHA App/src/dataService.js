/**
 * Data service for interacting with the Python backend
 */

// Base API URL - updated port to 5001
const API_URL = window.location.origin;

/**
 * Fetch a sample of data from the dataframe
 * @param {number} limit - Number of records to fetch
 * @returns {Promise<Object>} - Response containing the data
 */
export const fetchData = async (limit = 10) => {
  try {
    const response = await fetch(`${API_URL}/data?limit=${limit}`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

/**
 * Fetch paginated data from the dataframe
 * @param {number} page - Page number, starting from 1
 * @param {number} pageSize - Number of records per page
 * @returns {Promise<Object>} - Response containing the paginated data
 */
export const getPaginatedData = async (page = 1, pageSize = 50) => {
  try {
    const response = await fetch(`${API_URL}/data/paginated?page=${page}&size=${pageSize}`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching paginated data:', error);
    throw error;
  }
};

/**
 * Filter the dataframe based on specific criteria
 * @param {Object} filters - Key-value pairs for filtering, where key is column name
 * @returns {Promise<Object>} - Response containing the filtered data
 */
export const filterData = async (filters) => {
  try {
    const response = await fetch(`${API_URL}/data/filter`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(filters),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error filtering data:', error);
    throw error;
  }
};

/**
 * Get the list of columns in the dataframe
 * @returns {Promise<Object>} - Response containing the column names
 */
export const getColumns = async () => {
  try {
    const response = await fetch(`${API_URL}/columns`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching columns:', error);
    throw error;
  }
};

/**
 * Get a summary of the dataframe
 * @returns {Promise<Object>} - Response containing the summary information
 */
export const getSummary = async () => {
  try {
    const response = await fetch(`${API_URL}/summary`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching summary:', error);
    throw error;
  }
};

/**
 * Create a new user session
 * @returns {Promise<Object>} - Response containing the session ID
 */
export const createSession = async () => {
  try {
    const response = await fetch(`${API_URL}/session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error creating session:', error);
    throw error;
  }
};

/**
 * Save a user's response to the database
 * @param {Object} responseData - Data about the user's response
 * @returns {Promise<Object>} - Response indicating success or failure
 */
export const saveResponse = async (responseData) => {
  try {
    const response = await fetch(`${API_URL}/response`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(responseData),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error saving response:', error);
    throw error;
  }
};

/**
 * Get statistics about user responses
 * @param {string} sessionId - Optional session ID to filter by
 * @returns {Promise<Object>} - Response containing statistics
 */
export const getStats = async (sessionId) => {
  try {
    const url = sessionId 
      ? `${API_URL}/stats?session_id=${sessionId}` 
      : `${API_URL}/stats`;
      
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching stats:', error);
    throw error;
  }
};

/**
 * Fetch a random sentence and its associated row data from the dataframe
 * @returns {Promise<Object>} - Response containing the sentence and full row data
 */
export const getRandomSentence = async () => {
  try {
    const response = await fetch(`${API_URL}/random_sentence`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching random sentence:', error);
    throw error;
  }
};
